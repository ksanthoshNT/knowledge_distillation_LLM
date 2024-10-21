import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import torch.nn.functional as F
import logging
import sys
import os
import psutil
import re

from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LMDataset:
    def __init__(self, args, tokenizer, split):
        self.args = args
        self.tokenizer = tokenizer
        logger.info(f"Loading dataset: {args.dataset_name} ({args.dataset_config_name}) - {split}")
        self.data = load_from_disk(args.dataset_name)[split] if os.path.exists(args.dataset_name) else load_dataset(args.dataset_name,split=split)
        # logger.info(f"Dataset features: {next(iter(self.data)).keys()}")
        # logger.info(f"Sample data item: {next(iter(self.data))}")
        if not args.streaming:
            self.data = self.data.shuffle(seed=args.seed)
        self.max_length = args.max_length
        if args.dataset_num_samples is not None:
            self.data = self.data.take(args.dataset_num_samples)
        logger.info(
            f"Dataset loaded. Streaming: {args.streaming}, Max length: {self.max_length}, Samples: {'All' if args.num_samples is None else args.num_samples}"
        )

    def __len__(self):
        return self.args.dataset_num_samples if self.args.streaming else len(self.data)

    def __getitem__(self, idx):
        try:
            item = next(iter(self.data)) if self.args.streaming else self.data[idx]

            # Extract the SQL query from the 'output' field
            sql_query = item['output'].strip()

            # Parse the 'input' field
            input_text = item['input']

            def extract_and_transform(input_string):
                def transform_schema(schema):
                    tables = re.split(r'\n\s*\n', schema)
                    create_statements = []
                    foreign_keys = []

                    for table in tables:
                        lines = table.strip().split('\n')
                        table_name = lines[0].strip(':')
                        columns = lines[1:]

                        create_statement = f"CREATE TABLE {table_name} (\n"
                        for column in columns:
                            parts = column.split('[')
                            col_name = parts[0].strip()
                            col_type = parts[1].split(']')[0].strip()

                            if col_type == 'INT':
                                col_type = 'INTEGER'
                            elif col_type == 'TEXT':
                                col_type = 'VARCHAR(100)'

                            create_statement += f"  {col_name} {col_type}"

                            if 'primary_key' in column:
                                create_statement += " PRIMARY KEY"

                            create_statement += ",\n"

                            if '=' in column:
                                fk_parts = column.split('=')
                                fk_table, fk_column = fk_parts[1].strip().split('.')
                                foreign_keys.append(
                                    f"-- {table_name}.{col_name} can be joined with {fk_table}.{fk_column}")

                        create_statement = create_statement.rstrip(',\n') + "\n);\n"
                        create_statements.append(create_statement)

                    return "\n".join(create_statements) + "\n" + "\n".join(foreign_keys)

                # Extract the database schema
                schema_pattern = r"Here is a database schema:(.*?)Please write me a SQL statement"
                schema_match = re.search(schema_pattern, input_string, re.DOTALL)
                db_schema = schema_match.group(1).strip() if schema_match else "Schema not found"

                # Extract the question
                question_pattern = r"Please write me a SQL statement that answers the following question: (.*?)\s*\[/INST\]"
                question_match = re.search(question_pattern, input_string, re.DOTALL)
                question = question_match.group(1).strip() if question_match else "Question not found"

                # Transform the schema
                transformed_schema = transform_schema(db_schema)

                return transformed_schema, question

            create_table_statements, user_question = extract_and_transform(input_text)

            # Construct the text in the specified format
            text = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

    Generate a SQL query to answer this question: `{user_question}`

    DDL statements:
    {create_table_statements}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    The following SQL query best answers the question `{user_question}`:
    ```sql
    {sql_query}"""
            encoded = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors='pt')
            return {k: v.squeeze(0) for k, v in encoded.items()}
        except Exception as e:
            logger.error(f"Error in __getitem__: {e}")
            logger.error(f"Item causing error: {item}")
            raise

    def collate_fn(self, batch):
        try:
            return self.tokenizer.pad(batch, padding=True, return_tensors='pt')
        except Exception as e:
            logger.error(f"Error in collate_fn: {e}")
            raise


class KnowledgeDistillation:
    def __init__(self, args):
        self.args = args
        logger.info(f"Initializing KnowledgeDistillation with args: {args}")
        self.tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Set padding token to: {self.tokenizer.pad_token}")

        logger.info("Loading teacher model...")
        self.teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher_model_name,
                                                                  device_map="auto",
                                                                  torch_dtype=torch.bfloat16)
        self.teacher_model.config.pad_token_id = self.tokenizer.pad_token_id
        # self.teacher_model.gradient_checkpointing_enable()
        self.teacher_model.eval()
        logger.info("Gradient checkpointing enabled for teacher model")

        logger.info("Loading student model...")
        self.student_model = AutoModelForCausalLM.from_pretrained(args.student_model_name,
                                                                  device_map="auto",
                                                                  torch_dtype=torch.bfloat16)
        self.student_model.config.pad_token_id = self.tokenizer.pad_token_id
        self.student_model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled for student model")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def train(self):
        logger.info("Starting training process...")
        train_dataset = LMDataset(self.args, self.tokenizer, "train")
        eval_dataset = LMDataset(self.args, self.tokenizer, "validation")
        print(f"Validation dataset size: {len(eval_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, collate_fn=train_dataset.collate_fn)
        eval_loader = DataLoader(eval_dataset, batch_size=self.args.batch_size, collate_fn=eval_dataset.collate_fn)

        optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=self.args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                                    num_training_steps=len(train_loader) * self.args.num_epochs)

        self.teacher_model.eval().to(self.device)
        self.student_model.train().to(self.device)

        for epoch in range(self.args.num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.args.num_epochs}")
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.args.num_epochs}")):
                try:
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    with torch.no_grad():
                        teacher_outputs = self.teacher_model(**batch)

                    student_outputs = self.student_model(**batch)

                    loss = self.distillation_loss(student_outputs.logits, teacher_outputs.logits, batch['input_ids'],
                                                  self.args.temperature)

                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    if batch_idx % 100 == 0:
                        logger.info(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                        self.log_memory_usage()

                except Exception as e:
                    logger.error(f"Error in training loop: {e}")
                    self.log_memory_usage()
                    raise

            eval_loss = self.evaluate(eval_loader)
            logger.info(f"Epoch {epoch + 1}/{self.args.num_epochs}, Eval Loss: {eval_loss:.4f}")

        logger.info("Saving model...")
        self.student_model.save_pretrained(self.args.output_dir)
        logger.info(f"Model saved to {self.args.output_dir}")

    def distillation_loss(self, student_logits, teacher_logits, labels, temperature):
        try:
            soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
            loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                soft_targets,
                reduction='batchmean'
            ) * (temperature ** 2)
            return loss
        except Exception as e:
            logger.error(f"Error in distillation_loss: {e}")
            raise

    def evaluate(self, eval_loader):
        logger.info("Starting evaluation...")
        self.student_model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                try:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    student_outputs = self.student_model(**batch)
                    loss = F.cross_entropy(student_outputs.logits.view(-1, student_outputs.logits.size(-1)),
                                           batch['input_ids'].view(-1))
                    total_loss += loss.item()
                except Exception as e:
                    logger.error(f"Error in evaluation loop: {e}")
                    raise
        return total_loss / len(eval_loader)

    def log_memory_usage(self):
        process = psutil.Process(os.getpid())
        logger.info(f"CPU Memory Usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        if torch.cuda.is_available():
            logger.info(f"GPU Memory Usage: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")

    def save_model(self, output_dir="distilled_model"):
        print(f"Saving distilled model to {output_dir}...")
        self.student_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Distilled model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model_name", default="meta-llama/Llama-3.2-3B-Instruct", type=str)
    parser.add_argument("--student_model_name", default="meta-llama/Llama-3.2-1B-Instruct", type=str)
    parser.add_argument("--dataset_name",
                        default="/home/data_science/project_files/santhosh/lamini__spider_text_to_sql", type=str)
    parser.add_argument("--dataset_num_samples", type=int, default=None,
                        help="Number of samples to process. Use None for full dataset.")
    parser.add_argument("--dataset_config_name", default=None, type=str)
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--num_epochs", default=3, type=int)
    parser.add_argument("--warmup_steps", default=500, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--output_dir", default="distilled_model", type=str)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_samples", default=100, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    logger.info(f"Starting script with args: {args}")
    torch.manual_seed(args.seed)
    try:
        kd = KnowledgeDistillation(args)
        kd.train()
        kd.save_model()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()