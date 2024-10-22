import argparse
import gc

import torch
from tabulate import tabulate
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
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LMDataset:
    def __init__(self, args, tokenizer, split):
        self.args = args
        self.tokenizer = tokenizer
        logger.info(f"Loading dataset: {args.dataset_name} ({args.dataset_config_name}) - {split}")
        self.data = load_from_disk(args.dataset_name)[split] if os.path.exists(args.dataset_name) else load_dataset(args.dataset_name, split=split)
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


import torch
from torch.cuda.amp import autocast, GradScaler
import gc


class KnowledgeDistillation:
    def __init__(self, args, local_rank):
        self.args = args
        self.local_rank = local_rank
        self.device = f"cuda:{local_rank}"
        self.scaler = GradScaler()

        # Initialize tokenizer with streaming
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.teacher_model_name,
            use_fast=True,
            streaming=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load teacher model with memory optimizations
        self.teacher_model = self._load_teacher_model()

        # Load student model with memory optimizations
        self.student_model = self._load_student_model()

    def _load_teacher_model(self):
        # Load teacher model with memory efficient settings
        model = AutoModelForCausalLM.from_pretrained(
            self.args.teacher_model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            max_memory={0: "20GB"},  # Adjust based on GPU memory
            offload_folder="offload_teacher"
        )
        model.config.use_cache = False  # Disable KV cache
        return model

    def _load_student_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.args.student_model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            max_memory={0: "20GB"},  # Adjust based on GPU memory
            offload_folder="offload_student"
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        return DDP(model, device_ids=[self.local_rank])

    def train(self):
        train_dataset = self._create_dataset("train")
        eval_dataset = self._create_dataset("validation")

        train_loader = self._create_dataloader(train_dataset, shuffle=True)
        eval_loader = self._create_dataloader(eval_dataset, shuffle=False)

        optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=0.01
        )

        for epoch in range(self.args.num_epochs):
            self._train_epoch(train_loader, optimizer, epoch)
            if self.local_rank == 0:
                eval_loss = self.evaluate(eval_loader)
                logger.info(f"Epoch {epoch + 1}, Eval Loss: {eval_loss:.4f}")

            # Clear cache after each epoch
            self._clear_memory()

    def _train_epoch(self, train_loader, optimizer, epoch):
        self.student_model.train()

        for batch_idx, batch in enumerate(train_loader):
            try:
                with autocast():  # Enable automatic mixed precision
                    loss = self._process_batch(batch)

                # Scale loss and backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()

                # Regular memory cleanup
                if batch_idx % 5 == 0:
                    self._clear_memory()

            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                self._clear_memory()
                continue

    def _process_batch(self, batch):
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

        with torch.no_grad(), autocast():
            teacher_outputs = self.teacher_model(**batch)

        with autocast():
            student_outputs = self.student_model(**batch)
            loss = self.distillation_loss(
                student_outputs.logits,
                teacher_outputs.logits,
                batch['input_ids'],
                self.args.temperature
            )

        return loss

    def _clear_memory(self):
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(self.teacher_model, 'cpu'):
            self.teacher_model.cpu()
            torch.cuda.empty_cache()
            self.teacher_model.to(self.device)

    def _create_dataset(self, split):
        return LMDataset(
            self.args,
            self.tokenizer,
            split,
            max_samples=self.args.dataset_num_samples
        )

    def _create_dataloader(self, dataset, shuffle):
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=self.local_rank,
            shuffle=shuffle
        )

        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            sampler=sampler,
            collate_fn=dataset.collate_fn,
            pin_memory=True,
            num_workers=2
        )

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

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, args):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    torch.manual_seed(args.seed)

    try:
        kd = KnowledgeDistillation(args, rank)
        kd.train()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        cleanup()

def print_arguments(args):
    args_dict = vars(args)
    table_data = [[k, v] for k, v in args_dict.items()]
    print("\nKnowledge Distillation Arguments:")
    print(tabulate(table_data, headers=["Argument", "Value"], tablefmt="grid"))
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model_name", default="defog/llama-3-sqlcoder-8b", type=str)
    parser.add_argument("--student_model_name", default="meta-llama/Llama-3.2-3B-Instruct", type=str)
    parser.add_argument("--dataset_name", default="lamini/spider_text_to_sql", type=str)
    parser.add_argument("--dataset_num_samples", type=int, default=None, help="Number of samples to process. Use None for full dataset.")
    parser.add_argument("--dataset_config_name", default=None, type=str)
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--num_epochs", default=3, type=int)
    parser.add_argument("--warmup_steps", default=500, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--output_dir", default="distilled_model", type=str)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--num_samples", default=100, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use")

    args = parser.parse_args()
    print_arguments(args)
    logger.info(f"Starting script with args: {args}")

    world_size = args.world_size
    torch.multiprocessing.spawn(main, args=(world_size, args), nprocs=world_size, join=True)