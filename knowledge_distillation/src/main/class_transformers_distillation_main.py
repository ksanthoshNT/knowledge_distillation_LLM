import os
import re
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PretrainedConfig
from torch.utils.data import DataLoader, Dataset
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KnowledgeDistillationModelConfig(PretrainedConfig):
    def __init__(
            self,
            teacher_model_name: str,
            student_model_name: str,
            student_model_torch_dtype: str = 'bfloat16',
            teacher_model_torch_dtype: str = 'bfloat16',
            distillation_type: str = "combined",
            temperature: float = 2.0,
            alpha: float = 0.5,
            learning_rate: float = 5e-5,
            batch_size: int = 4,  # Reduced batch size
            num_epochs: int = 3,
            max_length: int = 512,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            **kwargs
    ):
        super().__init__(**kwargs)
        self.teacher_model_name = teacher_model_name
        self.student_model_name = student_model_name
        self.student_model_torch_dtype = student_model_torch_dtype
        self.teacher_model_torch_dtype = teacher_model_torch_dtype
        self.distillation_type = distillation_type
        self.temperature = temperature
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.max_length = max_length
        self.device = device


class KnowledgeDistillationModel(PreTrainedModel):
    config_class = KnowledgeDistillationModelConfig

    def __init__(self, config: KnowledgeDistillationModelConfig):
        super().__init__(config)

        # Initialize tokenizer with padding token
        self.tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load teacher model
        self.teacher = AutoModelForCausalLM.from_pretrained(
            config.teacher_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            use_cache=False
        )

        # Load student model
        self.student = AutoModelForCausalLM.from_pretrained(
            config.student_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            use_cache=False
        )

        # Enable gradient checkpointing
        self.student.gradient_checkpointing_enable()

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

    def black_box_distillation(self, student_logits, teacher_logits, attention_mask=None):
        """Modified black box distillation with proper masking"""
        # Scale logits by temperature
        student_logits_temp = student_logits / self.config.temperature
        teacher_logits_temp = teacher_logits / self.config.temperature

        # Convert to log probabilities and probabilities
        student_log_probs = F.log_softmax(student_logits_temp, dim=-1)
        teacher_probs = F.softmax(teacher_logits_temp, dim=-1)

        # Calculate KL divergence
        loss = F.kl_div(
            student_log_probs.view(-1, student_log_probs.size(-1)),
            teacher_probs.view(-1, teacher_probs.size(-1)),
            reduction='none'
        ).sum(-1)

        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.view(-1).float()
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()

        return loss * (self.config.temperature ** 2)

    def white_box_distillation(self, student_hidden, teacher_hidden, attention_mask=None):
        """Modified white box distillation with proper masking"""
        # Ensure hidden states have the same dimensions
        if student_hidden.shape != teacher_hidden.shape:
            student_hidden = F.interpolate(
                student_hidden.unsqueeze(1),
                size=teacher_hidden.shape[-1],
                mode='linear'
            ).squeeze(1)

        # Calculate MSE loss
        loss = F.mse_loss(student_hidden, teacher_hidden, reduction='none').mean(dim=-1)

        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.float()
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()

        return loss

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get teacher outputs
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

        # Get student outputs
        student_outputs = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Calculate losses based on distillation type
        if self.config.distillation_type == "black_box":
            loss = self.black_box_distillation(
                student_outputs.logits,
                teacher_outputs.logits,
                attention_mask
            )
        elif self.config.distillation_type == "white_box":
            loss = self.white_box_distillation(
                student_outputs.hidden_states[-1],
                teacher_outputs.hidden_states[-1],
                attention_mask
            )
        else:  # combined
            bb_loss = self.black_box_distillation(
                student_outputs.logits,
                teacher_outputs.logits,
                attention_mask
            )
            wb_loss = self.white_box_distillation(
                student_outputs.hidden_states[-1],
                teacher_outputs.hidden_states[-1],
                attention_mask
            )
            loss = self.config.alpha * bb_loss + (1 - self.config.alpha) * wb_loss

        return {"loss": loss, "logits": student_outputs.logits}


def transform_text(example):
    input_string = example['input']
    output = example['output']

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

    text = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

        Generate a SQL query to answer this question: `{question}`

        DDL statements:
        {transformed_schema}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        The following SQL query best answers the question `{question}`:
        ```sql
        {output}"""
    return text



class DistillationTrainer:
    def __init__(
            self,
            model: KnowledgeDistillationModel,
            train_dataset: Dataset,
            eval_dataset: Optional[Dataset] = None,
            checkpoint_dir: str = "checkpoints",
            checkpoint_frequency: int = 1
    ):
        self.model = model
        self.config = model.config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_frequency = checkpoint_frequency

        # Initialize optimizer with gradient clipping
        self.optimizer = torch.optim.AdamW(
            self.model.student.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )

        # Add gradient clipping
        self.grad_clip = 1.0

        os.makedirs(checkpoint_dir, exist_ok=True)

    def collate_fn(self, batch):
        texts = [transform_text(example) for example in batch]

        # Tokenize all texts in batch
        encodings = self.model.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        }

    def train(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )

        logger.info("Starting training...")
        for epoch in range(self.config.num_epochs):
            self.model.student.train()
            total_loss = 0
            num_batches = 0

            for batch_idx, batch in enumerate(train_dataloader):
                try:
                    # Move batch to device
                    batch = {k: v.to(self.config.device) for k, v in batch.items()}

                    # Forward pass
                    outputs = self.model(**batch)
                    loss = outputs["loss"]

                    # Check if loss is valid
                    if not torch.isfinite(loss):
                        logger.warning(f"Skipping batch {batch_idx} due to invalid loss")
                        continue

                    # Backward pass
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.student.parameters(),
                        self.grad_clip
                    )

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    total_loss += loss.item()
                    num_batches += 1

                    if batch_idx % 10 == 0:
                        avg_loss = total_loss / (num_batches) if num_batches > 0 else 0
                        logger.info(
                            f"Epoch: {epoch + 1}/{self.config.num_epochs}, "
                            f"Batch: {batch_idx}, "
                            f"Loss: {loss.item():.4f}, "
                            f"Avg Loss: {avg_loss:.4f}"
                        )

                except Exception as e:
                    logger.error(f"Error in batch {batch_idx}: {str(e)}")
                    continue

            if num_batches > 0:
                avg_loss = total_loss / num_batches
                logger.info(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")


if __name__ == '__main__':
    # Initialize configuration
    config = KnowledgeDistillationModelConfig(
        teacher_model_name="meta-llama/Llama-3.2-1B-Instruct",  # Changed model
        student_model_name="meta-llama/Llama-3.2-1B-Instruct",  # Changed model
        student_model_torch_dtype="float32",
        teacher_model_torch_dtype="float32",
        distillation_type="combined",  # Using combined distillation
        temperature=2.0,
        alpha=0.5,
        batch_size=2  # Reduced batch size
    )

    # Create model
    model = KnowledgeDistillationModel(config)

    # Load dataset
    dataset = load_dataset("lamini/spider_text_to_sql")

    # Create trainer
    trainer = DistillationTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation']
    )

    # Train model
    trainer.train()