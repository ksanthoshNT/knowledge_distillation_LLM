import json
import os
import re
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PretrainedConfig
from torch.utils.data import DataLoader, Dataset
from typing import Optional
from dataclasses import dataclass
import logging
from transformers import get_linear_schedule_with_warmup

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
            torch_dtype=torch.float32,
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

        # Add epsilon to avoid log(0)
        epsilon = 1e-8
        teacher_probs = torch.clamp(teacher_probs, min=epsilon)

        # Calculate KL divergence
        logger.debug(f"Teacher log shape: {teacher_probs.shape}")
        logger.debug(f"Student log shape: {student_log_probs.shape}")
        logger.debug(f"Attent Mask : {attention_mask is not None}")

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
{output}
"""
    return {
        "input": text,
        "output": output
    }


@dataclass
class TrainingConfig:
    evaluation_steps: int = 500  # Evaluate every N steps
    save_steps: int = 1000  # Save every N steps
    warmup_steps: int = 100  # Number of warmup steps for learning rate
    early_stopping_patience: int = 3  # Number of evaluations to wait before early stopping
    early_stopping_threshold: float = 0.01  # Minimum improvement required


class DistillationTrainer:
    def __init__(
            self,
            model,
            train_dataset,
            eval_dataset,
            training_config: TrainingConfig,
            checkpoint_dir: str = "checkpoints"
    ):
        self.model = model
        self.config = model.config
        self.training_config = training_config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.checkpoint_dir = checkpoint_dir

        self.optimizer = torch.optim.AdamW(
            self.model.student.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )

        # Add scheduler
        num_training_steps = len(train_dataset) // self.config.batch_size * self.config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.training_config.warmup_steps,
            num_training_steps=num_training_steps
        )

        self.best_eval_loss = float('inf')
        self.no_improvement_count = 0
        os.makedirs(checkpoint_dir, exist_ok=True)

    def collate_fn(self, batch):
        texts = [example['input'] for example in batch]

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

        global_step = 0
        logger.info("Starting training...")

        for epoch in range(self.config.num_epochs):
            self.model.student.train()
            epoch_loss = 0
            num_batches = 0
            epoch_start_time = time.perf_counter()

            for batch_idx, batch in enumerate(train_dataloader):
                try:
                    # Training step
                    loss = self._training_step(batch)
                    epoch_loss += loss
                    num_batches += 1
                    global_step += 1

                    # Logging
                    if global_step % 100 == 0:
                        logger.info(
                            f"Epoch: {epoch + 1}/{self.config.num_epochs}, "
                            f"Step: {global_step}, "
                            f"Loss: {loss:.4f}"
                        )

                    # Evaluation
                    if global_step % self.training_config.evaluation_steps == 0:
                        eval_metrics = self._run_evaluation()

                        # Early stopping check
                        if self._check_early_stopping(eval_metrics['eval_loss']):
                            logger.info("Early stopping triggered")
                            return

                    # Save checkpoint
                    if global_step % self.training_config.save_steps == 0:
                        self._save_checkpoint(epoch, global_step)

                except Exception as e:
                    logger.error(f"Error in batch {batch_idx}: {str(e)}")
                    continue

            # End of epoch logging
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            epoch_elapsed_time =  time.perf_counter() - epoch_start_time
            logger.info(f"Epoch {epoch + 1} completed {epoch_elapsed_time:.2f}s. Average loss: {avg_epoch_loss:.4f}")
            torch.cuda.empty_cache()  # Clear cache at the end of each epoch

    def _training_step(self, batch):
        batch = {k: v.to(self.config.device) for k, v in batch.items()}

        outputs = self.model(**batch)
        loss = outputs["loss"]

        if not torch.isfinite(loss):
            logger.warning("Skipping batch due to invalid loss")
            return 0.0

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.student.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()  # Clear cache after gradient update

        return loss.item()

    def _save_best_model(self):
        """Save the best performing model based on evaluation loss"""
        best_model_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        torch.save({
            'model_state_dict': self.model.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'eval_loss': self.best_eval_loss
        }, best_model_path)
        logger.info(f"Saved best model with eval loss {self.best_eval_loss:.4f} to {best_model_path}")

    def _check_early_stopping(self, eval_loss):
        if eval_loss < self.best_eval_loss - self.training_config.early_stopping_threshold:
            self.best_eval_loss = eval_loss
            self.no_improvement_count = 0
            self._save_best_model()
        else:
            self.no_improvement_count += 1

        return self.no_improvement_count >= self.training_config.early_stopping_patience

    def _save_checkpoint(self, epoch, global_step):
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_epoch_{epoch}_step_{global_step}.pt"
        )
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': self.model.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_eval_loss': self.best_eval_loss
        }, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def _run_evaluation(self):
        self.model.student.eval()
        eval_metrics = {}

        try:
            eval_loss = self._evaluate_loss()
            eval_metrics['eval_loss'] = eval_loss
            logger.info(f"Evaluation metrics: {eval_metrics}")

            self.model.student.train()
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")

        return eval_metrics

    def _evaluate_loss(self):
        eval_loss_start_time = time.perf_counter()
        total_loss = 0
        num_batches = 0

        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )

        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(self.config.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs["loss"].item()
                num_batches += 1
                torch.cuda.empty_cache()  # Clear cache after each evaluation batch

        eval_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        eval_loss_elapsed_time = time.perf_counter() - eval_loss_start_time
        logger.info(f"Evaluation completed in {eval_loss_elapsed_time:.2f}s. Loss: {eval_loss:.4f}")
        return eval_loss


if __name__ == '__main__':
    # Initialize configuration
    config = KnowledgeDistillationModelConfig(
        # teacher_model_name="meta-llama/Llama-3.2-1B-Instruct",  # Changed model
        teacher_model_name="defog/llama-3-sqlcoder-8b",  # Changed model
        teacher_model_torch_dtype="float32",
        # student_model_name="meta-llama/Llama-3.2-1B-Instruct",  # Changed model
        student_model_name="aspenita/llama-3-sqlcoder-8b-AWQ",  # Changed model
        student_model_torch_dtype="float32",
        distillation_type="black_box",  # Using combined distillation
        temperature=2.0,
        alpha=0.5,
        batch_size=8,  # Reduced batch size
        num_epochs=3,
        max_length=256
    )

    # Create model
    model = KnowledgeDistillationModel(config)

    # Load dataset
    dataset = load_dataset("lamini/spider_text_to_sql")

    logger.info(f'Dataset : {dataset}')

    dataset = dataset.map(lambda x: transform_text(x))

    logger.info(f"Sample: {dataset['train'][0]}")

    # Create trainer
    trainer = DistillationTrainer(
        model=model,
        training_config=TrainingConfig(
            evaluation_steps=500,
            save_steps=2500,
            warmup_steps=500,
            early_stopping_patience=3,
            early_stopping_threshold=0.005

        ),
        train_dataset=dataset['train'].select(range(5000)),
        eval_dataset=dataset['validation'].select(range(1000))
    )

    # Train model
    trainer.train()