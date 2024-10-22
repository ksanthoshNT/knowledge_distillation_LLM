import os
import re

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel, PretrainedConfig
)
from torch.utils.data import DataLoader, Dataset
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KnowledgeDistillationModelConfig(PretrainedConfig):
    """Configuration class for knowledge distillation"""

    def __init__(
            self,
            teacher_model_name: str,
            student_model_name: str,
            student_model_torch_dtype:str = 'bfloat16',
            teacher_model_torch_dtype:str = 'bfloat16',
            distillation_type: str = "combined",  # "black_box", "white_box", or "combined"
            temperature: float = 2.0,
            alpha: float = 0.5,
            learning_rate: float = 5e-5,
            batch_size: int = 8,
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
    """Main knowledge distillation model combining teacher and student"""
    config_class = KnowledgeDistillationModelConfig

    def __init__(self, config: KnowledgeDistillationModelConfig):
        super().__init__(config)

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.teacher_dtype = self._get_dtype(config.teacher_model_torch_dtype)
        self.student_dtype = self._get_dtype(config.student_model_torch_dtype)

        # Initialize models
        self.teacher = AutoModelForCausalLM.from_pretrained(
            config.teacher_model_name,
            torch_dtype=self.teacher_dtype
        ).to(config.device)

        self.student = AutoModelForCausalLM.from_pretrained(
            config.student_model_name,
            torch_dtype=self.teacher_dtype
        ).to(config.device)

        # Freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

    def _get_dtype(self,dtype:str):
        if dtype == "float16":
            return torch.float16
        elif dtype == "float32":
            return torch.float32
        elif dtype == "bfloat16":
            return torch.bfloat16
        else:
            raise ValueError("Invalid dtype")

    def black_box_distillation(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """Compute black box distillation loss using KL divergence"""
        # Apply temperature scaling
        soft_targets = F.softmax(teacher_logits / self.config.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.config.temperature, dim=-1)

        loss = F.kl_div(
            student_log_probs,
            soft_targets,
            reduction='batchmean'
        ) * (self.config.temperature ** 2)

        return loss

    def white_box_distillation(self, student_hidden: torch.Tensor, teacher_hidden: torch.Tensor) -> torch.Tensor:
        """Compute white box distillation loss using hidden states"""
        # Adapt dimensions if needed
        if student_hidden.shape != teacher_hidden.shape:
            student_hidden = F.interpolate(
                student_hidden.unsqueeze(1),
                size=teacher_hidden.shape[-1],
                mode='linear'
            ).squeeze(1)

        return F.mse_loss(student_hidden, teacher_hidden)

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

        # Calculate distillation loss based on type
        if self.config.distillation_type == "black_box":
            loss = self.black_box_distillation(
                student_outputs.logits,
                teacher_outputs.logits
            )
        elif self.config.distillation_type == "white_box":
            loss = self.white_box_distillation(
                student_outputs.hidden_states[-1],
                teacher_outputs.hidden_states[-1]
            )
        else:  # combined
            bb_loss = self.black_box_distillation(
                student_outputs.logits,
                teacher_outputs.logits
            )
            wb_loss = self.white_box_distillation(
                student_outputs.hidden_states[-1],
                teacher_outputs.hidden_states[-1]
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
    """Trainer class for handling the distillation process"""

    def __init__(
            self,
            model: KnowledgeDistillationModel,
            train_dataset: Dataset,
            eval_dataset: Optional[Dataset] = None,
            checkpoint_dir: str = "checkpoints",  # Add this parameter
            checkpoint_frequency: int = 1
    ):
        self.model = model
        self.config = model.config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_frequency = checkpoint_frequency
        self.best_eval_loss = float('inf')

        self.optimizer = torch.optim.AdamW(
            self.model.student.parameters(),
            lr=self.config.learning_rate
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

    def collate_fn(self, batch):
        input_ids = []
        attention_mask = []

        for example in batch:
            text = transform_text(example)
            # Assuming the dataset has a 'text' field - modify this based on your dataset structure
            encoded = self.model.tokenizer(
                text,  # Change this to match your dataset field
                padding='max_length',
                truncation=True,
                max_length=self.config.max_length,
                return_tensors='pt'
            )
            input_ids.append(encoded['input_ids'])
            attention_mask.append(encoded['attention_mask'])

        return {
            'input_ids': torch.cat(input_ids, dim=0),
            'attention_mask': torch.cat(attention_mask, dim=0)
        }

    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save a checkpoint of the model"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.model.config
        }

        # Save regular checkpoint
        if epoch % self.checkpoint_frequency == 0:
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f'checkpoint_epoch_{epoch}.pt'
            )
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Save best model if this is the best loss
        if is_best:
            best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_model_path)
            logger.info(f"Saved best model to {best_model_path}")


    def train(self):
        """Main training loop"""
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

            for batch_idx, batch in enumerate(train_dataloader):
                # Move batch to device
                batch = {k: v.to(self.config.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask')
                )
                loss = outputs["loss"]

                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()

                if batch_idx % 100 == 0:
                    logger.info(
                        f"Epoch: {epoch + 1}/{self.config.num_epochs}, "
                        f"Batch: {batch_idx}, "
                        f"Loss: {loss.item():.4f}"
                    )

            avg_loss = total_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

            if self.eval_dataset:
                eval_loss = self.evaluate()
                logger.info(f"Evaluation loss: {eval_loss:.4f}")
                is_best = eval_loss < self.best_eval_loss
                if is_best:
                    self.best_eval_loss = eval_loss
            else:
                is_best = False

                # Save checkpoint
            self.save_checkpoint(
                epoch=epoch + 1,
                loss=avg_loss,
                is_best=is_best
            )

    def evaluate(self):
        """Evaluation loop"""
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            collate_fn=self.collate_fn
        )

        self.model.student.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(self.config.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs["loss"].item()

        return total_loss / len(eval_dataloader)

    def save_model(self, output_dir: str):
        """Save the distilled student model"""
        self.model.student.save_pretrained(output_dir)
        self.model.tokenizer.save_pretrained(output_dir)

if __name__ == '__main__':

    # Initialize configuration
    config = KnowledgeDistillationModelConfig(
        teacher_model_name="meta-llama/Llama-3.2-1B-Instruct",
        student_model_name="meta-llama/Llama-3.2-1B-Instruct",
        student_model_torch_dtype="float32",
        teacher_model_torch_dtype="float32",
        distillation_type="white_box",
        temperature=2.0,
        alpha=0.5
    )

    # Create model
    model = KnowledgeDistillationModel(config)
    dataset = load_dataset("lamini/spider_text_to_sql")  # Replace with your dataset

    # Create trainer
    trainer = DistillationTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation']
    )

    # Train model
    trainer.train()

    # Save model
    trainer.save_model("distilled-model")