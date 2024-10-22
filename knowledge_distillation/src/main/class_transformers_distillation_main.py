import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel
)
from torch.utils.data import DataLoader, Dataset
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KnowledgeDistillationModelConfig:
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
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
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

    def __init__(self, config: KnowledgeDistillationModelConfig):
        self.config = config

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.teacher_dtype = self._get_dtype(config.teacher_model_torch_dtype)
        self.student_dtype = self._get_dtype(config.student_model_torch_dtype)

        # Initialize models
        self.teacher = AutoModelForCausalLM.from_pretrained(
            config.teacher_model_name,
            output_hidden_states=True,
            torch_dtype=self.teacher_dtype
        ).to(config.device)

        self.student = AutoModelForCausalLM.from_pretrained(
            config.student_model_name,
            output_hidden_states=True,
            torch_dtype=self.teacher_dtype
        ).to(config.device)

        # Freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        super().__init__(self.student.config)

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

        loss = None
        if labels is not None:
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

        return {"loss": loss} if loss is not None else student_outputs


class DistillationTrainer:
    """Trainer class for handling the distillation process"""

    def __init__(
            self,
            model: KnowledgeDistillationModel,
            train_dataset: Dataset,
            eval_dataset: Optional[Dataset] = None
    ):
        self.model = model
        self.config = model.config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.optimizer = torch.optim.AdamW(
            self.model.student.parameters(),
            lr=self.config.learning_rate
        )

    def train(self):
        """Main training loop"""
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        logger.info("Starting training...")
        for epoch in range(self.config.num_epochs):
            self.model.student.train()
            total_loss = 0

            for batch_idx, batch in enumerate(train_dataloader):
                # Move batch to device
                batch = {k: v.to(self.config.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch)
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

    def evaluate(self):
        """Evaluation loop"""
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size
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
        student_model_torch_dtype="bfloat16",
        teacher_model_torch_dtype="bfloat16",
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