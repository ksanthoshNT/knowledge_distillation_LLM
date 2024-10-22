from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
import torch.nn.functional as F
from typing import Dict, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class DistillationTrainingArguments(TrainingArguments):
    """
    Extends HuggingFace's TrainingArguments with distillation-specific parameters
    """
    temperature: float = 2.0
    alpha: float = 0.5
    distillation_type: str = "combined"  # "black_box", "white_box", or "combined"


class SimpleDistillationModel(PreTrainedModel):
    """
    A simple wrapper that combines teacher and student for distillation
    """

    def __init__(self, teacher, student, config=None):
        super().__init__(config or student.config)
        self.teacher = teacher
        self.student = student

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            **kwargs
    ):
        # Get teacher outputs (no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True
            )

        # Get student outputs
        student_outputs = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True
        )

        # Calculate losses
        loss = None
        if labels is not None:
            # Regular cross-entropy loss
            lm_loss = F.cross_entropy(
                student_outputs.logits.view(-1, student_outputs.logits.size(-1)),
                labels.view(-1)
            )

            # Distillation losses
            distil_loss = self.compute_distillation_loss(
                student_outputs,
                teacher_outputs
            )

            # Combine losses
            loss = (self.config.alpha * distil_loss +
                    (1 - self.config.alpha) * lm_loss)

        return {"loss": loss} if loss is not None else student_outputs

    def compute_distillation_loss(self, student_outputs, teacher_outputs):
        """Compute distillation loss based on configured type"""
        if self.config.distillation_type == "black_box":
            return self.black_box_loss(
                student_outputs.logits,
                teacher_outputs.logits
            )
        elif self.config.distillation_type == "white_box":
            return self.white_box_loss(
                student_outputs,
                teacher_outputs
            )
        else:  # combined
            bb_loss = self.black_box_loss(
                student_outputs.logits,
                teacher_outputs.logits
            )
            wb_loss = self.white_box_loss(
                student_outputs,
                teacher_outputs
            )
            return 0.5 * (bb_loss + wb_loss)

    def black_box_loss(self, student_logits, teacher_logits):
        """Simple KL divergence loss on softmax outputs"""
        temp = self.config.temperature
        soft_targets = F.softmax(teacher_logits / temp, dim=-1)
        student_log_probs = F.log_softmax(student_logits / temp, dim=-1)

        return F.kl_div(
            student_log_probs,
            soft_targets,
            reduction='batchmean'
        ) * (temp ** 2)

    def white_box_loss(self, student_outputs, teacher_outputs):
        """Simple MSE loss on hidden states"""
        # Match last hidden states
        s_hidden = student_outputs.hidden_states[-1]
        t_hidden = teacher_outputs.hidden_states[-1]

        # Simple interpolation if sizes don't match
        if s_hidden.shape != t_hidden.shape:
            s_hidden = F.interpolate(
                s_hidden.unsqueeze(1),
                size=t_hidden.shape[-1],
                mode='linear'
            ).squeeze(1)

        return F.mse_loss(s_hidden, t_hidden)


def train_distillation_model(
        teacher_model_name: str,
        student_model_name: str,
        train_dataset,
        eval_dataset=None,
        output_dir="distilled-model",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        learning_rate=5e-5,
        distillation_type="combined",
        temperature=2.0,
        alpha=0.5
):
    """Simple function to train a distillation model"""
    # Load models
    teacher = AutoModelForCausalLM.from_pretrained(teacher_model_name)
    student = AutoModelForCausalLM.from_pretrained(student_model_name)
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

    # Create distillation model
    model = SimpleDistillationModel(teacher, student)

    # Set distillation config
    model.config.temperature = temperature
    model.config.alpha = alpha
    model.config.distillation_type = distillation_type

    # Training arguments
    training_args = DistillationTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        temperature=temperature,
        alpha=alpha,
        distillation_type=distillation_type,
        save_strategy="epoch",
        evaluation_strategy="epoch" if eval_dataset else "no",
        logging_dir=f"{output_dir}/logs"
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )

    # Train and save
    trainer.train()
    model.student.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model.student, tokenizer


# Example usage:
if __name__ == "__main__":
    # Example of how to use the distillation training
    from datasets import load_dataset

    # Load dataset
    dataset = load_dataset("lamini/spider_text_to_sql")  # Replace with your dataset

    # Train distillation model
    student_model, tokenizer = train_distillation_model(
        teacher_model_name="large-sql-model",
        student_model_name="small-sql-model",
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        output_dir="distilled-sql-model",
        num_train_epochs=3,
        distillation_type="combined"
    )