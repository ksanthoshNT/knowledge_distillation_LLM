# Add these imports at the top of your file
import os
from datetime import datetime
from typing import Optional

import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from knowledge_distillation.src.main.class_transformers_distillation_main import KnowledgeDistillationModel
import logging

logger = logging.getLogger(__name__)

class DistillationTrainer:
    def __init__(
            self,
            model: KnowledgeDistillationModel,
            train_dataset: Dataset,
            eval_dataset: Optional[Dataset] = None,
            checkpoint_dir: str = "checkpoints",  # Add this parameter
            checkpoint_frequency: int = 1  # Add this parameter (save every N epochs)
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

        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

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
        """Main training loop with checkpointing"""
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
                # Existing training code remains the same
                batch = {k: v.to(self.config.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs["loss"]
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

            # Evaluate and save checkpoint
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

    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)
        self.model.student.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        logger.info(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")
        return epoch, loss