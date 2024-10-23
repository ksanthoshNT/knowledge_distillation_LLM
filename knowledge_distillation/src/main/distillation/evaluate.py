# evaluate.py
import torch
import logging
import json
from torch.utils.data import DataLoader
from typing import Dict, Any
from pathlib import Path

from knowledge_distillation.src.main.fixingclass_transformers_distillation_main import KnowledgeDistillationModelConfig, \
    KnowledgeDistillationModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self, model, eval_dataset, batch_size: int = 8):
        self.model = model
        self.eval_dataset = eval_dataset
        self.batch_size = batch_size

    def evaluate(self) -> Dict[str, Any]:
        """
        Comprehensive evaluation of the model
        """
        self.model.student.eval()
        metrics = {
            'loss': 0.0,
            'teacher_student_similarity': 0.0,
            'per_layer_similarity': {},
            'prediction_samples': []
        }

        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.model.collate_fn
        )

        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_dataloader):
                try:
                    batch_metrics = self._evaluate_batch(batch)

                    # Accumulate metrics
                    for key, value in batch_metrics.items():
                        if key not in metrics:
                            metrics[key] = 0.0
                        metrics[key] += value

                    num_batches += 1

                    if batch_idx % 10 == 0:
                        logger.info(f"Processed {batch_idx} evaluation batches")

                except Exception as e:
                    logger.error(f"Error in evaluation batch {batch_idx}: {str(e)}")
                    continue

        # Average the metrics
        for key in metrics:
            if isinstance(metrics[key], (int, float)):
                metrics[key] /= num_batches

        return metrics

    def _evaluate_batch(self, batch) -> Dict[str, Any]:
        """
        Evaluate a single batch and return metrics
        """
        batch = {k: v.to(self.model.config.device) for k, v in batch.items()}

        # Get outputs from both teacher and student
        teacher_outputs = self.model.teacher(**batch, output_hidden_states=True)
        student_outputs = self.model.student(**batch, output_hidden_states=True)

        # Calculate metrics
        batch_metrics = {}

        # Loss
        batch_metrics['loss'] = self.model.black_box_distillation(
            student_outputs.logits,
            teacher_outputs.logits,
            batch['attention_mask']
        ).item()

        # Teacher-student output similarity
        batch_metrics['teacher_student_similarity'] = torch.nn.functional.cosine_similarity(
            student_outputs.logits.view(-1),
            teacher_outputs.logits.view(-1),
            dim=0
        ).mean().item()

        return batch_metrics

    def save_metrics(self, metrics: Dict[str, Any], output_dir: str):
        """
        Save evaluation metrics to a file
        """
        output_path = Path(output_dir) / "evaluation_metrics.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open('w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved evaluation metrics to {output_path}")


def main():
    from datasets import load_dataset

    # Load model and config
    config = KnowledgeDistillationModelConfig(
        teacher_model_name="defog/llama-3-sqlcoder-8b",
        student_model_name="aspenita/llama-3-sqlcoder-8b-AWQ",
    )

    # Load model from checkpoint
    model = KnowledgeDistillationModel(config)
    checkpoint = torch.load("checkpoints/best_model.pt")
    model.student.load_state_dict(checkpoint['model_state_dict'])

    # Load evaluation dataset
    dataset = load_dataset("lamini/spider_text_to_sql")
    eval_dataset = dataset['validation']

    # Create evaluator and run evaluation
    evaluator = ModelEvaluator(model, eval_dataset)
    metrics = evaluator.evaluate()

    # Save metrics
    evaluator.save_metrics(metrics, "evaluation_results")


if __name__ == "__main__":
    main()