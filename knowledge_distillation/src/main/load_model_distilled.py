import logging
import os

import torch

from knowledge_distillation.src.main.fixingclass_transformers_distillation_main import KnowledgeDistillationModelConfig, \
    KnowledgeDistillationModel


def load_checkpoint_model(checkpoint_path: str, config: KnowledgeDistillationModelConfig):
    """
    Load a saved checkpoint model.

    Args:
        checkpoint_path (str): Path to the checkpoint file
        config (KnowledgeDistillationModelConfig): Configuration for the model

    Returns:
        tuple: (loaded_model, optimizer, epoch, metrics)
    """
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    # Create a new model instance with the same configuration
    model = KnowledgeDistillationModel(config)

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Load the model state
    model.student.load_state_dict(checkpoint['model_state_dict'])

    # Create optimizer (same as in training)
    optimizer = torch.optim.AdamW(
        model.student.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01
    )

    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Get epoch and metrics
    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']

    # Set model to evaluation mode
    model.student.eval()

    logging.info(f"Successfully loaded checkpoint from epoch {epoch}")
    logging.info(f"Metrics at checkpoint: {metrics}")

    return model, optimizer, epoch, metrics


if __name__ == '__main__':
    # First create the same config used during training
    config = KnowledgeDistillationModelConfig(
        teacher_model_name="defog/llama-3-sqlcoder-8b",
        teacher_model_torch_dtype="float32",
        student_model_name="aspenita/llama-3-sqlcoder-8b-AWQ",
        student_model_torch_dtype="float32",
        distillation_type="black_box",
        temperature=2.0,
        alpha=0.5,
        batch_size=4
    )

    # Load the checkpoint
    checkpoint_path = "/home/data_science/project_files/santhosh/knowledge_distillation_LLM/knowledge_distillation/src/main/checkpoints/checkpoint_epoch_3.pt"
    loaded_model, optimizer, epoch, metrics = load_checkpoint_model(checkpoint_path, config)
