import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
import torch.nn.functional as F
import deepspeed
import logging
import sys
import os
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LMDataset(Dataset):
    def __init__(self, args, tokenizer, split):
        self.args = args
        self.tokenizer = tokenizer
        logger.info(f"Loading dataset: {args.dataset_name} ({args.dataset_config_name}) - {split}")
        self.data = load_dataset(args.dataset_name, args.dataset_config_name, split=split, streaming=args.streaming)
        if not args.streaming:
            self.data = self.data.shuffle(seed=args.seed)
        self.max_length = args.max_length
        logger.info(f"Dataset loaded. Streaming: {args.streaming}, Max length: {self.max_length}")

    def __len__(self):
        return len(self.data) if not self.args.streaming else self.args.num_samples

    def __getitem__(self, idx):
        try:
            item = next(iter(self.data)) if self.args.streaming else self.data[idx]
            encoded = self.tokenizer(item['text'], truncation=True, max_length=self.max_length, return_tensors='pt')
            return {k: v.squeeze(0) for k, v in encoded.items()}
        except Exception as e:
            logger.error(f"Error in __getitem__: {e}")
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
        self.teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher_model_name, torch_dtype=torch.bfloat16)
        self.teacher_model.config.pad_token_id = self.tokenizer.pad_token_id

        logger.info("Loading student model...")
        self.student_model = AutoModelForCausalLM.from_pretrained(args.student_model_name, torch_dtype=torch.bfloat16)
        self.student_model.config.pad_token_id = self.tokenizer.pad_token_id

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def train(self):
        logger.info("Starting training process...")
        train_dataset = LMDataset(self.args, self.tokenizer, "train")
        eval_dataset = LMDataset(self.args, self.tokenizer, "validation")

        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, collate_fn=train_dataset.collate_fn)
        eval_loader = DataLoader(eval_dataset, batch_size=self.args.batch_size, collate_fn=eval_dataset.collate_fn)

        optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=self.args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                                    num_training_steps=len(train_loader) * self.args.num_epochs)

        ds_config = {
            "train_micro_batch_size_per_gpu": self.args.batch_size,
            "gradient_accumulation_steps": 2,
            "fp16": {"enabled": True},
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                      "device": "cpu",
                      "pin_memory": True
                  }
            },

            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.args.learning_rate,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.01
                }
            }
        }

        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=self.student_model,
            model_parameters=self.student_model.parameters(),
            config=ds_config
        )

        # Remove the separate optimizer creation
        # optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=self.args.learning_rate)

        logger.info("Initializing DeepSpeed...")
        model_engine, optimizer, _, _ = deepspeed.initialize(model=self.student_model,
                                                             model_parameters=self.student_model.parameters(),
                                                             config=ds_config)

        self.teacher_model.eval().to(self.device)
        model_engine.train()

        for epoch in range(self.args.num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.args.num_epochs}")
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.args.num_epochs}")):
                try:
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    with torch.no_grad():
                        teacher_outputs = self.teacher_model(**batch)

                    student_outputs = model_engine(**batch)

                    loss = self.distillation_loss(student_outputs.logits, teacher_outputs.logits, batch['input_ids'],
                                                  self.args.temperature)

                    model_engine.backward(loss)
                    model_engine.step()
                    scheduler.step()

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
        model_engine.save_checkpoint(self.args.output_dir)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model_name", default="meta-llama/Llama-3.2-3B-Instruct", type=str)
    parser.add_argument("--student_model_name", default="meta-llama/Llama-3.2-1B-Instruct", type=str)
    parser.add_argument("--dataset_name", default="wikitext", type=str)
    parser.add_argument("--dataset_config_name", default="wikitext-2-raw-v1", type=str)
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--num_epochs", default=3, type=int)
    parser.add_argument("--warmup_steps", default=500, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--output_dir", default="distilled_model", type=str)
    parser.add_argument("--streaming", type=bool , default=True)
    parser.add_argument("--num_samples", default=100, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    logger.info(f"Starting script with args: {args}")
    torch.manual_seed(args.seed)
    try:
        kd = KnowledgeDistillation(args)
        kd.train()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()