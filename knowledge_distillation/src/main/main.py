import argparse

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, DataCollatorWithPadding
from datasets import load_dataset, IterableDataset
from tqdm import tqdm
from torch.nn import functional as F
from evaluate import load


class KnowledgeDistillation:
    def __init__(self, teacher_model_name, dataset_name, dataset_config=None):
        self.teacher_model_name = teacher_model_name
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.teacher_model = None
        self.student_model = None
        self.tokenizer = None
        self.dataset = None
        self.dataloader = None

    def load_teacher_model(self, use_8bit=False):
        print("Loading teacher model...")
        if use_8bit:
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                self.teacher_model_name,
                device_map="auto",
                load_in_8bit=True
            )
        else:
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                self.teacher_model_name,
                device_map="auto",
                torch_dtype=torch.float16
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.teacher_model_name)

        # Set padding token if it's not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.teacher_model.config.pad_token_id = self.tokenizer.eos_token_id

        for param in self.teacher_model.parameters():
            param.requires_grad = False
        print("Teacher model loaded successfully.")

    def load_student_model(self, student_model_name=None, target_size=None, precision="float16", load_weights=True):
        print("Loading student model...")

        if precision == "float16":
            dtype = torch.float16
        elif precision == "float32":
            dtype = torch.float32
        else:
            raise ValueError("Precision must be either 'float16' or 'float32'")

        if student_model_name:
            if load_weights:
                self.student_model = AutoModelForCausalLM.from_pretrained(
                    student_model_name,
                    device_map="auto",
                    torch_dtype=dtype
                )
            else:
                config = AutoConfig.from_pretrained(student_model_name)
                self.student_model = AutoModelForCausalLM.from_config(config).to(dtype)
        elif target_size:
            # Create a pruned version of the teacher model
            self.student_model = self._prune_model(self.teacher_model, target_size, dtype)
        else:
            raise ValueError("Either student_model_name or target_size must be provided")

        # Ensure student model has the same pad token as the teacher
        self.student_model.config.pad_token_id = self.teacher_model.config.pad_token_id

        # Enable gradient computation for student model
        for param in self.student_model.parameters():
            param.requires_grad = True

        print(f"Student model loaded successfully with {precision} precision.")

    def _prune_model(self, model, target_size, dtype):
        # This is a placeholder for model pruning logic
        # In a real implementation, you would use techniques like weight pruning,
        # layer removal, or model compression to reduce the model size
        print(f"Pruning model to {target_size} parameters")
        return model.to(dtype)  # Return the pruned model in the specified dtype

    def load_dataset(self, streaming=True):
        print("Loading dataset...")
        self.dataset = load_dataset(self.dataset_name, self.dataset_config, split="train", streaming=streaming)
        print("Dataset loaded successfully.")

    def prepare_data(self, batch_size=2, max_length=128, num_samples=None):
        print("Preparing data...")

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )

        def preprocess_function(examples):
            processed = tokenize_function(examples)
            processed["labels"] = processed["input_ids"].clone()
            return processed

        if isinstance(self.dataset, IterableDataset):
            self.dataset = self.dataset.map(preprocess_function, batched=True, remove_columns=self.dataset.column_names)
            if num_samples is not None:
                self.dataset = self.dataset.take(num_samples)
        else:
            if num_samples is not None:
                self.dataset = self.dataset.select(range(min(num_samples, len(self.dataset))))
            self.dataset = self.dataset.map(preprocess_function, batched=True, remove_columns=self.dataset.column_names)

        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        print(f"Data preparation completed. Using {num_samples if num_samples is not None else 'all'} samples.")

    def train(self, num_epochs=3, learning_rate=1e-5, temperature=0.5, max_grad_norm=1.0):
        print("Starting training...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if next(self.student_model.parameters()).device != device:
            self.student_model.to(device)

        optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=learning_rate)

        self.student_model.gradient_checkpointing_enable()  # Enable gradient checkpointing

        for epoch in range(num_epochs):
            self.student_model.train()
            total_loss = 0
            num_batches = 0

            for batch in tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                batch = {k: v.to(device) for k, v in batch.items()}

                with torch.no_grad():
                    teacher_outputs = self.teacher_model(**batch)

                student_outputs = self.student_model(**batch)

                # Check for NaN in logits
                if torch.isnan(student_outputs.logits).any() or torch.isnan(teacher_outputs.logits).any():
                    print("NaN detected in model outputs")
                    continue

                student_logits = F.log_softmax(student_outputs.logits / temperature, dim=-1)
                teacher_logits = F.softmax(teacher_outputs.logits / temperature, dim=-1)

                # Check for NaN after softmax
                if torch.isnan(student_logits).any() or torch.isnan(teacher_logits).any():
                    print("NaN detected after softmax/log_softmax")
                    continue

                loss = F.kl_div(student_logits, teacher_logits, reduction="batchmean") * (temperature ** 2)

                if torch.isnan(loss).any():
                    print(f"NaN loss detected. Student logits: {student_logits}, Teacher logits: {teacher_logits}")
                    continue

                total_loss += loss.item()
                num_batches += 1

                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_grad_norm)

                optimizer.step()
                optimizer.zero_grad()

                # Print some sample logits and loss for debugging
                if num_batches % 10 == 0:
                    print(f"Sample student logits: {student_logits[0, :5]}")
                    print(f"Sample teacher logits: {teacher_logits[0, :5]}")
                    print(f"Current loss: {loss.item()}")

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

            # Evaluation
            eval_loss, perplexity = self._evaluate()

            print(f"Evaluation Loss: {eval_loss:.4f}")
            print(f"Perplexity: {perplexity:.4f}")

        print("Training completed.")

    def _evaluate(self):
        self.student_model.eval()
        total_loss = 0
        num_batches = 0
        all_logits = []
        all_labels = []
        device = next(self.student_model.parameters()).device

        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Evaluating"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.student_model(**batch)

                loss = outputs.loss
                total_loss += loss.item()
                num_batches += 1

                all_logits.append(outputs.logits.detach().cpu())
                all_labels.append(batch['labels'].detach().cpu())

        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        # Compute perplexity
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        perplexity = torch.exp(F.cross_entropy(all_logits.view(-1, all_logits.size(-1)), all_labels.view(-1)))

        return avg_loss, perplexity.item()

    def save_model(self, output_dir="distilled_model"):
        print(f"Saving distilled model to {output_dir}...")
        self.student_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Distilled model saved to {output_dir}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Knowledge Distillation")
    parser.add_argument("--teacher_model_name", default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Name of the teacher model")
    parser.add_argument("--dataset_name", default="wikitext", help="Name of the dataset")
    parser.add_argument("--dataset_config_name", default="wikitext-2-raw-v1", help="Configuration name of the dataset")
    parser.add_argument("--student_model_name", default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Name of the student model")
    parser.add_argument("--student_precision", default="float16", help="Precision to use for the student model")
    parser.add_argument("--load_student_weights", action="store_true",
                        help="Whether to load the weights for the student model")
    parser.add_argument("--streaming", action="store_true", help="Whether to use streaming for dataset loading")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for data preparation")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length for data preparation")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to use in data preparation")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for knowledge distillation")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for training")
    return parser.parse_args()


def main():
    args = parse_arguments()
    print(args.streaming)
    exit()

    kd = KnowledgeDistillation(args.teacher_model_name, args.dataset_name, args.dataset_config_name)
    kd.load_teacher_model()  # Load teacher in 8-bit quantization
    kd.load_student_model(
        student_model_name=args.student_model_name,
        precision=args.student_precision,
        load_weights=args.load_student_weights
    )
    kd.load_dataset(streaming=args.streaming)
    kd.prepare_data(batch_size=args.batch_size, max_length=args.max_length, num_samples=args.num_samples)
    kd.train(num_epochs=args.num_epochs, learning_rate=args.learning_rate, temperature=args.temperature,
             max_grad_norm=args.max_grad_norm)
    kd.save_model()


if __name__ == "__main__":
    main()
