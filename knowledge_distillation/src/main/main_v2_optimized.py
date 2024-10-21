import argparse
import torch
from torch.nn.utils import prune
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, DataCollatorWithPadding
from datasets import load_dataset, IterableDataset
from tqdm import tqdm
from torch.nn import functional as F
from evaluate import load
from tabulate import tabulate

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

    def load_teacher_model(self, use_8bit=False, precision="float16"):
        print("Loading teacher model...")
        dtype = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16
        }
        if dtype.get(precision,None) is None:
            raise ValueError(f"Invalid precision: {precision}")

        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            self.teacher_model_name,
            device_map="auto",
            load_in_8bit=use_8bit,
            torch_dtype=dtype[precision]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.teacher_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.teacher_model.config.pad_token_id = self.tokenizer.eos_token_id
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        print(f"Teacher model size: {teacher_params:,} parameters")
        print("Teacher model loaded successfully.")

    def load_student_model(self, student_model_name=None, target_size=None, precision="float16", load_weights=True):
        print(f"Loading student model... {student_model_name}")
        dtype = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16
        }
        if dtype.get(precision, None) is None:
            raise ValueError(f"Invalid precision: {precision}")

        if len(student_model_name)==2 and 'B' in student_model_name:
            self.student_model = self._prune_model(self.teacher_model, target_size, dtype)
        elif student_model_name:
            if load_weights:
                self.student_model = AutoModelForCausalLM.from_pretrained(
                    student_model_name,
                    device_map="auto",
                    torch_dtype=dtype[precision]
                )
            else:
                config = AutoConfig.from_pretrained(student_model_name)
                self.student_model = AutoModelForCausalLM.from_config(config).to(dtype)

        else:
            raise ValueError("Either student_model_name or target_size must be provided")
        self.student_model.config.pad_token_id = self.teacher_model.config.pad_token_id
        for param in self.student_model.parameters():
            param.requires_grad = True
        student_params = sum(p.numel() for p in self.student_model.parameters())
        print(f"Student model size: {student_params:,} parameters")
        print(f"Student model loaded successfully with {precision} precision.")

    def _prune_model(self, model, target_size, dtype):
        print(f"Pruning model to approximately {target_size} parameters")
        current_size = sum(p.numel() for p in model.parameters())
        target_size = int(target_size.rstrip('B')) * 1_000_000_000  # Convert 3B to 3000000000
        prune_ratio = 1 - (target_size / current_size)

        pruned_model = AutoModelForCausalLM.from_config(model.config).to(dtype)
        pruned_model.load_state_dict(model.state_dict())

        for name, module in pruned_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=prune_ratio)
                prune.remove(module, 'weight')

        pruned_size = sum(p.numel() for p in pruned_model.parameters())
        print(f"Pruned model size: {pruned_size:,} parameters")
        return pruned_model.to(dtype)

    def load_dataset(self, streaming=True):
        print("Loading dataset...")
        self.dataset = load_dataset(self.dataset_name, self.dataset_config, split="train", streaming=streaming)
        print(self.dataset)
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

    def train(self, num_epochs=3, learning_rate=1e-5, temperature=1.0):
        print("Starting training...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.student_model.to(device)
        optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=learning_rate)
        self.student_model.gradient_checkpointing_enable()

        for epoch in range(num_epochs):
            self.student_model.train()
            total_loss = 0
            num_batches = 0

            for batch in tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                batch = {k: v.to(device) for k, v in batch.items()}

                with torch.no_grad():
                    teacher_outputs = self.teacher_model(**batch)

                student_outputs = self.student_model(**batch)

                teacher_logits = teacher_outputs.logits / temperature
                student_logits = student_outputs.logits / temperature

                loss = F.kl_div(
                    F.log_softmax(student_logits, dim=-1),
                    F.softmax(teacher_logits, dim=-1),
                    reduction="batchmean",
                    log_target=False
                ) * (temperature ** 2)

                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"NaN or Inf loss detected. Skipping batch.")
                    continue

                total_loss += loss.item()
                num_batches += 1

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

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
    parser.add_argument("--teacher_model_name", default="meta-llama/Llama-3.2-3B-Instruct", help="Name of the teacher model")
    parser.add_argument("--teacher_precision", default="bfloat16", help="Precision to use for the teacher model")
    parser.add_argument("--dataset_name", default="wikitext", help="Name of the dataset")
    parser.add_argument("--dataset_config_name", default="wikitext-2-raw-v1", help="Configuration name of the dataset")
    parser.add_argument("--student_model_name", default="meta-llama/Llama-3.2-1B-Instruct", help="Name of the student model")
    parser.add_argument("--student_precision", default="bfloat16", help="Precision to use for the student model")
    parser.add_argument("--load_student_weights", action=argparse.BooleanOptionalAction, default=True, help="Whether to load the weights for the student model")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True, help="Whether to use streaming for dataset loading")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for data preparation")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length for data preparation")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to use in data preparation")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for knowledge distillation")
    return parser.parse_args()

def print_arguments(args):
    args_dict = vars(args)
    table_data = [[k, v] for k, v in args_dict.items()]
    print("\nKnowledge Distillation Arguments:")
    print(tabulate(table_data, headers=["Argument", "Value"], tablefmt="grid"))
    print()

def main():
    args = parse_arguments()
    print_arguments(args)

    kd = KnowledgeDistillation(args.teacher_model_name, args.dataset_name, args.dataset_config_name)
    kd.load_dataset(streaming=args.streaming)
    kd.load_teacher_model(precision=args.teacher_precision)
    kd.load_student_model(
        student_model_name=args.student_model_name,
        precision=args.student_precision,
        load_weights=args.load_student_weights
    )
    kd.prepare_data(batch_size=args.batch_size, max_length=args.max_length, num_samples=args.num_samples)
    kd.train(num_epochs=args.num_epochs, learning_rate=args.learning_rate, temperature=args.temperature)
    kd.save_model()

if __name__ == "__main__":
    main()