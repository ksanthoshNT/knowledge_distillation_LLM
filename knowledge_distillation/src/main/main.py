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

    def load_teacher_model(self, use_8bit=True):
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

    def train(self, num_epochs=3, learning_rate=5e-5, temperature=0.5):
        print("Starting training...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if next(self.student_model.parameters()).device != device:
            self.student_model.to(device)

        optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=learning_rate)
        perplexity = load("perplexity")

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

                loss = F.kl_div(
                    F.log_softmax(student_outputs.logits / temperature, dim=-1),
                    F.softmax(teacher_outputs.logits / temperature, dim=-1),
                    reduction="batchmean"
                ) * (temperature ** 2)

                total_loss += loss.item()
                num_batches += 1

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

            # Evaluation
            eval_loss = self._evaluate()
            perplexity_score = perplexity.compute(predictions=self.student_model, model_id=self.teacher_model_name)

            print(f"Evaluation Loss: {eval_loss:.4f}")
            print(f"Perplexity: {perplexity_score['perplexity']:.4f}")

        print("Training completed.")

    def _evaluate(self):
        self.student_model.eval()
        eval_loss = 0
        eval_steps = 0
        device = next(self.student_model.parameters()).device

        for batch in tqdm(self.dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = self.student_model(**batch)

            eval_loss += outputs.loss.item()
            eval_steps += 1

        return eval_loss / eval_steps

    def save_model(self, output_dir="distilled_model"):
        print(f"Saving distilled model to {output_dir}...")
        self.student_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Distilled model saved to {output_dir}")


# Example usage
if __name__ == "__main__":
    kd = KnowledgeDistillation("meta-llama/Llama-3.2-3B-Instruct", "wikitext", "wikitext-2-raw-v1")
    kd.load_teacher_model()  # Load teacher in 8-bit quantization
    kd.load_student_model(
        student_model_name="meta-llama/Llama-3.2-1B-Instruct",  # Example student model
        precision="float16",  # Use float16 precision
        load_weights=False  # Load only the architecture, not the weights
    )
    kd.load_dataset(streaming=True)
    kd.prepare_data(batch_size=2, max_length=128, num_samples=20)
    kd.train(num_epochs=3, learning_rate=5e-5, temperature=0.5)
    kd.save_model()
