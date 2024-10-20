import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
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

    def load_teacher_model(self):
        self.teacher_model = AutoModelForCausalLM.from_pretrained(self.teacher_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.teacher_model_name)
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def load_student_model(self, student_model_name=None, target_size=None):
        if student_model_name:
            self.student_model = AutoModelForCausalLM.from_pretrained(student_model_name)
        elif target_size:
            # Create a pruned version of the teacher model
            self.student_model = self._prune_model(self.teacher_model, target_size)
        else:
            raise ValueError("Either student_model_name or target_size must be provided")

    def _prune_model(self, model, target_size):
        # This is a placeholder for model pruning logic
        # In a real implementation, you would use techniques like weight pruning,
        # layer removal, or model compression to reduce the model size
        print(f"Pruning model to {target_size} parameters")
        return model  # Return the pruned model

    def load_dataset(self):
        self.dataset = load_dataset(self.dataset_name, self.dataset_config, split="train")

    def prepare_data(self, batch_size=8):
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], truncation=True, max_length=512)

        tokenized_dataset = self.dataset.map(tokenize_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=data_collator)

    def train(self, num_epochs=3, learning_rate=5e-5, temperature=0.5):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher_model.to(device)
        self.student_model.to(device)

        optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=learning_rate)
        perplexity = load("perplexity")

        for epoch in range(num_epochs):
            self.student_model.train()
            total_loss = 0

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

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

            # Evaluation
            eval_loss = self._evaluate()
            perplexity_score = perplexity.compute(predictions=self.student_model, model_id=self.teacher_model_name)

            print(f"Evaluation Loss: {eval_loss:.4f}")
            print(f"Perplexity: {perplexity_score['perplexity']:.4f}")

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
        self.student_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Distilled model saved to {output_dir}")


# Example usage
if __name__ == "__main__":
    kd = KnowledgeDistillation("meta-llama/Llama-3.2-3B-Instruct", "wikitext", "wikitext-2-raw-v1")
    kd.load_teacher_model()
    kd.load_student_model(student_model_name="meta-llama/Llama-3.2-1B-Instruct")  # 1B parameters
    kd.load_dataset()
    kd.prepare_data()
    kd.train()
    kd.save_model()