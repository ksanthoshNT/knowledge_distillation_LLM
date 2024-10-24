import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def show_sql_generations():
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "aspenita/llama-3-sqlcoder-8b-AWQ",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("aspenita/llama-3-sqlcoder-8b-AWQ")

    # Load checkpoint
    checkpoint = torch.load("checkpoints/best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load dataset
    dataset = load_dataset("lamini/spider_text_to_sql")
    eval_dataset = dataset['validation']

    # Generate for first 5 examples
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for idx in range(5):
        sample = eval_dataset[idx]
        print(f"\nExample {idx + 1}")
        print("-" * 50)
        print(f"Input: {sample['input']}")

        with torch.no_grad():
            inputs = tokenizer(
                sample['input'],
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(device)

            outputs = model.generate(
                **inputs,
                max_length=256,
                temperature=0.0,
                do_sample=False
            )

            generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\nGenerated SQL: {generated_sql}")
            print(f"Ground Truth: {sample['output']}")


if __name__ == "__main__":
    show_sql_generations()