import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def save_model_huggingface(model, tokenizer, save_directory="sql_model_saved"):
    """
    Save the model and tokenizer using HuggingFace's save_pretrained method

    Args:
        model: The trained model
        tokenizer: The tokenizer
        save_directory (str): Directory where to save the model and tokenizer
    """
    # Save the model
    model.save_pretrained(save_directory)

    # Save the tokenizer
    tokenizer.save_pretrained(save_directory)

    print(f"Model and tokenizer saved to {save_directory}")


def load_model_huggingface(model_directory="sql_model_saved"):
    """
    Load the model and tokenizer using HuggingFace's from_pretrained method

    Args:
        model_directory (str): Directory containing the saved model and tokenizer

    Returns:
        tuple: (model, tokenizer) loaded and ready for inference
    """
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_directory,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_directory)

    # Set model to evaluation mode
    model.eval()

    return model, tokenizer


# Example of how to convert your existing saved model to HuggingFace format
def convert_existing_to_huggingface():
    # First, load the original model architecture
    model = AutoModelForCausalLM.from_pretrained(
        "aspenita/llama-3-sqlcoder-8b-AWQ",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("aspenita/llama-3-sqlcoder-8b-AWQ")

    # Load your saved weights
    checkpoint= torch.load("/home/data_science/project_files/santhosh/knowledge_distillation_LLM/knowledge_distillation/src/main/distillation/best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])

    # Save in HuggingFace format
    save_model_huggingface(model, tokenizer,save_directory='llama3-8b-awq-distilled-f16')


if __name__ == "__main__":
    # Example 1: Convert existing saved model to HuggingFace format
    convert_existing_to_huggingface()

    # Example 2: Load the converted model
    model, tokenizer = load_model_huggingface()

    # Example usage of loaded model
    question = "What are the total sales by category?"
    schema = """
    Products:
    id [INT] primary_key
    category [TEXT]
    price [DECIMAL]

    Sales:
    id [INT] primary_key
    product_id [INT] = Products.id
    quantity [INT]
    """

    # Format input text
    input_text = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Generate a SQL query to answer this question: `{question}`

DDL statements:
{schema}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The following SQL query best answers the question `{question}`:
```sql
"""

    # Generate SQL
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.0001,
            do_sample=True
        )

    generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated SQL:\n{generated_sql}")