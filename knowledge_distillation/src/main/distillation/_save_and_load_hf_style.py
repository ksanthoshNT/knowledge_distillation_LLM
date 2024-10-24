import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_huggingface(model_directory="sql_model_saved"):
    model = AutoModelForCausalLM.from_pretrained(
        model_directory,
        torch_dtype=torch.float32,
        device_map="auto"
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_directory)

    # Set model to evaluation mode
    model.eval()

    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = load_model_huggingface("llama3-8b-awq-distilled-f32")

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