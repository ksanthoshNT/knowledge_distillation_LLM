from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from datasets import load_dataset
import re
from peft import PeftModel, PeftConfig
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score


def process_sql_text(input_text, sql_query):
    """Process input text to extract schema and question"""

    # Previous implementation remains the same
    def transform_schema(schema):
        tables = re.split(r'\n\s*\n', schema)
        create_statements = []
        foreign_keys = []

        for table in tables:
            lines = table.strip().split('\n')
            table_name = lines[0].strip(':')
            columns = lines[1:]

            create_statement = f"CREATE TABLE {table_name} (\n"
            for column in columns:
                parts = column.split('[')
                col_name = parts[0].strip()
                col_type = parts[1].split(']')[0].strip()

                if col_type == 'INT':
                    col_type = 'INTEGER'
                elif col_type == 'TEXT':
                    col_type = 'VARCHAR(100)'

                create_statement += f"  {col_name} {col_type}"
                if 'primary_key' in column:
                    create_statement += " PRIMARY KEY"
                create_statement += ",\n"

                if '=' in column:
                    fk_parts = column.split('=')
                    fk_table, fk_column = fk_parts[1].strip().split('.')
                    foreign_keys.append(
                        f"-- {table_name}.{col_name} can be joined with {fk_table}.{fk_column}")

            create_statement = create_statement.rstrip(',\n') + "\n);\n"
            create_statements.append(create_statement)

        return "\n".join(create_statements) + "\n" + "\n".join(foreign_keys)

    schema_pattern = r"Here is a database schema:(.*?)Please write me a SQL statement"
    schema_match = re.search(schema_pattern, input_text, re.DOTALL)
    db_schema = schema_match.group(1).strip() if schema_match else ""

    question_pattern = r"Please write me a SQL statement that answers the following question: (.*?)\s*\[/INST\]"
    question_match = re.search(question_pattern, input_text, re.DOTALL)
    question = question_match.group(1).strip() if question_match else ""

    transformed_schema = transform_schema(db_schema)

    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Generate a SQL query to answer this question: `{question}`

DDL statements:
{transformed_schema}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The following SQL query best answers the question `{question}`:
```sql
{sql_query}"""


def evaluate_model(model, tokenizer, dataset, device, max_samples=100):
    """Evaluate model performance on dataset"""
    model.eval()
    predictions = []
    references = []

    with torch.no_grad():
        for idx, item in enumerate(tqdm(dataset)):
            if idx >= max_samples:
                break

            # Process input text
            processed_text = process_sql_text(item['input'], item['output'])
            inputs = tokenizer(processed_text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate SQL query
            outputs = model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            # Decode prediction
            pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract SQL query from prediction
            pred_sql = extract_sql_query(pred_text)
            ref_sql = item['output']

            predictions.append(normalize_sql(pred_sql))
            references.append(normalize_sql(ref_sql))

    # Calculate metrics
    exact_match = accuracy_score(references, predictions)
    return {
        'exact_match': exact_match,
        'predictions': predictions,
        'references': references
    }


def normalize_sql(query):
    """Normalize SQL query for comparison"""
    # Remove extra whitespace and convert to lowercase
    query = ' '.join(query.lower().split())
    # Remove semicolon at the end if present
    query = query.rstrip(';')
    return query


def extract_sql_query(text):
    """Extract SQL query from generated text"""
    sql_pattern = r"```sql\n(.*?)```"
    match = re.search(sql_pattern, text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """Calculate distillation loss between student and teacher models"""
    teacher_logits = teacher_logits / temperature
    student_logits = student_logits / temperature

    return F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        reduction='batchmean'
    )


def train_step(text, tokenizer, teacher_model, student_model, optimizer, device):
    """Perform one training step of knowledge distillation"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        teacher_logits = teacher_model(**inputs).logits

    student_logits = student_model(**inputs).logits
    loss = distillation_loss(student_logits, teacher_logits)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()


def main():
    # Model configuration
    teacher_model_name = "defog/llama-3-sqlcoder-8b"
    student_model_name = "aspenita/llama-3-sqlcoder-8b-AWQ"
    output_dir = "saved_models/student_model"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    student_model = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    for param in student_model.parameters():
        param.requires_grad = True

    # Prepare models
    teacher_model.eval()
    student_model.train()

    # Setup optimizer
    optimizer = AdamW(student_model.parameters(), lr=5e-5)

    # Load datasets
    train_dataset = load_dataset("lamini/spider_text_to_sql", split="train[:100]")
    val_dataset = load_dataset("lamini/spider_text_to_sql", split="validation[:50]")

    # Training loop
    best_val_accuracy = 0
    for epoch in range(3):  # Number of epochs
        total_loss = 0
        print(f"\nEpoch {epoch + 1}")

        # Training
        for idx, item in enumerate(tqdm(train_dataset, desc="Training")):
            processed_text = process_sql_text(item['input'], item['output'])
            loss = train_step(
                processed_text,
                tokenizer,
                teacher_model,
                student_model,
                optimizer,
                device
            )
            total_loss += loss

            if idx % 10 == 0:
                print(f"Step {idx}, Loss: {loss:.4f}")

        avg_loss = total_loss / len(train_dataset)
        print(f"Average training loss: {avg_loss:.4f}")

        # Evaluation
        print("\nEvaluating models...")
        teacher_metrics = evaluate_model(teacher_model, tokenizer, val_dataset, device)
        student_metrics = evaluate_model(student_model, tokenizer, val_dataset, device)

        print(f"Teacher Model Exact Match: {teacher_metrics['exact_match']:.4f}")
        print(f"Student Model Exact Match: {student_metrics['exact_match']:.4f}")

        # Save best model
        if student_metrics['exact_match'] > best_val_accuracy:
            best_val_accuracy = student_metrics['exact_match']

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Save model and tokenizer
            student_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            print(f"Saved new best model with accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    main()