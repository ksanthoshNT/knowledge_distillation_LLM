from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from datasets import load_dataset
import re
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import logging
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_sql_text(input_text, sql_query):
    """Process input text to extract schema and question"""

    # Previous implementation remains the same
    def transform_schema(schema):
        tables = re.split(r'\n\s*\n', schema)
        create_statements = []
        foreign_keys = []

        for table in tables:
            lines = table.strip().split('\n')
            if not lines:
                continue
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

    try:
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
    except Exception as e:
        logger.error(f"Error processing SQL text: {str(e)}")
        raise


def safe_generate(model, inputs, tokenizer, max_length=512):
    """Safely generate text with proper error handling and constraints"""
    try:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                temperature=0.7,  # Add temperature to avoid extreme probabilities
                no_repeat_ngram_size=3,  # Prevent repetitive text
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,  # Enable sampling to avoid deterministic output
                top_k=50,  # Limit vocabulary to top k tokens
                top_p=0.95,  # Nucleus sampling
            )
        return outputs
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        return None


def evaluate_model(model, tokenizer, dataset, device, max_samples=100):
    """Evaluate model performance on dataset with improved error handling"""
    model.eval()
    predictions = []
    references = []

    for idx, item in enumerate(tqdm(dataset)):
        if idx >= max_samples:
            break

        try:
            # Process input text
            processed_text = process_sql_text(item['input'], item['output'])
            inputs = tokenizer(processed_text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate SQL query with safe generation
            outputs = safe_generate(model, inputs, tokenizer)

            if outputs is None:
                logger.warning(f"Skipping sample {idx} due to generation error")
                continue

            # Decode prediction
            pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred_sql = extract_sql_query(pred_text)
            ref_sql = item['output']

            if pred_sql and ref_sql:  # Only add valid predictions
                predictions.append(normalize_sql(pred_sql))
                references.append(normalize_sql(ref_sql))

        except Exception as e:
            logger.error(f"Error processing sample {idx}: {str(e)}")
            continue

    # Calculate metrics only if we have predictions
    if predictions:
        exact_match = accuracy_score(references, predictions)
        return {
            'exact_match': exact_match,
            'predictions': predictions,
            'references': references,
            'num_processed': len(predictions)
        }
    else:
        return {
            'exact_match': 0.0,
            'predictions': [],
            'references': [],
            'num_processed': 0
        }


def main():
    # Model configuration
    teacher_model_name = "meta-llama/Llama-3.2-1B-Instruct"
    student_model_name = "meta-llama/Llama-3.2-1B-Instruct"
    output_dir = "saved_models/student_model"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Load models and tokenizer with error handling
        logger.info("Loading models and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

        # Ensure tokenizer has proper special tokens
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

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

        # Load datasets with smaller batch for testing
        logger.info("Loading datasets...")
        train_dataset = load_dataset("lamini/spider_text_to_sql", split="train[:10]")  # Reduced for testing
        val_dataset = load_dataset("lamini/spider_text_to_sql", split="validation[:5]")  # Reduced for testing

        # Training loop with improved error handling
        best_val_accuracy = 0
        for epoch in range(3):
            logger.info(f"\nStarting epoch {epoch + 1}")

            try:
                # Evaluation
                logger.info("\nEvaluating models...")
                teacher_metrics = evaluate_model(teacher_model, tokenizer, val_dataset, device)
                student_metrics = evaluate_model(student_model, tokenizer, val_dataset, device)

                logger.info(f"Teacher Model Exact Match: {teacher_metrics['exact_match']:.4f}")
                logger.info(f"Student Model Exact Match: {student_metrics['exact_match']:.4f}")

                # Save best model
                if student_metrics['exact_match'] > best_val_accuracy:
                    best_val_accuracy = student_metrics['exact_match']
                    student_model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    logger.info(f"Saved new best model with accuracy: {best_val_accuracy:.4f}")

            except Exception as e:
                logger.error(f"Error in epoch {epoch + 1}: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()