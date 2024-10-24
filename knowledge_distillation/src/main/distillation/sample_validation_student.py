import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def transform_text(input_string):

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

    # Extract the database schema
    schema_pattern = r"Here is a database schema:(.*?)Please write me a SQL statement"
    schema_match = re.search(schema_pattern, input_string, re.DOTALL)
    db_schema = schema_match.group(1).strip() if schema_match else "Schema not found"

    # Extract the question
    question_pattern = r"Please write me a SQL statement that answers the following question: (.*?)\s*\[/INST\]"
    question_match = re.search(question_pattern, input_string, re.DOTALL)
    question = question_match.group(1).strip() if question_match else "Question not found"

    # Transform the schema
    transformed_schema = transform_schema(db_schema)

    text = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Generate a SQL query to answer this question: `{question}`

DDL statements:
{transformed_schema}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The following SQL query best answers the question `{question}`:
```sql
"""
    return text



def show_sql_generations():
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "aspenita/llama-3-sqlcoder-8b-AWQ",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("aspenita/llama-3-sqlcoder-8b-AWQ")

    # Load checkpoint
    checkpoint = torch.load("best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    torch.save(model.state_dict(), "saved_model.pt")
    print("Model saved")

    # Load dataset
    dataset = load_dataset("lamini/spider_text_to_sql")
    eval_dataset = dataset['validation']

    # Generate for first 5 examples
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for idx in range(5):
        sample = eval_dataset[idx]
        sample['input'] = transform_text(sample['input'])
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
                max_new_tokens=256,
                temperature=0.0,
                do_sample=False
            )

            generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\nGenerated SQL: {generated_sql}")
            print(f"Ground Truth: {sample['output']}")


if __name__ == "__main__":
    show_sql_generations()