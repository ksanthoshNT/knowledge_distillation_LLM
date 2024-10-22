from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from datasets import load_dataset
import re
from peft import PeftModel, PeftConfig


def process_sql_text(input_text, sql_query):
    """Process input text to extract schema and question"""

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

    # Extract schema and question
    schema_pattern = r"Here is a database schema:(.*?)Please write me a SQL statement"
    schema_match = re.search(schema_pattern, input_text, re.DOTALL)
    db_schema = schema_match.group(1).strip() if schema_match else ""

    question_pattern = r"Please write me a SQL statement that answers the following question: (.*?)\s*\[/INST\]"
    question_match = re.search(question_pattern, input_text, re.DOTALL)
    question = question_match.group(1).strip() if question_match else ""

    # Transform schema
    transformed_schema = transform_schema(db_schema)

    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Generate a SQL query to answer this question: `{question}`

DDL statements:
{transformed_schema}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The following SQL query best answers the question `{question}`:
```sql
{sql_query}"""


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
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get teacher predictions
    with torch.no_grad():
        teacher_logits = teacher_model(**inputs).logits

    # Get student predictions and calculate loss
    student_logits = student_model(**inputs).logits
    loss = distillation_loss(student_logits, teacher_logits)

    # Optimization step
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()


def main():
    # Model configuration
    teacher_model_name = "defog/llama-3-sqlcoder-8b"
    student_model_name = "jurieyel/77cdm-llama3-sqlcoder-8b-4bit"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    peft_config = PeftConfig.from_pretrained(student_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    student_model = PeftModel.from_pretrained(
        base_model,
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

    # Load dataset
    dataset = load_dataset("lamini/spider_text_to_sql", split="train[:10]")

    # Training loop
    for idx, item in enumerate(dataset):
        # Process the text
        processed_text = process_sql_text(item['input'], item['output'])

        # Training step
        loss = train_step(
            processed_text,
            tokenizer,
            teacher_model,
            student_model,
            optimizer,
            device
        )

        if idx % 10 == 0:
            print(f"Step {idx}, Loss: {loss:.4f}")

        if idx >= 100:  # For testing, remove this limit for full training
            break


if __name__ == "__main__":
    main()