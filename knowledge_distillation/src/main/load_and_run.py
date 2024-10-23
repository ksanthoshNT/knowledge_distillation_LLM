import logging

import torch
import re
from typing import Dict, Any

from knowledge_distillation.src.main.fixingclass_transformers_distillation_main import KnowledgeDistillationModelConfig, \
    KnowledgeDistillationModel


def transform_inference_text(schema: str, question: str) -> str:
    """
    Transform schema and question into the format expected by the model.

    Args:
        schema (str): Database schema text
        question (str): SQL question text

    Returns:
        str: Formatted input text
    """

    def transform_schema(schema: str) -> str:
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

    # Transform the schema
    transformed_schema = transform_schema(schema)

    # Format the input text
    text = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

        Generate a SQL query to answer this question: `{question}`

        DDL statements:
        {transformed_schema}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        The following SQL query best answers the question `{question}`:
        ```sql"""

    return text


def load_model_for_inference(checkpoint_path: str,
                             config: KnowledgeDistillationModelConfig) -> KnowledgeDistillationModel:
    """
    Load a saved checkpoint model for inference.

    Args:
        checkpoint_path (str): Path to the checkpoint file
        config (KnowledgeDistillationModelConfig): Configuration for the model

    Returns:
        KnowledgeDistillationModel: Loaded model ready for inference
    """
    # Create a new model instance
    model = KnowledgeDistillationModel(config)

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Load the model state
    model.student.load_state_dict(checkpoint['model_state_dict'])

    # Set model to evaluation mode
    model.student.eval()

    logging.info(f"Successfully loaded checkpoint from epoch {checkpoint['epoch']}")
    return model


def generate_sql_query(
        model: KnowledgeDistillationModel,
        schema: str,
        question: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95
) -> str:
    """
    Generate SQL query using the loaded model.

    Args:
        model: Loaded KnowledgeDistillationModel
        schema (str): Database schema text
        question (str): Question to convert to SQL
        max_length (int): Maximum length of generated sequence
        temperature (float): Sampling temperature
        top_p (float): Nucleus sampling parameter

    Returns:
        str: Generated SQL query
    """
    # Transform input text
    input_text = transform_inference_text(schema, question)

    # Tokenize input
    inputs = model.tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )

    # Move inputs to the same device as model
    inputs = {k: v.to(model.student.device) for k, v in inputs.items()}

    # Generate with the student model
    with torch.no_grad():
        outputs = model.student.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=model.tokenizer.pad_token_id,
            eos_token_id=model.tokenizer.eos_token_id
        )

    # Decode the generated output
    generated_sql = model.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract SQL query from the generated text
    sql_match = re.search(r'```sql\s*(.*?)\s*(?:```|$)', generated_sql, re.DOTALL)
    if sql_match:
        return sql_match.group(1).strip()
    return generated_sql.strip()


# Initialize configuration
if __name__ == '__main__':

    config = KnowledgeDistillationModelConfig(
        teacher_model_name="defog/llama-3-sqlcoder-8b",
        teacher_model_torch_dtype="float32",
        student_model_name="aspenita/llama-3-sqlcoder-8b-AWQ",
        student_model_torch_dtype="float32",
        distillation_type="black_box",
        temperature=2.0,
        alpha=0.5,
        batch_size=4
    )

    # Load the model
    checkpoint_path = "/home/data_science/project_files/santhosh/knowledge_distillation_LLM/knowledge_distillation/src/main/checkpoints/checkpoint_epoch_3.pt"
    model = load_model_for_inference(checkpoint_path, config)

    # Example schema and question
    schema = '''
    Students:
    student_id [INT] {primary_key}
    name [TEXT]
    age [INT]
    grade [INT]
    
    Courses:
    course_id [INT] {primary_key}
    course_name [TEXT]
    teacher_id [INT] = Teachers.teacher_id
    
    Teachers:
    teacher_id [INT] {primary_key}
    teacher_name [TEXT]
    department [TEXT]
    '''

    question = "Find all students who are taking courses taught by teachers in the Science department"

    # Generate SQL query
    sql_query = generate_sql_query(model, schema, question)
    print(f"Generated SQL Query:\n{sql_query}")
