import re

import torch
import logging
import json
from torch.utils.data import DataLoader
from typing import Dict, Any
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    def __init__(self, batch_size: int = 4):
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Load dataset
        self.dataset = load_dataset("lamini/spider_text_to_sql")
        self.eval_dataset = self.dataset['validation']
        logger.info(f"Loaded validation dataset with {len(self.eval_dataset)} samples")

        # Initialize models
        self.teacher_model, self.teacher_tokenizer = self._load_teacher_model()
        self.original_student_model, self.student_tokenizer = self._load_student_model()
        self.best_model = self._load_best_model()

    def _load_teacher_model(self):
        logger.info("Loading teacher model...")
        model = AutoModelForCausalLM.from_pretrained(
            "defog/llama-3-sqlcoder-8b",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained("defog/llama-3-sqlcoder-8b")
        return model, tokenizer

    def _load_student_model(self):
        logger.info("Loading original student model...")
        model = AutoModelForCausalLM.from_pretrained(
            "aspenita/llama-3-sqlcoder-8b-AWQ",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained("aspenita/llama-3-sqlcoder-8b-AWQ")
        return model, tokenizer

    def _load_best_model(self):
        logger.info("Loading best trained model...")
        model = AutoModelForCausalLM.from_pretrained(
            "aspenita/llama-3-sqlcoder-8b-AWQ",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        checkpoint = torch.load("best_model.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        torch.save(model.state_dict(), "saved_model.pt")
        logger.info("Student Model saved !! ")
        return model

    def transform_text(self,input_string):

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

    def evaluate_model(self, model, tokenizer, model_name: str):
        model.eval()
        metrics = {
            'exact_match': 0,
            'total_samples': 0,
            'generated_examples': []
        }

        logger.info(f"Evaluating {model_name}...")

        for idx in tqdm(range(min(100, len(self.eval_dataset)))):  # Evaluate on first 100 samples
            sample = self.eval_dataset[idx]
            sample['input'] = self.transform_text(sample['input'])

            with torch.no_grad():
                inputs = tokenizer(
                    sample['input'],
                    return_tensors="pt",
                    max_length=512,
                    truncation=True
                ).to(self.device)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.0,
                    do_sample=False
                )

                predicted_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Clean up predictions and ground truth
                predicted_sql = predicted_sql.strip().lower()
                ground_truth = sample['output'].strip().lower()

                # Update metrics
                exact_match = predicted_sql == ground_truth
                metrics['exact_match'] += int(exact_match)
                metrics['total_samples'] += 1

                # Store some examples
                if len(metrics['generated_examples']) < 5:
                    metrics['generated_examples'].append({
                        'input': sample['input'],
                        'prediction': predicted_sql,
                        'ground_truth': ground_truth,
                        'is_correct': exact_match
                    })

        metrics['accuracy'] = metrics['exact_match'] / metrics['total_samples']
        return metrics

    def compare_model_outputs(self):
        logger.info("Comparing model outputs...")
        comparison_metrics = {
            'teacher_student_similarity': [],
            'teacher_best_similarity': [],
            'student_best_similarity': []
        }

        for idx in tqdm(range(min(50, len(self.eval_dataset)))):  # Compare on first 50 samples
            sample = self.eval_dataset[idx]

            with torch.no_grad():
                # Get teacher outputs
                teacher_inputs = self.teacher_tokenizer(
                    sample['input'],
                    return_tensors="pt",
                    max_length=512,
                    truncation=True
                ).to(self.device)
                teacher_outputs = self.teacher_model(**teacher_inputs).logits

                # Get student outputs
                student_inputs = self.student_tokenizer(
                    sample['input'],
                    return_tensors="pt",
                    max_length=512,
                    truncation=True
                ).to(self.device)
                student_outputs = self.original_student_model(**student_inputs).logits
                best_outputs = self.best_model(**student_inputs).logits

                # Calculate similarities
                comparison_metrics['teacher_student_similarity'].append(
                    torch.nn.functional.cosine_similarity(
                        teacher_outputs.view(-1),
                        student_outputs.view(-1),
                        dim=0
                    ).mean().item()
                )

                comparison_metrics['teacher_best_similarity'].append(
                    torch.nn.functional.cosine_similarity(
                        teacher_outputs.view(-1),
                        best_outputs.view(-1),
                        dim=0
                    ).mean().item()
                )

                comparison_metrics['student_best_similarity'].append(
                    torch.nn.functional.cosine_similarity(
                        student_outputs.view(-1),
                        best_outputs.view(-1),
                        dim=0
                    ).mean().item()
                )

        # Calculate average similarities
        for key in comparison_metrics:
            comparison_metrics[key] = np.mean(comparison_metrics[key])

        return comparison_metrics

    def run_comprehensive_evaluation(self):
        results = {
            'teacher_metrics': self.evaluate_model(
                self.teacher_model,
                self.teacher_tokenizer,
                "Teacher Model"
            ),
            'student_metrics': self.evaluate_model(
                self.original_student_model,
                self.student_tokenizer,
                "Original Student Model"
            ),
            'best_model_metrics': self.evaluate_model(
                self.best_model,
                self.student_tokenizer,
                "Best Trained Model"
            ),
            'model_comparisons': self.compare_model_outputs()
        }

        # Save results
        output_path = Path("evaluation_results/comprehensive_evaluation.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open('w') as f:
            json.dump(results, f, indent=2)

        logger.info("Comprehensive evaluation completed. Results summary:")
        logger.info(f"Teacher Model Accuracy: {results['teacher_metrics']['accuracy']:.4f}")
        logger.info(f"Original Student Accuracy: {results['student_metrics']['accuracy']:.4f}")
        logger.info(f"Best Model Accuracy: {results['best_model_metrics']['accuracy']:.4f}")
        logger.info("\nModel Similarities:")
        logger.info(f"Teacher-Student Similarity: {results['model_comparisons']['teacher_student_similarity']:.4f}")
        logger.info(f"Teacher-Best Similarity: {results['model_comparisons']['teacher_best_similarity']:.4f}")
        logger.info(f"Student-Best Similarity: {results['model_comparisons']['student_best_similarity']:.4f}")


if __name__ == "__main__":
    evaluator = ComprehensiveEvaluator()
    evaluator.run_comprehensive_evaluation()