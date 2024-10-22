import os
from pathlib import Path
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Union
import logging
from logging.handlers import RotatingFileHandler

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm.auto import tqdm

from knowledge_distillation.src.main.evaluation.eval_args import EvalArguments, TransformerConfig
from knowledge_distillation.src.main.evaluation.metrics import MetricsCalculator


class TextTransformDataset(Dataset):
    def __init__(self, data, max_length=512):
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def transform_text(self, input_text: str) -> str:
        """Transform the input text into the desired format."""
        # Extract schema and question
        schema_pattern = r"Here is a database schema:(.*?)Please write me a SQL statement"
        schema_match = re.search(schema_pattern, input_text, re.DOTALL)
        db_schema = schema_match.group(1).strip() if schema_match else "Schema not found"

        question_pattern = r"Please write me a SQL statement that answers the following question: (.*?)\s*\[/INST\]"
        question_match = re.search(question_pattern, input_text, re.DOTALL)
        question = question_match.group(1).strip() if question_match else "Question not found"

        # Transform schema
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

        transformed_schema = transform_schema(db_schema)

        return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Generate a SQL query to answer this question: `{question}`

DDL statements:
{transformed_schema}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The following SQL query best answers the question `{question}`:"""

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_text': item['input'],
            'transformed_text': self.transform_text(item['input'])
        }


class TransformerEvaluator:
    def __init__(self, config: Union[str, EvalArguments, dict, None] = None):
        self.config = TransformerConfig(config)
        self.metrics = MetricsCalculator()
        self._setup_logging()
        self._setup_output_dir()

        # Set device and batch size
        self.device = torch.device(
            self.config.model_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.batch_size = self.config.evaluation_config.get('batch_size', 8)


    def _setup_logging(self):
        self.logger = logging.getLogger("transformer_evaluator")
        self.logger.setLevel(logging.INFO)

        if self.logger.handlers:
            self.logger.handlers.clear()

        log_dir = Path(self.config.dataset_config['output_dir']) / 'logs'
        log_dir.mkdir(exist_ok=True, parents=True)

        file_handler = RotatingFileHandler(
            log_dir / 'evaluator.log',
            maxBytes=10 * 1024 * 1024,
            backupCount=5
        )
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(file_handler)

    def _setup_output_dir(self):
        output_dir = self.config.dataset_config['output_dir']
        os.makedirs(output_dir, exist_ok=True)

    def _load_models(self):
        self.logger.info("Loading models...")
        try:
            # Load models
            teacher_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_config['teacher_model_name']
            ).to(self.device).eval()

            student_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_config['student_model_path']
            ).to(self.device).eval()

            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_config['teacher_model_name']
            )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                teacher_model.config.pad_token_id = tokenizer.pad_token_id
                student_model.config.pad_token_id = tokenizer.pad_token_id

            return teacher_model, student_model, tokenizer

        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}", exc_info=True)
            raise

    @torch.no_grad()
    def evaluate(self, split_name=None):
        self.logger.info("Starting evaluation process")

        try:
            # Load models and tokenizer
            teacher_model, student_model, tokenizer = self._load_models()

            # Load dataset
            dataset = load_dataset(
                self.config.dataset_config['name'],
                self.config.dataset_config['config']
            )

            # Process splits
            splits = [split_name] if split_name else ['train', 'test', 'validation']

            for split in splits:
                if split not in dataset:
                    self.logger.info(f"Split {split} not found - skipping")
                    continue

                self.logger.info(f"Evaluating {split} split")

                # Create dataset and dataloader
                split_data = dataset[split].select(range(self.config.dataset_config['num_samples']))
                eval_dataset = TextTransformDataset(split_data)
                dataloader = DataLoader(
                    eval_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=4
                )

                results = []

                for batch in tqdm(dataloader, desc=f"Evaluating {split}"):
                    # Tokenize transformed text
                    encodings = tokenizer(
                        batch['transformed_text'],
                        padding=True,
                        truncation=True,
                        return_tensors="pt"
                    ).to(self.device)

                    # Generate from both models
                    teacher_outputs = teacher_model.generate(
                        **encodings,
                        max_new_tokens=self.config.generation_config['max_new_tokens'],
                        num_return_sequences=1,
                        do_sample=self.config.generation_config['do_sample'],
                        temperature=self.config.generation_config['temperature'],
                        top_p=self.config.generation_config['top_p']
                    )

                    student_outputs = student_model.generate(
                        **encodings,
                        max_new_tokens=self.config.generation_config['max_new_tokens'],
                        num_return_sequences=1,
                        do_sample=self.config.generation_config['do_sample'],
                        temperature=self.config.generation_config['temperature'],
                        top_p=self.config.generation_config['top_p']
                    )

                    # Process each example in batch
                    for i in range(len(batch['input_text'])):
                        teacher_text = tokenizer.decode(teacher_outputs[i], skip_special_tokens=True)
                        student_text = tokenizer.decode(student_outputs[i], skip_special_tokens=True)

                        # Calculate metrics
                        metrics = self.metrics.compute_metrics(
                            teacher_model, student_model, tokenizer,
                            batch['transformed_text'][i], teacher_text, student_text
                        )

                        results.append({
                            "input": batch['input_text'][i],
                            "transformed_input": batch['transformed_text'][i],
                            "teacher_output": teacher_text,
                            "student_output": student_text,
                            "metrics": metrics.__dict__
                        })

                # Save results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = Path(self.config.dataset_config['output_dir']) / f"{split}_results_{timestamp}.json"
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)

                # Log summary
                avg_metrics = {
                    "teacher_perplexity": np.mean([r["metrics"]["teacher_perplexity"] for r in results]),
                    "student_perplexity": np.mean([r["metrics"]["student_perplexity"] for r in results]),
                    "bleu_score": np.mean([r["metrics"]["bleu_score"] for r in results])
                }

                self.logger.info(f"\n{split.capitalize()} Split Summary:")
                for metric, value in avg_metrics.items():
                    self.logger.info(f"Average {metric}: {value:.4f}")

        except Exception as e:
            self.logger.error("Evaluation failed", exc_info=True)
            raise

        self.logger.info("Evaluation completed successfully")