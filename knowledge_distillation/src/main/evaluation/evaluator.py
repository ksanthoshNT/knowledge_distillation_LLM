import os
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Any, Union
import logging
from logging.handlers import RotatingFileHandler

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm.auto import tqdm

from knowledge_distillation.src.main.evaluation.eval_args import EvalArguments, TransformerConfig
from knowledge_distillation.src.main.evaluation.metrics import MetricsCalculator


class TransformerEvaluator:
    def __init__(self, config: Union[str, EvalArguments, dict, None] = None):
        """
        Initialize evaluator with flexible configuration.

        Args:
            config: Can be one of:
                - str: Path to config.ini file
                - EvalArguments: HuggingFace-style arguments
                - dict: Dictionary of configuration values
                - None: Use default values
        """
        self.config = TransformerConfig(config)
        self.metrics = MetricsCalculator()
        self._setup_logging()
        self._setup_output_dir()

    def _setup_logging(self):
        """Configure logging with both file and console handlers."""
        self.logger = logging.getLogger("transformer_evaluator")
        self.logger.setLevel(logging.INFO)

        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()

        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )

        # File handler (with rotation)
        log_dir = Path(self.config.dataset_config['output_dir']) / 'logs'
        log_dir.mkdir(exist_ok=True, parents=True)
        file_handler = RotatingFileHandler(
            log_dir / 'evaluator.log',
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(detailed_formatter)
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _setup_output_dir(self):
        """Create output directory if it doesn't exist."""
        output_dir = self.config.dataset_config['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Output directory set to: {output_dir}")

    def _load_models(self):
        """Load teacher and student models using transformers."""
        self.logger.info("Loading models...")
        try:
            # Load teacher model and tokenizer
            teacher_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_config['teacher_model_name'],
                device_map=self.config.model_config['device']
            )
            self.logger.debug(f"Teacher model loaded: {self.config.model_config['teacher_model_name']}")

            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_config['teacher_model_name']
            )
            self.logger.debug("Tokenizer loaded")

            # Load student model
            student_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_config['student_model_path'],
                device_map=self.config.model_config['device']
            )
            self.logger.debug(f"Student model loaded: {self.config.model_config['student_model_path']}")

            # Handle tokenizer padding
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                teacher_model.config.pad_token_id = tokenizer.pad_token_id
                student_model.config.pad_token_id = tokenizer.pad_token_id
                self.logger.debug("Padding token configured")

            return teacher_model, student_model, tokenizer

        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}", exc_info=True)
            raise

    def _generate_text(self, model, tokenizer, input_text: str) -> str:
        """Generate text using transformers generate() method."""
        try:
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            gen_config = self.config.generation_config
            self.logger.debug(f"Generating with config: {gen_config}")

            outputs = model.generate(
                **inputs,
                max_new_tokens=gen_config['max_new_tokens'],
                num_return_sequences=gen_config['num_return_sequences'],
                do_sample=gen_config['do_sample'],
                temperature=gen_config['temperature'],
                top_p=gen_config['top_p'],
                pad_token_id=tokenizer.pad_token_id
            )

            return tokenizer.decode(outputs[0], skip_special_tokens=True)

        except Exception as e:
            self.logger.error(f"Error during text generation: {str(e)}", exc_info=True)
            raise

    def _save_results(self, results: List[Dict[str, Any]], split_name: str):
        """Save evaluation results."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(self.config.dataset_config['output_dir']) / f"{split_name}_results_{timestamp}.json"

            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

            self.logger.info(f"Results saved to {output_file}")

        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}", exc_info=True)
            raise

    def evaluate(self,split_name = None):
        """Run evaluation on the dataset."""
        self.logger.info("Starting evaluation process")

        try:
            # Load models and tokenizer
            teacher_model, student_model, tokenizer = self._load_models()

            # Load dataset
            self.logger.info(f"Loading dataset: {self.config.dataset_name}")
            dataset = load_dataset(self.config.dataset_name, self.config.dataset_config)

            # Evaluate on train and test, validation splits
            DEFAULT_SPLITS = ['train', 'test', 'validation']
            splits = [split_name] if split_name else DEFAULT_SPLITS

            for split_name in splits:
                if split_name not in dataset:
                    self.logger.info(f"Skipping {split_name} split - not found in dataset")
                    continue

                self.logger.info(f"Evaluating on {split_name} split...")
                split_data = dataset[split_name].shuffle().select(range(self.config.num_samples))
                results = []

                for idx, example in enumerate(tqdm(split_data, desc=f"Evaluating {split_name}")):
                    self.logger.debug(f"Processing example {idx + 1}/{len(split_data)}")
                    input_text = example['input']

                    # Generate outputs
                    teacher_output = self._generate_text(teacher_model, tokenizer, input_text)
                    student_output = self._generate_text(student_model, tokenizer, input_text)

                    # Calculate metrics
                    metrics = self.metrics.compute_metrics(
                        teacher_model, student_model, tokenizer,
                        input_text, teacher_output, student_output
                    )

                    # Store results
                    results.append({
                        "input": input_text,
                        "teacher_output": teacher_output,
                        "student_output": student_output,
                        "metrics": {
                            "teacher_perplexity": metrics.teacher_perplexity,
                            "student_perplexity": metrics.student_perplexity,
                            "bleu_score": metrics.bleu_score
                        }
                    })

                # Save results
                self._save_results(results, split_name)

                # Print and log summary
                avg_metrics = {
                    "teacher_perplexity": np.mean([r["metrics"]["teacher_perplexity"] for r in results]),
                    "student_perplexity": np.mean([r["metrics"]["student_perplexity"] for r in results]),
                    "bleu_score": np.mean([r["metrics"]["bleu_score"] for r in results])
                }

                self.logger.info(f"\n{split_name.capitalize()} Split Summary:")
                for metric, value in avg_metrics.items():
                    self.logger.info(f"Average {metric}: {value:.4f}")

        except Exception as e:
            self.logger.error("Evaluation failed", exc_info=True)
            raise

        self.logger.info("Evaluation completed successfully")