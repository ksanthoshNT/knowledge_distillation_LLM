# metrics.py
from dataclasses import dataclass
from typing import List, Dict
import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import nltk


@dataclass
class MetricOutput:
    """Stores metric calculation results."""
    teacher_perplexity: float
    student_perplexity: float
    bleu_score: float


class MetricsCalculator:
    """Calculates evaluation metrics using transformers-compatible methods."""

    def __init__(self):
        nltk.download('punkt', quiet=True)

    def compute_perplexity(self, model, tokenizer, text: str) -> float:
        """Compute perplexity using transformers model."""
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])

        return torch.exp(outputs.loss).item()

    def compute_bleu(self, reference: str, candidate: str) -> float:
        """Compute BLEU score between reference and candidate texts."""
        reference_tokens = nltk.word_tokenize(reference)
        candidate_tokens = nltk.word_tokenize(candidate)
        return sentence_bleu([reference_tokens], candidate_tokens)

    def compute_metrics(self, teacher_model, student_model, tokenizer,
                        input_text: str, teacher_output: str, student_output: str) -> MetricOutput:
        """Compute all metrics for a given sample."""
        return MetricOutput(
            teacher_perplexity=self.compute_perplexity(teacher_model, tokenizer, input_text),
            student_perplexity=self.compute_perplexity(student_model, tokenizer, input_text),
            bleu_score=self.compute_bleu(teacher_output, student_output)
        )

