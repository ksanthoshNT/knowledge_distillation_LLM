import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import nltk

# Download necessary NLTK data
nltk.download('punkt')


def load_model(model_name, device):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def generate_text(model, tokenizer, input_text, max_length=50):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, num_return_sequences=1, do_sample=True, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)



def calculate_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()


def calculate_bleu(reference, candidate):
    reference_tokens = nltk.word_tokenize(reference)
    candidate_tokens = nltk.word_tokenize(candidate)
    return sentence_bleu([reference_tokens], candidate_tokens)


def evaluate_models(teacher_model, student_model, tokenizer, dataset, num_samples=10):
    results = []
    for i, example in enumerate(dataset.shuffle().take(num_samples)):
        question = example['question']
        query = example['query']
        input_text = f"Question: {question}\nSQL Query: {query}"

        teacher_output = generate_text(teacher_model, tokenizer, input_text)
        student_output = generate_text(student_model, tokenizer, input_text)

        teacher_perplexity = calculate_perplexity(teacher_model, tokenizer, input_text)
        student_perplexity = calculate_perplexity(student_model, tokenizer, input_text)

        bleu_score = calculate_bleu(teacher_output, student_output)

        results.append({
            'input': input_text,
            'teacher_output': teacher_output,
            'student_output': student_output,
            'teacher_perplexity': teacher_perplexity,
            'student_perplexity': student_perplexity,
            'bleu_score': bleu_score
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model_name", default="meta-llama/Llama-3.2-3B-Instruct", type=str)
    parser.add_argument("--student_model_path", default="./distilled_model", type=str)
    parser.add_argument("--dataset_name", default="databricks/databricks-dolly-15k", type=str)
    parser.add_argument("--dataset_config_name", default=None, type=str)
    parser.add_argument("--num_samples", default=10, type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading models...")
    teacher_model, tokenizer = load_model(args.teacher_model_name, device)
    student_model, _ = load_model(args.student_model_path, device)
    student_model.config.pad_token_id = tokenizer.pad_token_id  # pad token

    print("Loading dataset...")
    train_dataset = load_dataset(args.dataset_name, args.dataset_config_name, split="train")
    test_dataset = load_dataset(args.dataset_name, args.dataset_config_name, split="validation")

    print("Evaluating on training data...")
    train_results = evaluate_models(teacher_model, student_model, tokenizer, train_dataset, args.num_samples)

    print("Evaluating on test data...")
    test_results = evaluate_models(teacher_model, student_model, tokenizer, test_dataset, args.num_samples)

    print("\nResults on Training Data:")
    for i, result in enumerate(train_results):
        print(f"\nSample {i + 1}:")
        print(f"Input: {result['input']}")
        print(f"Teacher Output: {result['teacher_output']}")
        print(f"Student Output: {result['student_output']}")
        print(f"Teacher Perplexity: {result['teacher_perplexity']:.2f}")
        print(f"Student Perplexity: {result['student_perplexity']:.2f}")
        print(f"BLEU Score: {result['bleu_score']:.4f}")

    print("\nResults on Test Data:")
    for i, result in enumerate(test_results):
        print(f"\nSample {i + 1}:")
        print(f"Input: {result['input']}")
        print(f"Teacher Output: {result['teacher_output']}")
        print(f"Student Output: {result['student_output']}")
        print(f"Teacher Perplexity: {result['teacher_perplexity']:.2f}")
        print(f"Student Perplexity: {result['student_perplexity']:.2f}")
        print(f"BLEU Score: {result['bleu_score']:.4f}")

    # Calculate and print average metrics
    avg_train_teacher_perplexity = np.mean([r['teacher_perplexity'] for r in train_results])
    avg_train_student_perplexity = np.mean([r['student_perplexity'] for r in train_results])
    avg_train_bleu = np.mean([r['bleu_score'] for r in train_results])

    avg_test_teacher_perplexity = np.mean([r['teacher_perplexity'] for r in test_results])
    avg_test_student_perplexity = np.mean([r['student_perplexity'] for r in test_results])
    avg_test_bleu = np.mean([r['bleu_score'] for r in test_results])

    print("\nAverage Metrics:")
    print(f"Training Data - Teacher Perplexity: {avg_train_teacher_perplexity:.2f}")
    print(f"Training Data - Student Perplexity: {avg_train_student_perplexity:.2f}")
    print(f"Training Data - BLEU Score: {avg_train_bleu:.4f}")
    print(f"Test Data - Teacher Perplexity: {avg_test_teacher_perplexity:.2f}")
    print(f"Test Data - Student Perplexity: {avg_test_student_perplexity:.2f}")
    print(f"Test Data - BLEU Score: {avg_test_bleu:.4f}")


if __name__ == "__main__":
    main()