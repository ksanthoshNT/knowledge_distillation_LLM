from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_path: str = "/home/data_science/project_files/santhosh/knowledge_distillation_LLM/distilled_model"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("aspenita/llama-3-sqlcoder-8b-AWQ")

# Generate text
prompt = "who are you"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=5)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)