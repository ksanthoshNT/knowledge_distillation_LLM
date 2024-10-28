from transformers import AutoModelForCausalLM, AutoTokenizer, AwqConfig
import torch

# Load model and tokenizer
model_path = "/home/data_science/project_files/santhosh/knowledge_distillation_LLM/knowledge_distillation/src/main/distillation/llama3-8b-awq-distilled-f32"

# Create AWQ config for CPU
quantization_config = AwqConfig(version="ipex")

# Load model with CPU configuration
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    # quantization_config=quantization_config,
    device_map="cpu",
    torch_dtype=torch.float32
)
tokenizer = AutoTokenizer.from_pretrained("aspenita/llama-3-sqlcoder-8b-AWQ")

# Define prompt
# Define prompt
prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n        Generate a SQL query to answer this question: `WHAT IS THE PREDICTION OF COLUMN source1_Uid WITH VALUE 9052`\n        Use the provided DDL statements to formulate your query. \n\n        DDL statements:\n        CREATE TABLE jim_ntngai_com_6712016e35c9b20eccb9052b_V1 (\nsource1_year_target BIGINT,\nsource1_Uid BIGINT,\nsource1_age BIGINT,\nsource1_Pid BIGINT,\nsource1_subscribe BIGINT,\nsource1_MARRIAGE BIGINT,\nsource1_EDUCATION BIGINT,\nprobability_0 DOUBLE PRECISION,\nprobability_1 DOUBLE PRECISION,\nprediction BIGINT\n);<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n        The following SQL query best answers the question `WHAT IS THE PREDICTION OF COLUMN source1_Uid WITH VALUE 9052`:\n        ```sql\n"""

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=40,
        top_p=0.95,
        repetition_penalty=1.1
    )

# Decode and print result
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)