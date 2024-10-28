from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, Any

app = FastAPI()

# Load model and tokenizer
MODEL_PATH = "/home/data_science/project_files/santhosh/knowledge_distillation_LLM/knowledge_distillation/src/main/distillation/llama3-8b-awq-distilled-f32"
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).cuda()
tokenizer = AutoTokenizer.from_pretrained("aspenita/llama-3-sqlcoder-8b-AWQ")


class GenerateRequest(BaseModel):
    model: str
    prompt: str
    options: Dict[str, Any]


@app.post("/api/generate")
async def generate(request: GenerateRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to('cuda')

    generate_kwargs = {
        'max_new_tokens': request.options.get('num_predict', 5),
        'do_sample': request.options.get('do_sample', False),
        'temperature': request.options.get('temperature', 0.0),
        'top_k': request.options.get('top_k', 40),
        'top_p': request.options.get('top_p', 0.95),
        'repetition_penalty': request.options.get('repeat_penalty', 1.1)
    }

    outputs = model.generate(**inputs, **generate_kwargs)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "model": request.model,
        "prompt": request.prompt,
        "response": result,
        "done": True
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=11435)