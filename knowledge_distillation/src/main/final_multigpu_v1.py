import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os
import logging
import json
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model_name", default="defog/llama-3-sqlcoder-8b", type=str)
    parser.add_argument("--student_model_name", default="meta-llama/Llama-3.2-3B-Instruct", type=str)
    parser.add_argument("--dataset_name", default="lamini/spider_text_to_sql", type=str)
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--num_epochs", default=3, type=int)
    parser.add_argument("--warmup_steps", default=100, type=int)
    parser.add_argument("--output_dir", default="distilled_model", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--kd_ratio", default=0.5, type=float, help="Knowledge distillation loss ratio")
    parser.add_argument("--num_gpus", type=int, default=1)

    # Manually add DeepSpeed arguments
    parser.add_argument("--deepspeed", default=None, type=str,
                        help="Path to deepspeed config file")
    parser.add_argument("--deepspeed_config", default=None, type=str,
                        help="Path to deepspeed config json file")
    parser.add_argument("--zero_stage", default=None, type=int,
                        help="ZeRO optimization stage for DeepSpeed")

    args = parser.parse_args()
    return args


class LMDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, split):
        self.args = args
        self.tokenizer = tokenizer
        self.data = load_dataset(args.dataset_name, split=split)
        self.max_length = args.max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        sql_query = item['output'].strip()
        input_text = item['input']

        text = f"""Generate a SQL query to answer this question: `{input_text}`

The following SQL query best answers the question:
```sql
{sql_query}"""

        encoded = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors='pt')
        return {k: v.squeeze(0) for k, v in encoded.items()}

    def collate_fn(self, batch):
        return self.tokenizer.pad(batch, padding=True, return_tensors='pt')


def setup_model(args, model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    return model


def get_optimizer_grouped_parameters(model, weight_decay, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]


def train(args, student_model, teacher_model, train_dataset, tokenizer):
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate_fn)

    optimizer_grouped_parameters = get_optimizer_grouped_parameters(student_model, weight_decay=0.01)

    if args.deepspeed or args.deepspeed_config:
        import deepspeed
        ds_config = None
        if args.deepspeed_config:
            with open(args.deepspeed_config, 'r') as f:
                ds_config = json.load(f)
        elif args.deepspeed:
            with open(args.deepspeed, 'r') as f:
                ds_config = json.load(f)

        if ds_config is None:
            raise ValueError("DeepSpeed config file not found or invalid")

        student_model, optimizer, _, _ = deepspeed.initialize(
            model=student_model,
            model_parameters=optimizer_grouped_parameters,
            config=ds_config
        )
    else:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Remove the lr_scheduler setup

    student_model.train()
    teacher_model.eval()

    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")):
            batch = {k: v.to(student_model.device) for k, v in batch.items()}

            student_outputs = student_model(**batch, use_cache=False)
            with torch.no_grad():
                teacher_outputs = teacher_model(**batch)

            student_logits = student_outputs.logits
            teacher_logits = teacher_outputs.logits

            loss_ce = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), batch['input_ids'].view(-1),
                                      ignore_index=tokenizer.pad_token_id)
            loss_kd = F.kl_div(
                F.log_softmax(student_logits / args.temperature, dim=-1),
                F.softmax(teacher_logits / args.temperature, dim=-1),
                reduction='batchmean'
            ) * (args.temperature ** 2)

            loss = args.kd_ratio * loss_kd + (1 - args.kd_ratio) * loss_ce

            student_model.backward(loss)
            student_model.step()

            if step % 100 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{args.num_epochs} - Step {step}/{len(train_dataloader)} - Loss: {loss.item():.4f}")

        # Save checkpoint
        if args.local_rank == 0:
            student_model.save_checkpoint(args.output_dir, f"epoch_{epoch + 1}")


def main():
    args = get_args()

    if args.deepspeed or args.deepspeed_config:
        import deepspeed
        deepspeed.init_distributed()

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_name)

    train_dataset = LMDataset(args, tokenizer, "train")

    teacher_model = setup_model(args, args.teacher_model_name)
    student_model = setup_model(args, args.student_model_name)

    train(args, student_model, teacher_model, train_dataset, tokenizer)


if __name__ == "__main__":
    main()