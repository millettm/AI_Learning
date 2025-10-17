from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import torch
import json
import os

os.environ["HF_TOKEN"] = "hf_xxxxxxxxxxxxxxxx"
with open('/home/ubuntu/AI_Learning/data/combined_dataset.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

def format_prompt(example):
    return f"{example['instruction']}\n### Input: {example['input']}\n### Output: {example['output']}<|endoftext|>"

data = [format_prompt(d) for d in data]
dataset = Dataset.from_list([{'text': d} for d in data])

model_name = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, lora_config)

def tokenize_function(examples):
    tokenized = tokenizer(examples['text'], truncation=True, padding="max_length", max_length=512)
    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="/home/ubuntu/AI_Learning/finetuned_peft",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    num_train_epochs=5,
    warmup_steps=1,
    logging_steps=1,
    save_steps=10,
    fp16=True,
    dataloader_num_workers=4
)

# Import Trainer after dependencies
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

trainer.train()
model.save_pretrained("/home/ubuntu/AI_Learning/finetuned_peft")
tokenizer.save_pretrained("/home/ubuntu/AI_Learning/finetuned_peft")
print("Fine-tuning complete!")