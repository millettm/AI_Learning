from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import torch
import pandas as pd
import json
import sys

sys.setrecursionlimit(4000)

# Load dataset
with open('C:/Users/the-s/PycharmProjects/AI_Learning/data/qa_pairs.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

# Format for LoRA
def format_prompt(example):
    return f"### Question: {example['question']}\n### Answer: {example['answer']}<|endoftext|>"

data = [format_prompt(d) for d in data]
dataset = Dataset.from_list([{'text': d} for d in data])

# Load model and tokenizer
model_name = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32, device_map="cpu")

# LoRA config
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, lora_config)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=512)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training args
training_args = TrainingArguments(
    output_dir="C:/Users/the-s/PycharmProjects/AI_Learning/finetuned_peft",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    warmup_steps=1,
    logging_steps=1,
    save_steps=10,
    fp16=False,
    dataloader_num_workers=0
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# Train
trainer.train()

# Save
model.save_pretrained("C:/Users/the-s/PycharmProjects/AI_Learning/finetuned_peft")
tokenizer.save_pretrained("C:/Users/the-s/PycharmProjects/AI_Learning/finetuned_peft")
print("Fine-tuning complete!")