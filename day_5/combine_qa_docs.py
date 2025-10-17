import pandas as pd
from transformers import AutoTokenizer
import json
import os

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
tokenizer.pad_token = tokenizer.eos_token

# Load document
csv_path = 'C:/Users/the-s/PycharmProjects/AI_Learning/data/cleaned_combined_docs.csv'
try:
    df = pd.read_csv(csv_path)
    # Ensure Cleaned_Document column exists
    if 'Cleaned_Document' not in df.columns:
        raise KeyError("Column 'Cleaned_Document' not found in CSV. Check CSV header.")
    # Convert to string, handle non-string values
    df['Cleaned_Document'] = df['Cleaned_Document'].astype(str).replace('nan', '')
    texts = [text for text in df['Cleaned_Document'].tolist() if text.strip()]
except Exception as e:
    print(f"Error loading CSV: {str(e)}")
    exit(1)

# Chunk text (512 tokens max)
max_length = 512
doc_examples = []
for text in texts:
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        for i in range(0, len(tokens), max_length - 50):  # 50 tokens for prompt
            chunk = tokens[i:i + max_length - 50]
            chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
            if chunk_text.strip():  # Skip empty chunks
                doc_examples.append({
                    "instruction": "You are an IRS leadership trainer. Provide a detailed explanation based on IRS guidelines.",
                    "input": "Provide context from IRS performance management or veterans employment guidelines.",
                    "output": chunk_text
                })
    except Exception as e:
        print(f"Error tokenizing text: {str(e)}")
        continue

# Load existing Q&A pairs
qa_path = 'C:/Users/the-s/PycharmProjects/AI_Learning/data/qa_pairs.jsonl'
if not os.path.exists(qa_path):
    print(f"Error: Q&A file {qa_path} not found.")
    exit(1)
with open(qa_path, 'r') as f:
    qa_pairs = [json.loads(line) for line in f]
qa_examples = [{"instruction": "You are an IRS leadership trainer. Provide a precise answer citing relevant IRM sections.", "input": pair['question'], "output": pair['answer']} for pair in qa_pairs]

# Combine datasets
combined_examples = qa_examples + doc_examples

# Save combined dataset
output_path = 'C:/Users/the-s/PycharmProjects/AI_Learning/data/combined_dataset.jsonl'
with open(output_path, 'w') as f:
    for example in combined_examples:
        f.write(json.dumps(example) + '\n')

print(f"Generated {len(combined_examples)} training examples.")