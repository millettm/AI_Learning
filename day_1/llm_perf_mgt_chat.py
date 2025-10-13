import ollama
from ollama import Client

# Load text from CSV (links to perf_mgt_sample.csv)
import pandas as pd
df = pd.read_csv('C:/Users/the-s/PycharmProjects/AI_Learning/data/cleaned_combined_docs.csv')

# Filter out NaN and non-string values
df = df.dropna(subset=['text'])  # Drop rows where 'text' is NaN
df['text'] = df['text'].astype(str)  # Ensure all are strings

prompt = f"""
You are an IRS leadership trainer. Using this Performance Management snippet: '{df['text']}'
Provide a brief overview of the options, actions, and strategies available to managers in relation to poor performers.
Cite relevant portions of the IRM whenever possible.
Generate a 3-question multiple-choice quiz highlighting key points after you finish providing the overview.
"""

# Generate with explicit host
client = Client(host='http://127.0.0.1:11434')
response = client.generate(model='llama3.2', prompt=prompt)
print(response['response'])
