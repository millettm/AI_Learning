import ollama
from ollama import Client

# Load text from CSV (links to perf_mgt_sample.csv)
import pandas as pd
df = pd.read_csv('C:/Users/the-s/PycharmProjects/AI_Learning/data/perf_mgt_sample.csv')
text = ' '.join(df['sentences'])

prompt = f"""
You are an IRS leadership trainer. Using this Performance Management snippet: '{text}'
Explain the Four Step Model to new managers, based on Performance Management guidelines.
Keep it concise, step-by-step.
"""

# Generate with explicit host
client = Client(host='http://127.0.0.1:11434')
response = client.generate(model='llama3.2', prompt=prompt)
print(response['response'])
