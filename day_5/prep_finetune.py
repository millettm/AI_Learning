import pandas as pd
df = pd.read_csv('C:/Users/the-s/PycharmProjects/AI_Learning/data/qa_pairs.csv')
df = df.dropna(subset=['question', 'answer']).astype(str)
print(df.head())
df.to_json('C:/Users/the-s/PycharmProjects/AI_Learning/data/qa_pairs.jsonl', orient='records', lines=True)