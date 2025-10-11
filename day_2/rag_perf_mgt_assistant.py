import pandas as pd
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Set up Ollama for LLM and embeddings
Settings.llm = Ollama(model='llama3.2:1b', base_url='http://127.0.0.1:11434', options={'num_gpu': 0})
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Load CSV with Pandas
df = pd.read_csv('C:/Users/the-s/PycharmProjects/AI_Learning/data/irs_intro_to_perf_mgt.csv')

# Convert rows to Documents (assuming 'sentence' column; swap if different)
documents = []
for index, row in df.iterrows():
    documents.append(Document(text=row['sentence']))

# Create index
v_index = VectorStoreIndex.from_documents(documents)

# Query
query_engine = v_index.as_query_engine()
response = query_engine.query("Explain IRS Performance Management in a performance discussion, based on IRS guidelines.")
print(response)