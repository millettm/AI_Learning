import pandas as pd
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import PromptTemplate

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

# Custom template with NLP-inspired structure (e.g., few-shot examples)
query_template = PromptTemplate(
    f"You are an IRS leadership trainer. Context: {context_str}\n"
    "Few-shot example: Query: Explain the Four Step Model. Response: Step 1: Plan Expectations ... (from IRM 6.430.1).\n"
    "Query: {query_str}\n"
    "Respond step-by-step, citing IRM sections if available. Use NLP to extract key terms like 'planning' or 'monitoring'."
)
query_engine = v_index.as_query_engine(text_qa_template=query_template)

# Test
response = query_engine.query("Explain IRS Performance Management to new managers, including feedback and goals examples.")
print(response)
