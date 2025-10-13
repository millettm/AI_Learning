import asyncio
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core import PromptTemplate
import pandas as pd
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Set up Ollama
Settings.llm = Ollama(model='llama3.2:3b', base_url='http://127.0.0.1:11434', request_timeout=1200.0) #, options={'num_gpu': 0})
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Load CSV
df = pd.read_csv('C:/Users/the-s/PycharmProjects/AI_Learning/data/cleaned_combined_docs.csv')

# Filter out NaN and non-string values
df = df.dropna(subset=['text'])  # Drop rows where 'text' is NaN
df['text'] = df['text'].astype(str)  # Ensure all are strings

# Convert to Documents
documents = [Document(text=row['text']) for index, row in df.iterrows()]

# Query template
query_template = PromptTemplate(
    "You are an IRS leadership trainer. Context: {context_str}\n"
    "Few-shot example: Query: Explain Four Step Model. Response: Step 1: Plan Expectations ... (IRM 6.430.1).\n"
    "Query: {query_str}\n"
    "Respond step-by-step, citing IRM sections. Extract NLP key terms like 'planning', 'monitoring'."
)

# Create index and query engine
v_index = VectorStoreIndex.from_documents(documents)
query_engine = v_index.as_query_engine(text_qa_template=query_template)

def query_rag(query: str) -> str:
    """Query IRS performance management data."""
    return str(query_engine.query(query))

rag_tool = FunctionTool.from_defaults(fn=query_rag)

agent = ReActAgent(
    tools=[rag_tool],
    llm=Settings.llm,
    verbose=True,
    max_iterations=100,
    system_prompt="You are an IRS leadership trainer. Use query_rag to fetch data from multiple sources, then generate items like quizzes. Output 3 multiple-choice questions with 4 options and answers for quizzes, citing IRM if possible."
)

async def main():
    print("Starting agent chat...")
    response = await agent.run("Explain the Performance Management System in the IRS, then provide a 3-question quiz.")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
