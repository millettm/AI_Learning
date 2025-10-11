import asyncio
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core import PromptTemplate
import pandas as pd
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Set up Ollama for LLM and embeddings
Settings.llm = Ollama(model='mistral:instruct', base_url='http://127.0.0.1:11434', request_timeout=300.0, options={'num_gpu': 0})  # Increased timeout
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Load CSV with Pandas
df = pd.read_csv('C:/Users/the-s/PycharmProjects/AI_Learning/data/irs_intro_to_perf_mgt.csv')

# Convert rows to Documents (assuming 'sentence' column; swap if different)
documents = [Document(text=row['sentence']) for index, row in df.iterrows()]

# Create a query template
query_template = PromptTemplate(
    "You are an IRS leadership trainer. Context: {context_str}\n"
    "Few-shot example: Query: Explain the Four Step Model. Response: Step 1: Plan Expectations ... (from IRM 6.430.1).\n"
    "Query: {query_str}\n"
    "Respond step-by-step, citing IRM sections if available. Use NLP to extract key terms like 'planning' or 'monitoring'."
)

# Create index
v_index = VectorStoreIndex.from_documents(documents)

# Create query engine
query_engine = v_index.as_query_engine(text_qa_template=query_template)

def query_rag(query: str) -> str:
    """Query IRS performance management data."""
    return str(query_engine.query(query))

rag_tool = FunctionTool.from_defaults(fn=query_rag)

agent = ReActAgent(
    tools=[rag_tool],
    llm=Settings.llm,
    verbose=True,
    max_iterations=10,  # Safety limit
    system_prompt="You are an IRS leadership trainer. Always use the query_rag tool to fetch data first, then generate any requested items like quizzes or simulations step-by-step. For quizzes, always create exactly 3 questions with answers."
)

async def main():
    print("Starting agent chat...")  # Diagnostic
    response = await agent.run("Explain IRS Performance Management, then generate a 3-question quiz on feedback.")  # Use arun for async
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
