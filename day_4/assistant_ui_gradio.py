import gradio as gr
import pandas as pd
import nltk
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core import PromptTemplate
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Set up Ollama
Settings.llm = Ollama(model='llama3.2:3b', base_url='http://127.0.0.1:11434', request_timeout=1800.0, options={'num_gpu': 0})
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device='cpu')

# Load CSV
df = pd.read_csv('C:/Users/the-s/PycharmProjects/AI_Learning/data/cleaned_combined_docs.csv')
df = df.dropna(subset=['text']).astype(str)
documents = [Document(text=row['text']) for index, row in df.iterrows()]

# Query template
query_template = PromptTemplate(
    "You are an IRS leadership trainer. Context: {context_str}\n"
    "Few-shot example: Query: Explain Four Step Model. Response: Step 1: Plan Expectations ... (IRM 6.430.1).\n"
    "Query: {query_str}\n"
    "Respond step-by-step, citing IRM sections. Extract NLP key terms like 'planning', 'monitoring'."
)

v_index = VectorStoreIndex.from_documents(documents)
query_engine = v_index.as_query_engine(text_qa_template=query_template)

def query_rag(query: str) -> str:
    return str(query_engine.query(query))

rag_tool = FunctionTool.from_defaults(fn=query_rag)

agent = ReActAgent(
    tools=[rag_tool],
    llm=Settings.llm,
    verbose=True,
    max_iterations=100,
    system_prompt="You are an IRS leadership trainer. Use query_rag to fetch data from multiple sources, then generate items like quizzes. Output 3 multiple-choice questions with 4 options and answers for quizzes, citing IRM if possible."
)

def ui_query(query):
    try:
        response = agent.run(query)
        sentiment = nltk.FreqDist(nltk.word_tokenize(str(response).lower())).most_common(5)
        return f"Response: {response}\nSentiment: {sentiment}\nKeywords: {nltk.FreqDist(nltk.word_tokenize(str(response).lower())).most_common(5)}"
    except Exception as e:
        return f"Error: {str(e)}"

iface = gr.Interface(fn=ui_query, inputs="text", outputs="text", title="IRS Leadership Assistant")
iface.launch()