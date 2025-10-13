import streamlit as st
import pandas as pd
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core import PromptTemplate
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import requests
import json
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# Set up Streamlit page
st.title("IRS Leadership Assistant")
st.write("Query performance management or request quizzes/simulations (e.g., 'Explain Weingarten Rights' or '3-question quiz on feedback').")

# Set up Ollama explicitly
ollama_llm = Ollama(model='llama3.2:3b', base_url='http://127.0.0.1:11434', request_timeout=1800.0, options={'num_gpu': 0})
Settings.llm = ollama_llm  # Force Ollama, no OpenAI fallback
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device='cpu')

# Pre-flight Ollama check
try:
    test_response = ollama_llm.complete("Test connection")
    st.write("Ollama connection successful!")
except Exception as e:
    st.error(f"Ollama connection failed: {str(e)}")

# Load CSV
df = pd.read_csv('C:/Users/the-s/PycharmProjects/AI_Learning/data/cleaned_combined_docs.csv')
df = df.dropna(subset=['text']).astype(str)

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

# Direct Ollama API call (fallback if agent fails)
def call_ollama(prompt, model='llama3.2:3b'):
    url = 'http://127.0.0.1:11434/api/generate'
    payload = {
        'model': model,
        'prompt': prompt,
        'stream': False,
        'options': {'num_gpu': 0, 'temperature': 0.7}
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()['response']
    else:
        return f"Error: {response.status_code} - {response.text}"

# Agent setup
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

# Streamlit UI
query = st.text_input("Enter your query:", placeholder="e.g., 'Explain IRS Performance Management' or '3-question quiz on feedback'")
if st.button("Submit"):
    with st.spinner("Processing..."):
        try:
            # Use direct Ollama call with RAG context
            rag_context = str(query_engine.query(query))
            full_prompt = query_template.format(context_str=rag_context, query_str=query)
            response = call_ollama(full_prompt)
            st.write("**Response:**")
            st.write(response)
            # Sentiment and keyword analysis
            sia = SentimentIntensityAnalyzer()
            st.write("**Sentiment Analysis:**", sia.polarity_scores(response))
            st.write("**Top Keywords:**", nltk.FreqDist(nltk.word_tokenize(response.lower())).most_common(5))
        except Exception as e:
            st.error(f"Error: {str(e)}")