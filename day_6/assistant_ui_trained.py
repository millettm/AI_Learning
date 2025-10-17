import streamlit as st
import pandas as pd
from llama_index.core import PromptTemplate, Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import nltk
import os
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# Set up Streamlit page
st.title("IRS Leadership Assistant")
st.write("Query performance management or request quizzes/simulations (e.g., 'Explain Weingarten Rights' or '3-question quiz on feedback').")

# Set up embeddings
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device='cpu')

# Load CSV
df = pd.read_csv('C:/Users/the-s/PycharmProjects/AI_Learning/data/cleaned_combined_docs.csv')
documents = [Document(text=row['Cleaned_Document']) for index, row in df.iterrows()]

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

# Load fine-tuned model
finetuned_path = "C:/Users/the-s/PycharmProjects/AI_Learning/finetuned_peft"
if not os.path.exists(finetuned_path):
    st.error(f"Error: Fine-tuned model directory {finetuned_path} not found. Rerun finetune_peft.py.")
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B",
        dtype=torch.float32,
        device_map="cpu"
    )
    model = PeftModel.from_pretrained(base_model, finetuned_path)
    tokenizer = AutoTokenizer.from_pretrained(finetuned_path)

    def query_finetuned(query):
        rag_context = str(query_engine.query(query))
        full_prompt = f"You are an IRS leadership trainer. Provide a precise answer citing relevant IRM sections. {query_template.format(context_str=rag_context, query_str=query)}"
        inputs = tokenizer(full_prompt, return_tensors="pt").to("cpu")
        outputs = model.generate(**inputs, max_new_tokens=500, do_sample=True, temperature=0.3)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Streamlit UI
    query = st.text_input("Enter your query:", placeholder="e.g., 'Explain IRS Performance Management' or '3-question quiz on feedback'")
    if st.button("Submit"):
        with st.spinner("Processing..."):
            try:
                response = query_finetuned(query)
                st.write("**Response:**")
                st.write(response)
                sia = SentimentIntensityAnalyzer()
                st.write("**Sentiment Analysis:**", sia.polarity_scores(response))
                st.write("**Top Keywords:**", nltk.FreqDist(nltk.word_tokenize(response.lower())).most_common(5))
            except Exception as e:
                st.error(f"Error: {str(e)}")
