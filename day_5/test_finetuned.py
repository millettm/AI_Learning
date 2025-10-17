from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

finetuned_path = "C:/Users/the-s/PycharmProjects/AI_Learning/finetuned_peft"

try:
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B",
        dtype=torch.float32,
        device_map="cpu"
    )
    model = PeftModel.from_pretrained(base_model, finetuned_path)
    tokenizer = AutoTokenizer.from_pretrained(finetuned_path)

    # Test queries with stricter prompt
    queries = [
        ("What are Weingarten Rights?", "You are an IRS leadership trainer. Provide a precise answer citing IRM 6.752.1. ### Question: What are Weingarten Rights?\n### Answer:"),
        ("Explain the Four Step Model", "You are an IRS leadership trainer. Provide a step-by-step explanation citing IRM 6.430.1. ### Question: What are the steps in the 'Four Step Model' of performance management based on the IRM?\n### Answer:"),
        ("What is feedback’s role in performance management?", "You are an IRS leadership trainer. Provide a brief, detailed answer citing IRM 6.430.3. ### Question: What is feedback’s role in performance management?\n### Answer:"),
        ("Why are the Four Steps in performance management important?", "You are an IRS leadership trainer.  Provide a detailed answer citing IRM 6.430.1. ### Question: Why are the Four Step Model steps of Planning, Monitoring, Evaluating, and Recognizing important to performance management?\n### Answer:")
    ]

    for query, prompt in queries:
        print(f"\nTesting: {query}")
        inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
        outputs = model.generate(**inputs, max_new_tokens=777, do_sample=True, temperature=0.333)  # Increased tokens, lower temp
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Response:", response)
        sia = SentimentIntensityAnalyzer()
        print("Sentiment:", sia.polarity_scores(response))
        print("Keywords:", nltk.FreqDist(nltk.word_tokenize(response.lower())).most_common(5))
except Exception as e:
    print(f"Error: {str(e)}")
