from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# Set Hugging Face token
import os
os.environ["HF_TOKEN"] = "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"  # Your token

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    torch_dtype=torch.float32,
    device_map="cpu"
)
model = PeftModel.from_pretrained(base_model, "C:/Users/the-s/PycharmProjects/AI_Learning/finetuned_peft")
tokenizer = AutoTokenizer.from_pretrained("C:/Users/the-s/PycharmProjects/AI_Learning/finetuned_peft")

# Test query
prompt = "### Question: What should a new manager do when an employee has poor performance?\n### Answer:"
inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)

# Sentiment and keyword analysis
sia = SentimentIntensityAnalyzer()
print("Sentiment:", sia.polarity_scores(response))
print("Keywords:", nltk.FreqDist(nltk.word_tokenize(response.lower())).most_common(5))
