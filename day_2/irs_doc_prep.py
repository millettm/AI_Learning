import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')

# Load or parse IRS text
file_path = 'C:/Users/the-s/OneDrive/Documents/Part 6. Human Resources Management.txt'

try:
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# Extract sentences, words, and key terms
sentences = sent_tokenize(text)
words = word_tokenize(text.lower())
stop_words = set(stopwords.words('english'))
key_terms = [w for w in words if w.isalpha() and w not in stop_words]

# Frequency analysis
freq_dist = nltk.FreqDist(key_terms)
print("Top 5 terms: ", freq_dist.most_common(5))

# Save to DataFrame
df = pd.DataFrame({
    'sentence': sentences,
    'key terms': [','.join([w for w in word_tokenize(s.lower()) if w in key_terms]) for s in sentences]
})
df.to_csv('C:/Users/the-s/PycharmProjects/AI_Learning/data/irs_intro_to_perf_mgt.csv', index=False)
print(df.head())