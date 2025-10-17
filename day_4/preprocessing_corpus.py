import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import pandas as pd

nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_document(doc):
    if pd.isna(doc) or not isinstance(doc, str):
        return ""
    doc = str(doc).lower().strip()
    if not doc:
        return ""
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, flags=re.I | re.A)
    tokens = word_tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(filtered_tokens)

corpus_path = '/home/ubuntu/AI_Learning/data/combined_docs.csv'
try:
    corpus = pd.read_csv(corpus_path, sep=',', header=None, names=['Document'])
    corpus['Document'] = corpus['Document'].astype(str).replace('nan', '')
    corpus['Cleaned_Document'] = corpus['Document'].apply(preprocess_document)
    corpus = corpus[corpus['Cleaned_Document'].str.strip() != '']
    output_path = '/home/ubuntu/AI_Learning/data/cleaned_combined_docs.csv'
    corpus['Cleaned_Document'].to_csv(output_path, index=False)
    print(f"Cleaned corpus saved to {output_path}")
    print(corpus.head())
except Exception as e:
    print(f"Error processing corpus: {str(e)}")