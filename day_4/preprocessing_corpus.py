import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import pandas as pd

nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def preprocess_document(doc):
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I | re.A)
    doc = doc.lower()
    doc = doc.strip()
    tokens = word_tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    doc = ' '.join(filtered_tokens)
    return doc


corpus = pd.read_csv(
    'C:/Users/the-s/PycharmProjects/AI_Learning/data/combined_docs.csv',
    sep='"', header=None, names=['Document'])

corpus['Cleaned_Document'] = corpus['Document'].apply(preprocess_document)

# Save cleaned data
corpus['Cleaned_Document'].to_csv('C:/Users/the-s/PycharmProjects/AI_Learning/data/cleaned_combined_docs.csv', index=False)

print(corpus)