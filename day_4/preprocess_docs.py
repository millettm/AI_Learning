import os
import nltk
import pandas as pd

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load multiple sources
files = []
documents = []
directory_path = 'C:/Users/the-s/PycharmProjects/AI_Learning/training_subset/'

def get_training_files_in_directory_os(directory_path):
    for training_doc in os.listdir(directory_path):
        file_path = os.path.join(directory_path, training_doc)
        if os.path.isfile(file_path):  # Check if it's a file
            files.append(training_doc)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:  # Use file_path instead of directory_path
                    text = f.read()
                    sentences = nltk.sent_tokenize(text)
                    documents.extend(sentences)  # Simply extend with sentences
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return files

# Call the function
get_training_files_in_directory_os(directory_path)

# Save cleaned data
df_out = pd.DataFrame({'text': documents})  # Use documents directly
df_out.to_csv('C:/Users/the-s/PycharmProjects/AI_Learning/data/combined_docs.csv', index=False)
print("Docs ready:", len(documents))