import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

print("Starting training process...")

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# --- Section for NLTK data download ---
# A more robust way to check and download NLTK data
def download_nltk_data():
    data_to_download = {
        'punkt': 'tokenizers/punkt',
        'wordnet': 'corpora/wordnet',
        'punkt_tab': 'tokenizers/punkt_tab'  # Added this to fix the LookupError
    }

    for name, path in data_to_download.items():
        try:
            nltk.data.find(path)
            print(f"NLTK '{name}' is already downloaded.")
        except LookupError:
            print(f"Downloading NLTK '{name}'...")
            # Use 'all' for wordnet to get omw-1.4 as well, which can be a dependency
            download_target = 'all' if name == 'wordnet' else name
            nltk.download(download_target, quiet=True)
            print(f"'{name}' downloaded successfully.")

# Call the function to ensure data is present
download_nltk_data()

# Load intents file
with open('intents.json', 'r') as file:
    intents = json.load(file)['intents']

# --- Step 1: Preprocessing the data ---
print("\nStep 1: Preprocessing data...")

documents = []
tags = []

for intent in intents:
    for pattern in intent['patterns']:
        # Tokenize and lemmatize each word in the sentence
        word_list = nltk.word_tokenize(pattern)
        lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in word_list]
        
        # Join the words back into a sentence string
        processed_pattern = ' '.join(lemmatized_words)
        
        # Store the processed pattern and its tag
        documents.append(processed_pattern)
        tags.append(intent['tag'])

print(f"Processed {len(documents)} patterns.")

# --- Step 2: Vectorization using TF-IDF ---
print("\nStep 2: Vectorizing text data with TF-IDF...")

vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
X = vectorizer.fit_transform(documents)

print(f"Created a document-term matrix of shape: {X.shape}")

# --- Step 3: Saving the trained objects ---
print("\nStep 3: Saving the trained model and associated data...")

with open('chatbot_model.pkl', 'wb') as f:
    pickle.dump((vectorizer, X, tags), f)

print("\nTraining complete! The model 'chatbot_model.pkl' has been created.")