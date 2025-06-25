import json
import random
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load Pre-trained Model and Data ---
print("Loading trained model...")
lemmatizer = WordNetLemmatizer()

try:
    with open('chatbot_model.pkl', 'rb') as f:
        vectorizer, X_train, tags = pickle.load(f)
except FileNotFoundError:
    print("\nError: Model file 'chatbot_model.pkl' not found.")
    print("Please run the 'train.py' script first to create the model.")
    exit()

with open('intents.json', 'r') as file:
    intents = json.load(file)['intents']

# --- Chatbot Logic ---

def get_response(user_input):
    """
    Finds the most relevant response for a given user input.
    """
    # 1. Preprocess the user input (same as in training)
    user_tokens = nltk.word_tokenize(user_input)
    user_lemmas = [lemmatizer.lemmatize(word.lower()) for word in user_tokens]
    processed_user_input = ' '.join(user_lemmas)
    
    # 2. Transform user input using the loaded vectorizer
    user_vector = vectorizer.transform([processed_user_input])
    
    # 3. Calculate cosine similarity between user input and all training patterns
    similarities = cosine_similarity(user_vector, X_train)
    
    # 4. Find the index of the most similar pattern
    most_similar_index = np.argmax(similarities)
    
    # 5. Check if the similarity is above a confidence threshold
    confidence_score = similarities[0][most_similar_index]
    
    if confidence_score > 0.4: # You can adjust this threshold
        matching_tag = tags[most_similar_index]
        
        # 6. Find the corresponding intent and return a random response
        for intent in intents:
            if intent['tag'] == matching_tag:
                return random.choice(intent['responses'])
    else:
        # If no confident match is found
        return "I'm sorry, I don't quite understand. Can you ask me about Programming language details or our hours?"

# --- Main Chat Loop ---

if __name__ == "__main__":
    print("\GITA the Chatbot is now live! Type 'quit' to exit.")
    
    while True:
        user_message = input("You: ")
        if user_message.lower() == 'quit':
            print("Bot: Goodbye! Have a great day.")
            break
        
        response = get_response(user_message)
        print(f"Bot: {response}")