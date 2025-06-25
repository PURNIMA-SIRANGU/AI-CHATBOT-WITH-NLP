# AI-CHATBOT-WITH-NLP

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: Sirangu Purnima

*DOMAIN*: PYTHON

*DURATION*: 4WEEKS

*MENTOR NAME*: NEELA SANTOSH

GITA: A Retrieval-Based Technical Chatbot

GITA (General-purpose Intelligent Technical Assistant) is a sophisticated yet easy-to-understand retrieval-based chatbot built entirely in Python. Developed within the Visual Studio Code environment, this project demonstrates core Natural Language Processing (NLP) principles to create a conversational agent that provides accurate, pre-defined answers on technical topics, making it an ideal digital assistant for developers and students.



Project Overview

Unlike generative models (like GPT) that create new text, GITA is a retrieval-based bot. Its intelligence lies in its ability to accurately "retrieve" the best-fit response from a structured intents.json knowledge base. The project operates in two main phases: a training phase where it learns from predefined query patterns, and an inference phase where it analyzes user messages to find and deliver the most appropriate response. This model is highly effective for applications where response accuracy and control are critical, such as FAQ bots and customer service agents.



Key Features

Intelligent Intent Recognition: Utilizes TF-IDF and Cosine Similarity to accurately map user input to the correct intent.
Natural Language Processing: Employs the NLTK library for essential NLP tasks like tokenization and lemmatization to understand user queries.
Advanced Vectorization: Uses n-grams to consider both single words and two-word phrases, significantly improving its contextual understanding.
Confidence Thresholding: Avoids giving incorrect answers by only responding when the similarity score for a match is above a set confidence level.
Modular and Scalable: The knowledge base is stored in a simple intents.json file, making it easy to add new knowledge without changing the core Python code.
Efficient and Lightweight: The entire trained model is saved into a single chatbot_model.pkl file for fast loading and execution.



The Technical Pipeline

The project is split into two scripts that handle the training and chatting logic.
Training (train.py): This script builds the bot's "brain." It reads all patterns from intents.json, preprocesses the text (tokenization and lemmatization), and then converts the patterns into a numerical matrix using a TF-IDF vectorizer. The final vectorizer, matrix, and corresponding tags are saved as chatbot_model.pkl.
Inference (chat.py): In a live chat, user input undergoes the same preprocessing. The loaded vectorizer transforms the input into a vector, and cosine_similarity calculates how closely it matches each trained pattern. If the best match exceeds the confidence threshold, the bot selects a corresponding response; otherwise, it provides a polite message stating it doesn't understand.
Technologies Used
Language: Python 3
Core Libraries: Scikit-learn (TF-IDF, Cosine Similarity), NLTK (Tokenization, Lemmatization), NumPy.
Data Handling: JSON (for intents), Pickle (for model serialization).
Development Environment: Visual Studio Code



Potential Applications

The architecture is highly versatile and can be adapted for numerous real-world applications, including:
Automated Customer Support & FAQ Bots
Educational Tutors for specific subjects
First-Level IT Helpdesk Assistants
Onboarding and training tools
Website Navigational Guides

OUTPUT:

![Image](https://github.com/user-attachments/assets/8640556a-03c0-44a6-838e-f56158cd36a8)
