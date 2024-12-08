import io
import random
import string
import warnings
import sqlite3
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer

# Suppress warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK packages
nltk.download('popular', quiet=True)
nltk.download('punkt_tab')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmer = nltk.stem.WordNetLemmatizer()

# Functions for text normalization
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Greeting responses
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "nods", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# SQLite Database Integration
def initialize_database():
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect('faq_data.db')
    cursor = conn.cursor()

    # Create the table if not exists
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS faq_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT NOT NULL,
        answer TEXT NOT NULL
    )
    ''')

    # Sample data
    sample_data = [
        ("What is machine learning?", "Machine learning is a field of AI that uses algorithms to learn patterns from data."),
        ("How can I install Python?", "You can install Python from the official website https://www.python.org/downloads."),
        ("What is a neural network?", "A neural network is a computational model inspired by the way biological neural networks in the human brain process information."),
        ("How do I use pandas in Python?", "You can use pandas by importing it with 'import pandas as pd' and leveraging its DataFrame and Series objects."),
        ("What is cosine similarity?", "Cosine similarity measures the cosine of the angle between two non-zero vectors, often used in text analysis."),
        ("How can I preprocess text for NLP?", "Text preprocessing involves steps like tokenization, stopword removal, lemmatization, and vectorization."),
        ("What is TF-IDF?", "TF-IDF stands for Term Frequency-Inverse Document Frequency, a technique to evaluate the importance of a word in a document relative to a collection of documents."),
        ("How to train a machine learning model?", "To train a model, you need labeled data, an algorithm, and tools like Python libraries such as scikit-learn, TensorFlow, or PyTorch."),
        ("What is natural language processing?", "Natural Language Processing (NLP) is a branch of AI that focuses on the interaction between computers and humans using natural language."),
        ("How do I connect Python to MySQL?", "You can connect Python to MySQL using libraries like MySQL Connector or SQLAlchemy.")
    ]

    # Insert sample data if table is empty
    cursor.execute("SELECT COUNT(*) FROM faq_data")
    if cursor.fetchone()[0] == 0:
        cursor.executemany("INSERT INTO faq_data (question, answer) VALUES (?, ?)", sample_data)

    conn.commit()
    conn.close()

# Fetch the answer from the database
def fetch_answer(user_query):
    conn = sqlite3.connect('faq_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT answer FROM faq_data WHERE question = ?", (user_query,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

# Response function
def response(user_response):
    # Check for a database match
    database_answer = fetch_answer(user_response)
    if database_answer:
        return database_answer

    # Fallback to cosine similarity if no exact match is found
    robo_response = ""
    sent_tokens.append(user_response)

    if len(sent_tokens) > 1:  # Ensure there are enough tokens for comparison
        TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
        tfidf = TfidfVec.fit_transform(sent_tokens)
        vals = cosine_similarity(tfidf[-1], tfidf)
        idx = vals.argsort()[0][-2]  # Get the second-highest similarity score
        flat = vals.flatten()
        flat.sort()
        req_tfidf = flat[-2]

        if req_tfidf == 0:
            robo_response = "I am sorry! I don't understand you."
        else:
            robo_response = sent_tokens[idx]
    else:
        robo_response = "I am sorry! I need more context to provide an answer."

    sent_tokens.pop(-1)  # Remove the user query to keep the token list clean
    return robo_response

# Initialize the chatbot
initialize_database()

# Preload Knowledge Base
with open("/content/cs.txt", "r", encoding="utf-8") as file:
    sent_tokens = nltk.sent_tokenize(file.read().lower())

print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")

flag = True

while flag:
    user_response = input().lower()
    if user_response != 'bye':
        if user_response in ['thanks', 'thank you']:
            flag = False
            print("ROBO: You are welcome..")
        elif greeting(user_response) is not None:
            print("ROBO:", greeting(user_response))
        else:
            print("ROBO:", response(user_response))
    else:
        flag = False
        print("ROBO: Bye! Take care..")
