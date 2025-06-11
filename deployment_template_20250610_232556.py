
# DEPLOYMENT CODE TEMPLATE
# Generated on 20250610_232556

import pickle
import pandas as pd
import spacy
import re

# Load the model and vectorizer
with open('best_model_20250610_232556.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer_20250610_232556.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def clean_text_spacy(text):
    """Clean text using spaCy for better preprocessing"""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower().strip()

    doc = nlp(text)
    cleaned_tokens = []
    for token in doc:
        if (not token.is_stop and not token.is_punct and 
            not token.is_space and token.is_alpha and len(token.lemma_) > 2):
            cleaned_tokens.append(token.lemma_)

    return ' '.join(cleaned_tokens)

def predict_sentiment(text):
    """Predict sentiment for new text"""
    cleaned = clean_text_spacy(text)
    text_vector = vectorizer.transform([cleaned])
    prediction = model.predict(text_vector)[0]
    confidence = model.predict_proba(text_vector)[0].max()

    return {
        'sentiment': 'Positive' if prediction == 1 else 'Negative',
        'confidence': confidence,
        'prediction_score': int(prediction)
    }

# Example usage:
# result = predict_sentiment("This product is amazing!")
# print(result)
