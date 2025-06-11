#!/usr/bin/env python3
"""
DEPLOYMENT CODE TEMPLATE
Generated on 20250610_233548
Sentiment Analysis Model Deployment Script
"""

import pickle
import pandas as pd
import spacy
import re
import sys
from typing import Dict, Any

class SentimentAnalyzer:
    def __init__(self, model_path: str, vectorizer_path: str):
        """Initialize the sentiment analyzer with saved model and vectorizer"""
        self.model = self._load_model(model_path)
        self.vectorizer = self._load_vectorizer(vectorizer_path)
        self.nlp = spacy.load('en_core_web_sm')

    def _load_model(self, model_path: str):
        """Load the trained model"""
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    def _load_vectorizer(self, vectorizer_path: str):
        """Load the TF-IDF vectorizer"""
        with open(vectorizer_path, 'rb') as f:
            return pickle.load(f)

    def clean_text_spacy(self, text: str) -> str:
        """Clean text using spaCy for better preprocessing"""
        # Basic cleaning
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower().strip()

        # Process with spaCy
        doc = self.nlp(text)
        cleaned_tokens = []
        for token in doc:
            if (not token.is_stop and not token.is_punct and 
                not token.is_space and token.is_alpha and len(token.lemma_) > 2):
                cleaned_tokens.append(token.lemma_)

        return ' '.join(cleaned_tokens)

    def predict_sentiment(self, text: str) -> Dict[str, Any]:
        """Predict sentiment for new text"""
        try:
            cleaned = self.clean_text_spacy(text)

            if not cleaned or len(cleaned.strip()) == 0:
                return {
                    'sentiment': 'Neutral',
                    'confidence': 0.5,
                    'prediction_score': -1,
                    'original_text': text,
                    'cleaned_text': cleaned,
                    'error': 'No meaningful text after cleaning'
                }

            text_vector = self.vectorizer.transform([cleaned])
            prediction = self.model.predict(text_vector)[0]
            confidence = self.model.predict_proba(text_vector)[0].max()

            return {
                'sentiment': 'Positive' if prediction == 1 else 'Negative',
                'confidence': float(confidence),
                'prediction_score': int(prediction),
                'original_text': text,
                'cleaned_text': cleaned,
                'error': None
            }

        except Exception as e:
            return {
                'sentiment': 'Error',
                'confidence': 0.0,
                'prediction_score': -1,
                'original_text': text,
                'cleaned_text': '',
                'error': str(e)
            }

    def predict_batch(self, texts: list) -> list:
        """Predict sentiment for multiple texts"""
        return [self.predict_sentiment(text) for text in texts]

# Example usage
if __name__ == "__main__":
    # Initialize the analyzer
    analyzer = SentimentAnalyzer(
        model_path='best_model_20250610_233548.pkl',
        vectorizer_path='tfidf_vectorizer_20250610_233548.pkl'
    )

    # Example predictions
    test_texts = [
        "This product is amazing! I love it!",
        "Terrible experience. Would not recommend.",
        "It's okay, nothing special."
    ]

    print("Sentiment Analysis Results:")
    print("-" * 50)

    for text in test_texts:
        result = analyzer.predict_sentiment(text)
        print(f"Text: {text}")
        print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.3f})")
        print("-" * 50)
