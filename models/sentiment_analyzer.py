import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import pickle
import os
import re
import config
from utils.data_processor import TextProcessor


class SentimentAnalyzer:
    """LSTM model for sentiment analysis"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.max_length = config.SENTIMENT_CONFIG['max_sequence_length']
        self.vocab_size = config.SENTIMENT_CONFIG['vocab_size']
        self.embedding_dim = config.SENTIMENT_CONFIG['embedding_dim']
        self.num_classes = config.SENTIMENT_CONFIG['num_classes']
        self.class_names = ['Negative', 'Neutral', 'Positive']
        self.text_processor = TextProcessor()
    
    def build_model(self):
        """Build LSTM architecture for sentiment analysis"""
        model = models.Sequential([
            layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Dropout(0.5),
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, texts, labels, validation_split=0.2, epochs=10, batch_size=32):
        """
        Train the sentiment model
        Args:
            texts: List of text strings
            labels: List of sentiment labels (0=Negative, 1=Neutral, 2=Positive)
            validation_split: Validation data proportion
            epochs: Number of training epochs
            batch_size: Batch size
        Returns:
            Training history
        """
        # Preprocess texts
        cleaned_texts = [self.text_processor.clean_text(text) for text in texts]
        
        # Tokenize and pad
        sequences, self.tokenizer = self.text_processor.tokenize_and_pad(
            cleaned_texts, 
            max_length=self.max_length,
            vocab_size=self.vocab_size
        )
        
        if self.model is None:
            self.build_model()
        
        # Train
        history = self.model.fit(
            sequences, np.array(labels),
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            verbose=1
        )
        
        return history.history
    
    def predict(self, text):
        """
        Predict sentiment for a text
        Args:
            text: Input text string
        Returns:
            Dictionary with sentiment prediction and confidence
        """
        if self.model is None or self.tokenizer is None:
            # Use simple rule-based sentiment for demo
            return self._rule_based_sentiment(text)
        
        # Preprocess
        cleaned_text = self.text_processor.clean_text(text)
        sequences, _ = self.text_processor.tokenize_and_pad(
            [cleaned_text],
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        # Predict
        predictions = self.model.predict(sequences, verbose=0)[0]
        
        # Get sentiment
        sentiment_idx = int(np.argmax(predictions))
        sentiment = self.class_names[sentiment_idx]
        confidence = float(predictions[sentiment_idx])
        
        # Get all confidences
        all_confidences = {
            self.class_names[i]: float(predictions[i]) 
            for i in range(self.num_classes)
        }
        
        # Analyze emotions (simple keyword-based)
        emotions = self._detect_emotions(text)
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'all_confidences': all_confidences,
            'emotions': emotions,
            'text_length': len(text),
            'word_count': len(text.split())
        }
    
    def _rule_based_sentiment(self, text):
        """Simple rule-based sentiment analysis for demo purposes"""
        text_lower = text.lower()
        
        # Positive and negative word lists
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                         'love', 'best', 'awesome', 'perfect', 'happy', 'beautiful', 'brilliant']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'poor',
                         'disappointing', 'sad', 'angry', 'useless', 'waste', 'annoying']
        
        # Count positive and negative words
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        # Determine sentiment
        if pos_count > neg_count:
            sentiment = 'Positive'
            confidence = min(0.6 + (pos_count * 0.1), 0.95)
        elif neg_count > pos_count:
            sentiment = 'Negative'
            confidence = min(0.6 + (neg_count * 0.1), 0.95)
        else:
            sentiment = 'Neutral'
            confidence = 0.5
        
        all_confidences = {
            'Positive': confidence if sentiment == 'Positive' else (1 - confidence) / 2,
            'Neutral': confidence if sentiment == 'Neutral' else 0.3,
            'Negative': confidence if sentiment == 'Negative' else (1 - confidence) / 2
        }
        
        emotions = self._detect_emotions(text)
        
        return {
            'sentiment': sentiment,
            'confidence': float(confidence),
            'all_confidences': all_confidences,
            'emotions': emotions,
            'text_length': len(text),
            'word_count': len(text.split()),
            'note': 'Using rule-based analysis (demo mode)'
        }
    
    def _detect_emotions(self, text):
        """Detect emotions using keyword matching"""
        text_lower = text.lower()
        
        emotion_keywords = {
            'joy': ['happy', 'joy', 'excited', 'wonderful', 'amazing', 'love'],
            'sadness': ['sad', 'depressed', 'unhappy', 'miserable', 'disappointed'],
            'anger': ['angry', 'furious', 'mad', 'annoyed', 'frustrated'],
            'fear': ['scared', 'afraid', 'worried', 'anxious', 'nervous'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished']
        }
        
        detected_emotions = {}
        for emotion, keywords in emotion_keywords.items():
            count = sum(1 for word in keywords if word in text_lower)
            if count > 0:
                detected_emotions[emotion] = min(count * 0.3, 1.0)
        
        return detected_emotions
    
    def save_model(self, model_path=None, tokenizer_path=None):
        """Save model and tokenizer"""
        if model_path is None:
            model_path = config.SENTIMENT_MODEL_PATH
        if tokenizer_path is None:
            tokenizer_path = config.SENTIMENT_TOKENIZER_PATH
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        if self.model:
            self.model.save(model_path)
        if self.tokenizer:
            with open(tokenizer_path, 'wb') as f:
                pickle.dump(self.tokenizer, f)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path=None, tokenizer_path=None):
        """Load model and tokenizer"""
        if model_path is None:
            model_path = config.SENTIMENT_MODEL_PATH
        if tokenizer_path is None:
            tokenizer_path = config.SENTIMENT_TOKENIZER_PATH
        
        try:
            if os.path.exists(model_path):
                self.model = keras.models.load_model(model_path)
                print(f"Model loaded from {model_path}")
            
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)
                print(f"Tokenizer loaded from {tokenizer_path}")
            
            return True
        except Exception as e:
            print(f"Could not load model: {e}")
            return False


def initialize_sentiment_analyzer():
    """Initialize sentiment analyzer"""
    analyzer = SentimentAnalyzer()
    analyzer.load_model()  # Try to load if exists
    return analyzer
