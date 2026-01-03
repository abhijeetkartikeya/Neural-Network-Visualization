import numpy as np
import pandas as pd
from PIL import Image
import io
import base64
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class ImageProcessor:
    """Utilities for image preprocessing"""
    
    @staticmethod
    def preprocess_digit_image(image_data, target_size=(28, 28)):
        """
        Preprocess image for digit recognition
        Args:
            image_data: Base64 encoded image or PIL Image
            target_size: Target dimensions (height, width)
        Returns:
            Preprocessed numpy array
        """
        if isinstance(image_data, str):
            # Decode base64
            if 'base64,' in image_data:
                image_data = image_data.split('base64,')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        else:
            image = image_data
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Resize
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array / 255.0
        
        # Reshape for model input (batch_size, height, width, channels)
        img_array = img_array.reshape(1, target_size[0], target_size[1], 1)
        
        return img_array
    
    @staticmethod
    def preprocess_classification_image(image_data, target_size=(224, 224)):
        """
        Preprocess image for classification
        Args:
            image_data: File object or PIL Image
            target_size: Target dimensions (height, width)
        Returns:
            Preprocessed numpy array
        """
        if hasattr(image_data, 'read'):
            image = Image.open(image_data)
        else:
            image = image_data
        
        # Convert to RGB
        image = image.convert('RGB')
        
        # Resize
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32)
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        # Reshape for model input
        img_array = img_array.reshape(1, target_size[0], target_size[1], 3)
        
        return img_array
    
    @staticmethod
    def augment_image(image_array):
        """Apply data augmentation to image"""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            image_array = np.fliplr(image_array)
        
        # Random rotation (-15 to 15 degrees)
        # Note: This is a simplified version, use tf.image for more robust augmentation
        
        return image_array


class TextProcessor:
    """Utilities for text preprocessing"""
    
    @staticmethod
    def clean_text(text):
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters (keep basic punctuation)
        import re
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    @staticmethod
    def tokenize_and_pad(texts, tokenizer=None, max_length=200, vocab_size=10000):
        """
        Tokenize and pad text sequences
        Args:
            texts: List of text strings
            tokenizer: Existing tokenizer (if None, creates new one)
            max_length: Maximum sequence length
            vocab_size: Vocabulary size
        Returns:
            Padded sequences and tokenizer
        """
        if tokenizer is None:
            tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
            tokenizer.fit_on_texts(texts)
        
        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
        
        return padded, tokenizer


class DataProcessor:
    """Utilities for general data processing"""
    
    @staticmethod
    def load_csv(file_path):
        """Load CSV file"""
        return pd.read_csv(file_path)
    
    @staticmethod
    def prepare_features(df, target_column, categorical_columns=None):
        """
        Prepare features for training
        Args:
            df: Pandas DataFrame
            target_column: Name of target column
            categorical_columns: List of categorical column names
        Returns:
            X, y, scaler, label_encoder
        """
        # Separate features and target
        y = df[target_column].values
        X = df.drop(columns=[target_column])
        
        # Handle categorical columns
        if categorical_columns:
            X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
        
        # Convert to numpy
        X = X.values.astype(np.float32)
        
        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Encode labels if classification
        label_encoder = None
        if y.dtype == 'object' or len(np.unique(y)) < 20:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
        
        return X, y, scaler, label_encoder
    
    @staticmethod
    def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split data into train, validation, and test sets
        Args:
            X: Features
            y: Labels
            test_size: Proportion for test set
            val_size: Proportion for validation set (from training data)
            random_state: Random seed
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Split into train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Split train+val into train and val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    @staticmethod
    def normalize_data(data, mean=None, std=None):
        """Normalize data using mean and standard deviation"""
        if mean is None:
            mean = np.mean(data, axis=0)
        if std is None:
            std = np.std(data, axis=0)
        
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        
        normalized = (data - mean) / std
        return normalized, mean, std
