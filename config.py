import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Flask configuration
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5001
DEBUG = True

# Upload configuration
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MAX_UPLOAD_SIZE = 16 * 1024 * 1024  # 16MB
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
ALLOWED_DATA_EXTENSIONS = {'csv', 'txt'}

# Model paths
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')
DIGIT_MODEL_PATH = os.path.join(MODELS_DIR, 'mnist_cnn.h5')
IMAGE_CLASSIFIER_MODEL = 'MobileNetV2'  # or 'ResNet50'
SENTIMENT_MODEL_PATH = os.path.join(MODELS_DIR, 'sentiment_lstm.h5')
SENTIMENT_TOKENIZER_PATH = os.path.join(MODELS_DIR, 'sentiment_tokenizer.pkl')

# Sample data paths
SAMPLES_DIR = os.path.join(BASE_DIR, 'data', 'samples')

# Model hyperparameters
DIGIT_RECOGNITION_CONFIG = {
    'input_shape': (28, 28, 1),
    'num_classes': 10,
    'confidence_threshold': 0.5
}

IMAGE_CLASSIFIER_CONFIG = {
    'input_shape': (224, 224, 3),
    'top_k': 5,
    'confidence_threshold': 0.1
}

SENTIMENT_CONFIG = {
    'max_sequence_length': 200,
    'vocab_size': 10000,
    'embedding_dim': 128,
    'num_classes': 3  # Positive, Negative, Neutral
}

PREDICTIVE_MODEL_CONFIG = {
    'hidden_layers': [64, 32, 16],
    'activation': 'relu',
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'epochs': 50,
    'batch_size': 32,
    'validation_split': 0.2
}

# Visualization settings
VIZ_DPI = 100
VIZ_FIGSIZE = (10, 6)
VIZ_STYLE = 'dark_background'

# Feature flags
ENABLE_DIGIT_RECOGNITION = True
ENABLE_IMAGE_CLASSIFICATION = True
ENABLE_SENTIMENT_ANALYSIS = True
ENABLE_PREDICTIVE_ANALYTICS = True
ENABLE_MODEL_TRAINING = True
ENABLE_VISUALIZATION = True

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
