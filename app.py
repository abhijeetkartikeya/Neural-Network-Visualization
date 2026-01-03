from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
import io
import base64

import config
from models.digit_recognition import initialize_digit_model
from models.image_classifier import initialize_image_classifier
from models.sentiment_analyzer import initialize_sentiment_analyzer
from models.predictive_model import initialize_predictive_model
from utils.data_processor import ImageProcessor, TextProcessor, DataProcessor
from utils.visualization import Visualizer

# Initialize Flask app
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Configure upload settings
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = config.MAX_UPLOAD_SIZE

# Initialize models (will load or train on startup)
print("Initializing models...")
digit_model = None
image_classifier = None
sentiment_analyzer = None
visualizer = Visualizer()

# Initialize processors
image_processor = ImageProcessor()
text_processor = TextProcessor()
data_processor = DataProcessor()


def allowed_file(filename, allowed_extensions):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('static', 'index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models': {
            'digit_recognition': digit_model is not None,
            'image_classifier': image_classifier is not None,
            'sentiment_analyzer': sentiment_analyzer is not None
        }
    })


# ===== DIGIT RECOGNITION ENDPOINTS =====

@app.route('/api/digit/predict', methods=['POST'])
def predict_digit():
    """Predict handwritten digit from canvas image"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Preprocess image
        img_array = image_processor.preprocess_digit_image(image_data)
        
        # Make prediction
        result = digit_model.predict(img_array)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ===== IMAGE CLASSIFICATION ENDPOINTS =====

@app.route('/api/image/classify', methods=['POST'])
def classify_image():
    """Classify uploaded image"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename, config.ALLOWED_IMAGE_EXTENSIONS):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Preprocess image
        img_array = image_processor.preprocess_classification_image(file)
        
        # Make prediction
        result = image_classifier.predict(img_array)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ===== SENTIMENT ANALYSIS ENDPOINTS =====

@app.route('/api/sentiment/analyze', methods=['POST'])
def analyze_sentiment():
    """Analyze sentiment of text"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text or len(text.strip()) == 0:
            return jsonify({'error': 'No text provided'}), 400
        
        if len(text) > 5000:
            return jsonify({'error': 'Text too long (max 5000 characters)'}), 400
        
        # Analyze sentiment
        result = sentiment_analyzer.predict(text)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ===== PREDICTIVE ANALYTICS ENDPOINTS =====

@app.route('/api/predict/train', methods=['POST'])
def train_predictive_model():
    """Train custom predictive model on uploaded data"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        task_type = request.form.get('task_type', 'regression')
        target_column = request.form.get('target_column')
        
        if not target_column:
            return jsonify({'error': 'Target column not specified'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Load and process data
        df = data_processor.load_csv(filepath)
        
        if target_column not in df.columns:
            return jsonify({'error': f'Target column "{target_column}" not found'}), 400
        
        # Prepare features
        X, y, scaler, label_encoder = data_processor.prepare_features(df, target_column)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = data_processor.split_data(X, y)
        
        # Initialize and train model
        model = initialize_predictive_model(task_type=task_type)
        history = model.train(X_train, y_train, X_val, y_val, epochs=30)
        
        # Evaluate
        eval_results = model.evaluate(X_test, y_test)
        
        # Get feature importance
        feature_importance = model.get_feature_importance(X_train[:100])
        feature_names = [col for col in df.columns if col != target_column]
        
        # Generate visualizations
        training_plot = visualizer.plot_training_history(history)
        importance_plot = visualizer.plot_feature_importance(feature_names, feature_importance)
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'evaluation': eval_results,
            'training_plot': training_plot,
            'importance_plot': importance_plot,
            'feature_importance': {
                name: float(score) 
                for name, score in zip(feature_names, feature_importance)
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/inference', methods=['POST'])
def predict_custom():
    """Make prediction using trained model"""
    try:
        data = request.get_json()
        features = data.get('features', [])
        
        if not features:
            return jsonify({'error': 'No features provided'}), 400
        
        # This is a simplified version - in production, you'd load the saved model
        return jsonify({
            'error': 'Please train a model first using the /api/predict/train endpoint'
        }), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ===== VISUALIZATION ENDPOINTS =====

@app.route('/api/visualize/network', methods=['POST'])
def visualize_network():
    """Generate neural network architecture visualization"""
    try:
        data = request.get_json()
        layer_sizes = data.get('layer_sizes', [784, 128, 64, 10])
        
        # Generate visualization
        img_base64 = visualizer.plot_network_architecture(layer_sizes)
        
        return jsonify({'image': img_base64})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/visualize/confusion-matrix', methods=['POST'])
def visualize_confusion_matrix():
    """Generate confusion matrix visualization"""
    try:
        data = request.get_json()
        y_true = data.get('y_true', [])
        y_pred = data.get('y_pred', [])
        class_names = data.get('class_names', None)
        
        if not y_true or not y_pred:
            return jsonify({'error': 'Missing data'}), 400
        
        # Generate visualization
        img_base64 = visualizer.plot_confusion_matrix(y_true, y_pred, class_names)
        
        return jsonify({'image': img_base64})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ===== MODEL INFO ENDPOINTS =====

@app.route('/api/models/info', methods=['GET'])
def get_models_info():
    """Get information about all loaded models"""
    try:
        info = {
            'digit_recognition': digit_model.get_model_summary() if digit_model else None,
            'image_classifier': image_classifier.get_model_info() if image_classifier else None,
            'sentiment_analyzer': {
                'model_name': 'LSTM Sentiment Analyzer',
                'num_classes': 3,
                'classes': ['Negative', 'Neutral', 'Positive']
            } if sentiment_analyzer else None
        }
        
        return jsonify(info)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def initialize_models():
    """Initialize all models on startup"""
    global digit_model, image_classifier, sentiment_analyzer
    
    try:
        if config.ENABLE_DIGIT_RECOGNITION:
            print("\n=== Initializing Digit Recognition Model ===")
            digit_model = initialize_digit_model()
        
        if config.ENABLE_IMAGE_CLASSIFICATION:
            print("\n=== Initializing Image Classifier ===")
            image_classifier = initialize_image_classifier()
        
        if config.ENABLE_SENTIMENT_ANALYSIS:
            print("\n=== Initializing Sentiment Analyzer ===")
            sentiment_analyzer = initialize_sentiment_analyzer()
        
        print("\n✅ All models initialized successfully!\n")
    
    except Exception as e:
        print(f"\n❌ Error initializing models: {e}\n")


if __name__ == '__main__':
    # Initialize models before starting server
    initialize_models()
    
    # Run Flask app
    print(f"\n🚀 Starting Flask server on http://{config.FLASK_HOST}:{config.FLASK_PORT}\n")
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.DEBUG
    )
