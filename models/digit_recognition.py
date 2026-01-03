import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import pickle
import os
import config
from utils.visualization import Visualizer


class DigitRecognitionModel:
    """CNN model for handwritten digit recognition (MNIST)"""
    
    def __init__(self):
        self.model = None
        self.input_shape = config.DIGIT_RECOGNITION_CONFIG['input_shape']
        self.num_classes = config.DIGIT_RECOGNITION_CONFIG['num_classes']
        self.visualizer = Visualizer()
        
    def build_model(self):
        """Build CNN architecture"""
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train=None, y_train=None, X_val=None, y_val=None, epochs=10, batch_size=128):
        """
        Train the model
        Args:
            X_train: Training images (if None, loads MNIST)
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
        Returns:
            Training history
        """
        if X_train is None:
            # Load MNIST dataset
            (X_train, y_train), (X_val, y_val) = mnist.load_data()
            
            # Preprocess
            X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
            X_val = X_val.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        
        if self.model is None:
            self.build_model()
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        return history.history
    
    def predict(self, image_array):
        """
        Make prediction on a single image
        Args:
            image_array: Preprocessed image array (1, 28, 28, 1)
        Returns:
            Dictionary with predictions and confidence scores
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load or train the model first.")
        
        # Get predictions
        predictions = self.model.predict(image_array, verbose=0)[0]
        
        # Get top prediction
        predicted_digit = int(np.argmax(predictions))
        confidence = float(predictions[predicted_digit])
        
        # Get all confidences
        all_confidences = {str(i): float(predictions[i]) for i in range(10)}
        
        # Get activations for visualization
        activations = self._get_activations(image_array)
        
        return {
            'predicted_digit': predicted_digit,
            'confidence': confidence,
            'all_confidences': all_confidences,
            'activations': activations
        }
    
    def _get_activations(self, image_array):
        """Get layer activations for visualization"""
        if self.model is None:
            return []
        
        # Get outputs from each layer
        layer_outputs = [layer.output for layer in self.model.layers[:6]]  # First 6 layers
        activation_model = models.Model(inputs=self.model.input, outputs=layer_outputs)
        activations = activation_model.predict(image_array, verbose=0)
        
        # Convert to list of layer info
        activation_info = []
        for i, activation in enumerate(activations):
            activation_info.append({
                'layer_index': i,
                'layer_name': self.model.layers[i].name,
                'shape': activation.shape,
                'mean_activation': float(np.mean(activation)),
                'max_activation': float(np.max(activation))
            })
        
        return activation_info
    
    def save_model(self, path=None):
        """Save model to disk"""
        if path is None:
            path = config.DIGIT_MODEL_PATH
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path=None):
        """Load model from disk"""
        if path is None:
            path = config.DIGIT_MODEL_PATH
        
        if os.path.exists(path):
            self.model = keras.models.load_model(path)
            print(f"Model loaded from {path}")
            return True
        else:
            print(f"Model not found at {path}. Training new model...")
            return False
    
    def get_model_summary(self):
        """Get model architecture summary"""
        if self.model is None:
            return "Model not built yet"
        
        # Get layer sizes for visualization
        layer_sizes = []
        for layer in self.model.layers:
            if hasattr(layer, 'units'):
                layer_sizes.append(layer.units)
            elif isinstance(layer, layers.Flatten):
                layer_sizes.append(np.prod(layer.input_shape[1:]))
        
        return {
            'total_params': self.model.count_params(),
            'layer_count': len(self.model.layers),
            'layer_sizes': layer_sizes
        }
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test data"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Get predictions for confusion matrix
        y_pred = np.argmax(self.model.predict(X_test, verbose=0), axis=1)
        
        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'predictions': y_pred.tolist()
        }


def initialize_digit_model():
    """Initialize and load/train digit recognition model"""
    model = DigitRecognitionModel()
    
    # Try to load existing model
    if not model.load_model():
        # Train new model
        print("Training new MNIST model...")
        model.build_model()
        history = model.train(epochs=5)  # Quick training
        model.save_model()
        print("Model training complete!")
    
    return model
