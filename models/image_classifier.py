import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from PIL import Image
import config


class ImageClassifier:
    """Image classification using pre-trained MobileNetV2"""
    
    def __init__(self):
        self.model = None
        self.input_shape = config.IMAGE_CLASSIFIER_CONFIG['input_shape']
        self.top_k = config.IMAGE_CLASSIFIER_CONFIG['top_k']
        self.load_model()
    
    def load_model(self):
        """Load pre-trained MobileNetV2 model"""
        print("Loading MobileNetV2 model...")
        self.model = MobileNetV2(
            weights='imagenet',
            include_top=True,
            input_shape=self.input_shape
        )
        print("MobileNetV2 model loaded successfully!")
    
    def predict(self, image_array):
        """
        Classify an image
        Args:
            image_array: Preprocessed image array (1, 224, 224, 3)
        Returns:
            Dictionary with top predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Preprocess for MobileNetV2
        processed_image = preprocess_input(image_array * 255.0)
        
        # Get predictions
        predictions = self.model.predict(processed_image, verbose=0)
        
        # Decode predictions
        decoded = decode_predictions(predictions, top=self.top_k)[0]
        
        # Format results
        results = []
        for i, (imagenet_id, label, score) in enumerate(decoded):
            results.append({
                'rank': i + 1,
                'label': label.replace('_', ' ').title(),
                'confidence': float(score),
                'imagenet_id': imagenet_id
            })
        
        # Get layer activations for visualization
        activations = self._get_activations(processed_image)
        
        return {
            'predictions': results,
            'top_prediction': results[0]['label'],
            'top_confidence': results[0]['confidence'],
            'activations': activations
        }
    
    def _get_activations(self, image_array):
        """Get activations from intermediate layers"""
        # Get some intermediate layer outputs
        layer_names = ['block_1_expand_relu', 'block_6_expand_relu', 'block_13_expand_relu']
        layer_outputs = []
        
        for layer_name in layer_names:
            try:
                layer = self.model.get_layer(layer_name)
                layer_outputs.append(layer.output)
            except:
                continue
        
        if not layer_outputs:
            return []
        
        activation_model = keras.Model(inputs=self.model.input, outputs=layer_outputs)
        activations = activation_model.predict(image_array, verbose=0)
        
        # Convert to summary info
        activation_info = []
        for i, (activation, layer_name) in enumerate(zip(activations, layer_names)):
            activation_info.append({
                'layer_index': i,
                'layer_name': layer_name,
                'shape': list(activation.shape),
                'mean_activation': float(np.mean(activation)),
                'max_activation': float(np.max(activation))
            })
        
        return activation_info
    
    def get_model_info(self):
        """Get model information"""
        return {
            'model_name': 'MobileNetV2',
            'input_shape': self.input_shape,
            'total_params': self.model.count_params() if self.model else 0,
            'num_classes': 1000,
            'pretrained_on': 'ImageNet'
        }


def initialize_image_classifier():
    """Initialize image classifier"""
    return ImageClassifier()
