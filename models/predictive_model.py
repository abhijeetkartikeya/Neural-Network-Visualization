import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import config


class PredictiveModel:
    """Customizable neural network for regression/classification tasks"""
    
    def __init__(self, task_type='regression'):
        """
        Initialize predictive model
        Args:
            task_type: 'regression' or 'classification'
        """
        self.model = None
        self.task_type = task_type
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.history = None
    
    def build_model(self, input_dim, output_dim=1, hidden_layers=None):
        """
        Build neural network architecture
        Args:
            input_dim: Number of input features
            output_dim: Number of output classes (for classification) or 1 (for regression)
            hidden_layers: List of hidden layer sizes
        """
        if hidden_layers is None:
            hidden_layers = config.PREDICTIVE_MODEL_CONFIG['hidden_layers']
        
        activation = config.PREDICTIVE_MODEL_CONFIG['activation']
        dropout_rate = config.PREDICTIVE_MODEL_CONFIG['dropout_rate']
        
        # Build model
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Dense(hidden_layers[0], activation=activation, input_dim=input_dim))
        model.add(layers.Dropout(dropout_rate))
        
        # Hidden layers
        for units in hidden_layers[1:]:
            model.add(layers.Dense(units, activation=activation))
            model.add(layers.Dropout(dropout_rate))
        
        # Output layer
        if self.task_type == 'classification':
            if output_dim == 2:
                model.add(layers.Dense(1, activation='sigmoid'))
                loss = 'binary_crossentropy'
            else:
                model.add(layers.Dense(output_dim, activation='softmax'))
                loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        else:
            model.add(layers.Dense(1, activation='linear'))
            loss = 'mse'
            metrics = ['mae']
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.PREDICTIVE_MODEL_CONFIG['learning_rate']),
            loss=loss,
            metrics=metrics
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=None, batch_size=None):
        """
        Train the model
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
        Returns:
            Training history
        """
        if epochs is None:
            epochs = config.PREDICTIVE_MODEL_CONFIG['epochs']
        if batch_size is None:
            batch_size = config.PREDICTIVE_MODEL_CONFIG['batch_size']
        
        # Build model if not exists
        if self.model is None:
            input_dim = X_train.shape[1]
            output_dim = len(np.unique(y_train)) if self.task_type == 'classification' else 1
            self.build_model(input_dim, output_dim)
        
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=[early_stopping],
            verbose=1
        )
        
        self.history = history.history
        return self.history
    
    def predict(self, X):
        """
        Make predictions
        Args:
            X: Input features
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        predictions = self.model.predict(X, verbose=0)
        
        if self.task_type == 'classification':
            if predictions.shape[1] == 1:
                # Binary classification
                pred_classes = (predictions > 0.5).astype(int).flatten()
                confidences = np.maximum(predictions, 1 - predictions).flatten()
            else:
                # Multi-class classification
                pred_classes = np.argmax(predictions, axis=1)
                confidences = np.max(predictions, axis=1)
            
            return {
                'predictions': pred_classes.tolist(),
                'confidences': confidences.tolist(),
                'probabilities': predictions.tolist()
            }
        else:
            # Regression
            return {
                'predictions': predictions.flatten().tolist()
            }
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test data"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        if self.task_type == 'classification':
            return {
                'loss': float(results[0]),
                'accuracy': float(results[1])
            }
        else:
            return {
                'loss': float(results[0]),
                'mae': float(results[1])
            }
    
    def get_feature_importance(self, X_sample):
        """
        Calculate feature importance using gradient-based method
        Args:
            X_sample: Sample of input features
        Returns:
            Feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Convert to tensor
        X_tensor = tf.convert_to_tensor(X_sample, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            predictions = self.model(X_tensor)
            if self.task_type == 'classification':
                # Use max probability
                output = tf.reduce_max(predictions, axis=1)
            else:
                output = predictions
        
        # Get gradients
        gradients = tape.gradient(output, X_tensor)
        
        # Calculate importance as mean absolute gradient
        importance = np.mean(np.abs(gradients.numpy()), axis=0)
        
        # Normalize
        importance = importance / np.sum(importance)
        
        return importance
    
    def get_model_summary(self):
        """Get model architecture summary"""
        if self.model is None:
            return "Model not built yet"
        
        layer_sizes = []
        for layer in self.model.layers:
            if hasattr(layer, 'units'):
                layer_sizes.append(layer.units)
        
        return {
            'task_type': self.task_type,
            'total_params': self.model.count_params(),
            'layer_count': len(self.model.layers),
            'layer_sizes': layer_sizes
        }


def initialize_predictive_model(task_type='regression'):
    """Initialize predictive model"""
    return PredictiveModel(task_type=task_type)
