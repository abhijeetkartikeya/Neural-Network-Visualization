import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from io import BytesIO
import base64
from sklearn.metrics import confusion_matrix
import config


class Visualizer:
    """Utilities for generating visualizations using Matplotlib"""
    
    def __init__(self):
        plt.style.use(config.VIZ_STYLE)
        self.dpi = config.VIZ_DPI
        self.figsize = config.VIZ_FIGSIZE
    
    def plot_training_history(self, history):
        """
        Plot training history (loss and accuracy)
        Args:
            history: Training history dict with 'loss', 'accuracy', 'val_loss', 'val_accuracy'
        Returns:
            Base64 encoded image
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=self.dpi)
        
        # Plot loss
        ax1.plot(history.get('loss', []), label='Training Loss', color='#00d4ff', linewidth=2)
        if 'val_loss' in history:
            ax1.plot(history['val_loss'], label='Validation Loss', color='#ff00ff', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        if 'accuracy' in history:
            ax2.plot(history['accuracy'], label='Training Accuracy', color='#00d4ff', linewidth=2)
        if 'val_accuracy' in history:
            ax2.plot(history['val_accuracy'], label='Validation Accuracy', color='#ff00ff', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        img_base64 = self._fig_to_base64(fig)
        plt.close(fig)
        
        return img_base64
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None):
        """
        Plot confusion matrix
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
        Returns:
            Base64 encoded image
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 8), dpi=self.dpi)
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'}, ax=ax)
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        img_base64 = self._fig_to_base64(fig)
        plt.close(fig)
        
        return img_base64
    
    def plot_feature_importance(self, feature_names, importance_scores):
        """
        Plot feature importance
        Args:
            feature_names: List of feature names
            importance_scores: Importance scores for each feature
        Returns:
            Base64 encoded image
        """
        # Sort by importance
        indices = np.argsort(importance_scores)[::-1][:20]  # Top 20
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(indices)))
        ax.barh(range(len(indices)), importance_scores[indices], color=colors)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title('Feature Importance (Top 20)', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        img_base64 = self._fig_to_base64(fig)
        plt.close(fig)
        
        return img_base64
    
    def plot_prediction_confidence(self, labels, confidences, top_k=5):
        """
        Plot prediction confidence bars
        Args:
            labels: List of class labels
            confidences: Confidence scores
            top_k: Number of top predictions to show
        Returns:
            Base64 encoded image
        """
        # Get top k predictions
        indices = np.argsort(confidences)[::-1][:top_k]
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        
        colors = ['#00d4ff' if i == 0 else '#8b5cf6' for i in range(len(indices))]
        bars = ax.barh(range(len(indices)), confidences[indices], color=colors)
        
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([labels[i] for i in indices])
        ax.set_xlabel('Confidence', fontsize=12)
        ax.set_title('Prediction Confidence', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add percentage labels
        for i, (bar, conf) in enumerate(zip(bars, confidences[indices])):
            ax.text(conf + 0.02, i, f'{conf*100:.1f}%', 
                   va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        img_base64 = self._fig_to_base64(fig)
        plt.close(fig)
        
        return img_base64
    
    def plot_network_architecture(self, layer_sizes):
        """
        Plot neural network architecture diagram
        Args:
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
        Returns:
            Base64 encoded image
        """
        fig, ax = plt.subplots(figsize=(12, 8), dpi=self.dpi)
        
        max_neurons = max(layer_sizes)
        layer_spacing = 1.0 / (len(layer_sizes) - 1) if len(layer_sizes) > 1 else 1
        
        # Draw neurons
        for i, size in enumerate(layer_sizes):
            x = i * layer_spacing
            neuron_spacing = 1.0 / (size + 1)
            
            for j in range(size):
                y = (j + 1) * neuron_spacing
                
                # Color based on layer type
                if i == 0:
                    color = '#00d4ff'  # Input layer
                elif i == len(layer_sizes) - 1:
                    color = '#ff00ff'  # Output layer
                else:
                    color = '#8b5cf6'  # Hidden layers
                
                circle = plt.Circle((x, y), 0.02, color=color, zorder=2)
                ax.add_patch(circle)
                
                # Draw connections to next layer
                if i < len(layer_sizes) - 1:
                    next_size = layer_sizes[i + 1]
                    next_spacing = 1.0 / (next_size + 1)
                    for k in range(next_size):
                        next_y = (k + 1) * next_spacing
                        ax.plot([x, (i + 1) * layer_spacing], [y, next_y], 
                               'w-', alpha=0.2, linewidth=0.5, zorder=1)
        
        # Add layer labels
        for i, size in enumerate(layer_sizes):
            x = i * layer_spacing
            if i == 0:
                label = f'Input\n({size})'
            elif i == len(layer_sizes) - 1:
                label = f'Output\n({size})'
            else:
                label = f'Hidden {i}\n({size})'
            ax.text(x, -0.1, label, ha='center', fontsize=10, fontweight='bold')
        
        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-0.2, 1.1])
        ax.axis('off')
        ax.set_title('Neural Network Architecture', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        img_base64 = self._fig_to_base64(fig)
        plt.close(fig)
        
        return img_base64
    
    def _fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 string"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', facecolor='#1a1a2e')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        return f'data:image/png;base64,{img_base64}'
