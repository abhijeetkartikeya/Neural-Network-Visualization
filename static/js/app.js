// Main Application Logic
class App {
    constructor() {
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.checkServerHealth();
    }

    setupEventListeners() {
        // Digit Recognition
        document.getElementById('clear-canvas').addEventListener('click', () => {
            drawingCanvas.clear();
            document.getElementById('digit-results').classList.add('hidden');
        });

        document.getElementById('predict-digit').addEventListener('click', () => {
            this.handleDigitPrediction();
        });

        // Image Classification
        document.getElementById('classify-image').addEventListener('click', () => {
            this.handleImageClassification();
        });

        // Sentiment Analysis
        const sentimentText = document.getElementById('sentiment-text');
        const charCount = document.getElementById('char-count');

        sentimentText.addEventListener('input', (e) => {
            charCount.textContent = `${e.target.value.length} / 5000`;
        });

        document.getElementById('analyze-sentiment').addEventListener('click', () => {
            this.handleSentimentAnalysis();
        });

        // Predictive Analytics
        document.getElementById('train-model').addEventListener('click', () => {
            this.handleModelTraining();
        });
    }

    async checkServerHealth() {
        try {
            const health = await api.healthCheck();
            console.log('Server health:', health);

            if (health.status === 'healthy') {
                showToast('Connected to AI server successfully!', 'success');
            }
        } catch (error) {
            console.error('Server health check failed:', error);
            showToast('Warning: Could not connect to AI server. Make sure Flask is running.', 'error');
        }
    }

    async handleDigitPrediction() {
        if (drawingCanvas.isEmpty()) {
            showToast('Please draw a digit first!', 'error');
            return;
        }

        try {
            showLoading(true);

            const imageData = drawingCanvas.getImageData();
            const result = await api.predictDigit(imageData);

            ResultsDashboard.displayDigitResults(result);

            // Update network visualization
            if (result.activations && result.activations.length > 0) {
                const layerSizes = [784, 128, 64, 10]; // MNIST architecture
                networkVisualizer.setArchitecture(layerSizes);
            }

            showToast(`Predicted digit: ${result.predicted_digit}`, 'success');
        } catch (error) {
            console.error('Digit prediction error:', error);
            showToast('Error predicting digit: ' + error.message, 'error');
        } finally {
            showLoading(false);
        }
    }

    async handleImageClassification() {
        if (!window.currentImageFile) {
            showToast('Please upload an image first!', 'error');
            return;
        }

        try {
            showLoading(true);

            const result = await api.classifyImage(window.currentImageFile);

            ResultsDashboard.displayImageResults(result);

            showToast(`Top prediction: ${result.top_prediction}`, 'success');
        } catch (error) {
            console.error('Image classification error:', error);
            showToast('Error classifying image: ' + error.message, 'error');
        } finally {
            showLoading(false);
        }
    }

    async handleSentimentAnalysis() {
        const text = document.getElementById('sentiment-text').value.trim();

        if (!text) {
            showToast('Please enter some text to analyze!', 'error');
            return;
        }

        try {
            showLoading(true);

            const result = await api.analyzeSentiment(text);

            ResultsDashboard.displaySentimentResults(result);

            showToast(`Sentiment: ${result.sentiment}`, 'success');
        } catch (error) {
            console.error('Sentiment analysis error:', error);
            showToast('Error analyzing sentiment: ' + error.message, 'error');
        } finally {
            showLoading(false);
        }
    }

    async handleModelTraining() {
        if (!window.currentDataFile) {
            showToast('Please upload a CSV file first!', 'error');
            return;
        }

        const targetColumn = document.getElementById('target-column').value.trim();
        const taskType = document.getElementById('task-type').value;

        if (!targetColumn) {
            showToast('Please specify the target column!', 'error');
            return;
        }

        try {
            showLoading(true);

            const result = await api.trainModel(window.currentDataFile, targetColumn, taskType);

            if (result.success) {
                ResultsDashboard.displayPredictiveResults(result);
                showToast('Model trained successfully!', 'success');
            }
        } catch (error) {
            console.error('Model training error:', error);
            showToast('Error training model: ' + error.message, 'error');
        } finally {
            showLoading(false);
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const app = new App();
    console.log('AI Neural Network Hub initialized!');
});
