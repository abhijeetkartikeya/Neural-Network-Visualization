// API Configuration
const API_BASE_URL = 'http://localhost:5001/api';

// API Client
class APIClient {
    constructor(baseURL) {
        this.baseURL = baseURL;
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;

        try {
            const response = await fetch(url, {
                ...options,
                headers: {
                    ...options.headers,
                },
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Request failed');
            }

            return await response.json();
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    }

    async get(endpoint) {
        return this.request(endpoint, { method: 'GET' });
    }

    async post(endpoint, data) {
        return this.request(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });
    }

    async postFormData(endpoint, formData) {
        return this.request(endpoint, {
            method: 'POST',
            body: formData,
        });
    }

    // Digit Recognition
    async predictDigit(imageData) {
        return this.post('/digit/predict', { image: imageData });
    }

    // Image Classification
    async classifyImage(file) {
        const formData = new FormData();
        formData.append('file', file);
        return this.postFormData('/image/classify', formData);
    }

    // Sentiment Analysis
    async analyzeSentiment(text) {
        return this.post('/sentiment/analyze', { text });
    }

    // Predictive Analytics
    async trainModel(file, targetColumn, taskType) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('target_column', targetColumn);
        formData.append('task_type', taskType);
        return this.postFormData('/predict/train', formData);
    }

    // Visualizations
    async visualizeNetwork(layerSizes) {
        return this.post('/visualize/network', { layer_sizes: layerSizes });
    }

    // Health Check
    async healthCheck() {
        return this.get('/health');
    }

    // Model Info
    async getModelsInfo() {
        return this.get('/models/info');
    }
}

// Create global API client instance
const api = new APIClient(API_BASE_URL);
