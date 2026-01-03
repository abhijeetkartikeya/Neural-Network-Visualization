// Results Dashboard
class ResultsDashboard {
    // Display digit recognition results
    static displayDigitResults(data) {
        const resultsContainer = document.getElementById('digit-results');
        const predictedDigit = document.getElementById('predicted-digit');
        const digitConfidence = document.getElementById('digit-confidence');
        const confidenceBars = document.getElementById('digit-confidence-bars');

        // Show results
        resultsContainer.classList.remove('hidden');

        // Display predicted digit
        predictedDigit.textContent = data.predicted_digit;
        digitConfidence.textContent = `${(data.confidence * 100).toFixed(1)}%`;

        // Create confidence bars for all digits
        confidenceBars.innerHTML = '';
        for (let i = 0; i < 10; i++) {
            const confidence = data.all_confidences[i.toString()];
            const barHTML = `
                <div class="confidence-bar">
                    <span class="confidence-bar-label">Digit ${i}</span>
                    <div class="confidence-bar-track">
                        <div class="confidence-bar-fill" style="width: ${confidence * 100}%">
                            <span class="confidence-bar-value">${(confidence * 100).toFixed(1)}%</span>
                        </div>
                    </div>
                </div>
            `;
            confidenceBars.innerHTML += barHTML;
        }
    }

    // Display image classification results
    static displayImageResults(data) {
        const resultsContainer = document.getElementById('image-results');
        const predictionsList = document.getElementById('image-predictions');

        resultsContainer.classList.remove('hidden');

        // Display top predictions
        predictionsList.innerHTML = '';
        data.predictions.forEach((pred, index) => {
            const itemHTML = `
                <div class="prediction-item">
                    <div>
                        <span style="color: var(--accent-cyan); font-weight: 700;">#${pred.rank}</span>
                        <span class="prediction-item-label">${pred.label}</span>
                    </div>
                    <span class="prediction-item-confidence">${(pred.confidence * 100).toFixed(1)}%</span>
                </div>
            `;
            predictionsList.innerHTML += itemHTML;
        });
    }

    // Display sentiment analysis results
    static displaySentimentResults(data) {
        const resultsContainer = document.getElementById('sentiment-results');
        const sentimentValue = document.getElementById('sentiment-value');
        const sentimentConfidence = document.getElementById('sentiment-confidence');
        const sentimentBreakdown = document.getElementById('sentiment-breakdown');
        const emotionsContainer = document.getElementById('emotions-container');
        const emotionsList = document.getElementById('emotions-list');

        resultsContainer.classList.remove('hidden');

        // Display sentiment
        sentimentValue.textContent = data.sentiment;
        sentimentValue.className = `sentiment-badge ${data.sentiment.toLowerCase()}`;
        sentimentConfidence.textContent = `${(data.confidence * 100).toFixed(1)}%`;

        // Display all sentiments breakdown
        sentimentBreakdown.innerHTML = '';
        for (const [sentiment, confidence] of Object.entries(data.all_confidences)) {
            const barHTML = `
                <div class="confidence-bar">
                    <span class="confidence-bar-label">${sentiment}</span>
                    <div class="confidence-bar-track">
                        <div class="confidence-bar-fill" style="width: ${confidence * 100}%">
                            <span class="confidence-bar-value">${(confidence * 100).toFixed(1)}%</span>
                        </div>
                    </div>
                </div>
            `;
            sentimentBreakdown.innerHTML += barHTML;
        }

        // Display emotions if detected
        if (data.emotions && Object.keys(data.emotions).length > 0) {
            emotionsContainer.classList.remove('hidden');
            emotionsList.innerHTML = '';

            for (const [emotion, score] of Object.entries(data.emotions)) {
                const emotionHTML = `
                    <div class="confidence-bar">
                        <span class="confidence-bar-label">${emotion.charAt(0).toUpperCase() + emotion.slice(1)}</span>
                        <div class="confidence-bar-track">
                            <div class="confidence-bar-fill" style="width: ${score * 100}%">
                                <span class="confidence-bar-value">${(score * 100).toFixed(0)}%</span>
                            </div>
                        </div>
                    </div>
                `;
                emotionsList.innerHTML += emotionHTML;
            }
        } else {
            emotionsContainer.classList.add('hidden');
        }
    }

    // Display predictive model results
    static displayPredictiveResults(data) {
        const resultsContainer = document.getElementById('predictive-results');
        const metricsDiv = document.getElementById('training-metrics');
        const visualizationsDiv = document.getElementById('training-visualizations');

        resultsContainer.classList.remove('hidden');

        // Display metrics
        metricsDiv.innerHTML = `
            <div class="prediction-main">
                <span class="prediction-label">Test Accuracy:</span>
                <span class="prediction-value">${(data.evaluation.accuracy * 100).toFixed(2)}%</span>
            </div>
            <div class="prediction-main">
                <span class="prediction-label">Test Loss:</span>
                <span class="prediction-value">${data.evaluation.loss.toFixed(4)}</span>
            </div>
        `;

        // Display visualizations
        visualizationsDiv.innerHTML = `
            <div style="margin-bottom: 1rem;">
                <h4>Training History</h4>
                <img src="${data.training_plot}" alt="Training History" style="width: 100%; border-radius: 0.5rem;">
            </div>
            <div>
                <h4>Feature Importance</h4>
                <img src="${data.importance_plot}" alt="Feature Importance" style="width: 100%; border-radius: 0.5rem;">
            </div>
        `;
    }
}

// Toast notification system
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;

    container.appendChild(toast);

    // Auto remove after 3 seconds
    setTimeout(() => {
        toast.style.animation = 'toastSlideOut 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Loading overlay
function showLoading(show = true) {
    const overlay = document.getElementById('loading-overlay');
    if (show) {
        overlay.classList.remove('hidden');
    } else {
        overlay.classList.add('hidden');
    }
}
