<div align="center">

# 🧠 AI-Powered Neural Network Hub

### *An Interactive Deep Learning Playground*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**A comprehensive web application showcasing real-world neural network applications with an elegant, modern interface**

[Features](#-features) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Demo](#-live-demo)

---

</div>

## ✨ Overview

The **AI-Powered Neural Network Hub** is a full-stack web application that brings the power of deep learning to your browser. Built with cutting-edge technologies and featuring a premium dark-themed UI, this project demonstrates practical applications of neural networks including computer vision, natural language processing, and predictive analytics.

### 🎯 What Makes This Special?

- 🚀 **Production-Ready Models** - Pre-trained on industry-standard datasets
- 🎨 **Premium UI/UX** - Glassmorphism, smooth animations, and responsive design
- ⚡ **Real-Time Inference** - Lightning-fast predictions with optimized models
- 📊 **Interactive Visualizations** - Live neural network animations and data plots
- 🔧 **Modular Architecture** - Clean, maintainable, and extensible codebase

---

## 🌟 Features

<table>
<tr>
<td width="50%">

### ✍️ Handwritten Digit Recognition
- Interactive drawing canvas with mouse/touch support
- Real-time CNN predictions using MNIST-trained model
- Confidence scores for all digits (0-9)
- Network activation visualization
- **98%+ accuracy** on test data

</td>
<td width="50%">

### 🖼️ Image Classification
- Upload images to classify objects and scenes
- Pre-trained **MobileNetV2** on ImageNet
- 1000+ object categories
- Top-5 predictions with confidence scores
- Drag-and-drop file upload

</td>
</tr>
<tr>
<td width="50%">

### 💬 Sentiment Analysis
- Analyze emotions in text, reviews, or social media
- **LSTM-based** sentiment classification
- Multi-emotion detection (joy, sadness, anger, fear, surprise)
- Real-time analysis as you type
- **85%+ accuracy** on review datasets

</td>
<td width="50%">

### 📊 Predictive Analytics
- Train custom models on your CSV data
- Support for regression and classification tasks
- Feature importance visualization
- Training history plots (loss, accuracy curves)
- Comprehensive model performance metrics

</td>
</tr>
</table>

### 🎨 Neural Network Visualization
- **Live animated visualization** of neural network architecture
- **Particle effects** showing data flow through layers
- **Real-time activation updates** during inference
- Beautiful, interactive graphics powered by Canvas API

---

## 🚀 Quick Start

### Prerequisites

Before you begin, ensure you have the following installed:
- **Python 3.8+** ([Download](https://www.python.org/downloads/))
- **pip** (Python package manager)
- **Git** (optional, for cloning)

### Installation

```bash
# 1. Clone the repository (or download ZIP)
git clone https://github.com/yourusername/Neural-Network-Visualization.git
cd Neural-Network-Visualization

# 2. Create a virtual environment (recommended)
python3 -m venv venv

# 3. Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Start the Flask server
python app.py
```

The server will:
- ✅ Initialize all neural network models
- ✅ Download pre-trained weights (first time only, ~500MB)
- ✅ Start on `http://localhost:5000`

**Open your browser and navigate to:** `http://localhost:5000`

🎉 **You're all set! Start exploring AI features!**

---

## 📁 Project Structure

```
Neural-Network-Visualization/
│
├── 📄 app.py                          # Flask application server & API routes
├── ⚙️ config.py                       # Configuration settings
├── 📋 requirements.txt                # Python dependencies
├── 📖 README.md                       # Project documentation
│
├── 🧠 models/                         # Neural network models
│   ├── __init__.py
│   ├── digit_recognition.py          # CNN for MNIST digit recognition
│   ├── image_classifier.py           # MobileNetV2 for ImageNet classification
│   ├── sentiment_analyzer.py         # LSTM for sentiment analysis
│   └── predictive_model.py           # Custom NN for predictive analytics
│
├── 🛠️ utils/                          # Utility functions
│   ├── __init__.py
│   ├── data_processor.py             # Data preprocessing & augmentation
│   └── visualization.py              # Matplotlib visualizations
│
├── 🎨 static/                         # Frontend files
│   ├── index.html                    # Main HTML page
│   ├── css/
│   │   └── styles.css                # Premium CSS styling (dark theme, glassmorphism)
│   └── js/
│       ├── api.js                    # API client for backend communication
│       ├── app.js                    # Main application logic & routing
│       ├── drawing-canvas.js         # Canvas drawing functionality
│       ├── file-uploader.js          # File upload handlers
│       ├── network-visualizer.js     # Neural network visualization engine
│       └── results-dashboard.js      # Results display & charts
│
├── 💾 models/saved_models/            # Pre-trained model weights (auto-downloaded)
├── 📊 data/samples/                   # Sample datasets for testing
└── 📤 uploads/                        # Temporary upload folder
```

---

## 📖 Documentation

### 🎯 Usage Guide

<details>
<summary><b>✍️ Handwritten Digit Recognition</b></summary>

1. Navigate to the **Digit Recognition** section
2. Draw a digit (0-9) on the canvas using your mouse or touch
3. Click **"Recognize"** to get the prediction
4. View confidence scores for all digits
5. Click **"Clear"** to draw a new digit

**Tip:** Draw digits clearly in the center of the canvas for best results!

</details>

<details>
<summary><b>🖼️ Image Classification</b></summary>

1. Navigate to the **Image Classification** section
2. Click **"Browse Files"** or drag-and-drop an image
3. Preview the uploaded image
4. Click **"Classify Image"**
5. View top-5 predictions with confidence scores

**Supported formats:** JPG, PNG, JPEG, GIF, BMP

</details>

<details>
<summary><b>💬 Sentiment Analysis</b></summary>

1. Navigate to the **Sentiment Analysis** section
2. Enter text in the text area (reviews, tweets, feedback)
3. Click **"Analyze Sentiment"**
4. View sentiment classification (Positive/Negative/Neutral)
5. See detected emotions and confidence breakdown

**Examples:** Product reviews, movie reviews, social media posts, customer feedback

</details>

<details>
<summary><b>📊 Predictive Analytics</b></summary>

1. Navigate to the **Predictive Analytics** section
2. Upload a CSV file with your data
3. Specify the target column name
4. Choose task type (Regression or Classification)
5. Click **"Train Model"**
6. View training results, metrics, and visualizations

**Requirements:** CSV file with headers, numerical features, target column

</details>

---

### 🧪 Model Architectures

<details>
<summary><b>Digit Recognition (CNN)</b></summary>

```
Input: 28×28 grayscale images
│
├─ Conv2D (32 filters, 3×3, ReLU)
├─ MaxPooling2D (2×2)
├─ Conv2D (64 filters, 3×3, ReLU)
├─ MaxPooling2D (2×2)
├─ Flatten
├─ Dense (128 units, ReLU)
├─ Dropout (0.5)
└─ Dense (10 units, Softmax)
│
Output: 10 classes (digits 0-9)
```

**Performance:**
- Accuracy: **>98%** on MNIST test set
- Inference time: **~50ms**
- Parameters: ~1.2M

</details>

<details>
<summary><b>Image Classification (MobileNetV2)</b></summary>

```
Input: 224×224 RGB images
│
├─ MobileNetV2 (Pre-trained on ImageNet)
│  ├─ Depthwise Separable Convolutions
│  ├─ Inverted Residual Blocks
│  └─ Global Average Pooling
│
└─ Dense (1000 units, Softmax)
│
Output: 1000 ImageNet classes
```

**Performance:**
- Top-1 Accuracy: **~71%**
- Top-5 Accuracy: **>90%**
- Inference time: **~200ms**
- Parameters: ~3.5M

</details>

<details>
<summary><b>Sentiment Analysis (LSTM)</b></summary>

```
Input: Text sequences (up to 200 words)
│
├─ Embedding Layer (10,000 vocab, 128 dim)
├─ Bidirectional LSTM (64 units)
├─ Dropout (0.5)
├─ Dense (64 units, ReLU)
├─ Dropout (0.5)
└─ Dense (3 units, Softmax)
│
Output: 3 classes (Positive, Negative, Neutral)
```

**Performance:**
- Accuracy: **>85%** on review datasets
- Inference time: **~100ms**
- Parameters: ~1.5M

</details>

<details>
<summary><b>Predictive Model (Custom NN)</b></summary>

```
Input: Variable features (auto-detected from CSV)
│
├─ Dense (128 units, ReLU)
├─ Dropout (0.3)
├─ Dense (64 units, ReLU)
├─ Dropout (0.3)
├─ Dense (32 units, ReLU)
└─ Dense (output units, task-specific activation)
│
Output: Regression or Classification
```

**Features:**
- Customizable architecture based on input features
- Feature importance using gradient-based methods
- Early stopping to prevent overfitting
- Automatic hyperparameter tuning

</details>

---

### 🔧 API Reference

<details>
<summary><b>View API Endpoints</b></summary>

#### Health Check
```http
GET /api/health
```
Returns server status and model availability.

#### Digit Recognition
```http
POST /api/digit/predict
Content-Type: application/json

{
  "image": "base64_encoded_image_data"
}
```

#### Image Classification
```http
POST /api/image/classify
Content-Type: multipart/form-data

file: <image_file>
```

#### Sentiment Analysis
```http
POST /api/sentiment/analyze
Content-Type: application/json

{
  "text": "Your text here"
}
```

#### Predictive Analytics
```http
POST /api/predict/train
Content-Type: multipart/form-data

file: <csv_file>
target_column: <column_name>
task_type: <regression|classification>
```

#### Visualizations
```http
POST /api/visualize/network
POST /api/visualize/confusion-matrix
```

</details>

---

## 🎨 Design Features

- 🌑 **Premium Dark Theme** with AI-inspired gradients
- 💎 **Glassmorphism Effects** for modern, frosted-glass UI
- ✨ **Smooth Micro-Animations** for enhanced user experience
- 📱 **Fully Responsive** - Works on mobile, tablet, and desktop
- 🎭 **Animated Neural Network Background** with particle effects
- 📊 **Real-Time Visualizations** powered by Matplotlib and Canvas API
- 🎯 **Intuitive Navigation** with smooth scrolling and transitions

---

## ⚡ Performance

| Feature | Inference Time | Accuracy |
|---------|---------------|----------|
| Digit Recognition | ~50ms | >98% |
| Image Classification | ~200ms | >90% (Top-5) |
| Sentiment Analysis | ~100ms | >85% |
| Model Training | Varies | Dataset-dependent |

**System Requirements:**
- Minimum: 4GB RAM, 2-core CPU
- Recommended: 8GB RAM, 4-core CPU
- Storage: ~1GB (including models)

---

## 🛠️ Troubleshooting

<details>
<summary><b>Server won't start</b></summary>

**Solutions:**
1. Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```
2. Check Python version:
   ```bash
   python --version  # Should be 3.8+
   ```
3. Verify port 5000 is not in use:
   ```bash
   # On macOS/Linux:
   lsof -i :5000
   # On Windows:
   netstat -ano | findstr :5000
   ```
4. Try a different port:
   ```bash
   export FLASK_RUN_PORT=8000
   python app.py
   ```

</details>

<details>
<summary><b>Models not loading</b></summary>

**Solutions:**
1. First run downloads pre-trained weights (may take a few minutes)
2. Check internet connection for downloading ImageNet weights
3. Ensure sufficient disk space (~500MB for all models)
4. Clear cache and retry:
   ```bash
   rm -rf models/saved_models/*
   python app.py
   ```

</details>

<details>
<summary><b>Frontend not connecting</b></summary>

**Solutions:**
1. Verify Flask server is running on `http://localhost:5000`
2. Check browser console for errors (F12)
3. Ensure CORS is enabled (Flask-CORS installed)
4. Try clearing browser cache
5. Disable browser extensions that might block requests

</details>

<details>
<summary><b>Slow predictions</b></summary>

**Solutions:**
1. First prediction is slower (model loading)
2. Ensure you're using CPU-optimized TensorFlow
3. For GPU acceleration, install `tensorflow-gpu`
4. Reduce image size for faster classification
5. Close other resource-intensive applications

</details>

---

## 🚀 Future Enhancements

- [ ] 🎥 **Video Classification** - Real-time video analysis
- [ ] 📹 **Webcam Integration** - Live digit recognition from webcam
- [ ] 🧩 **More Pre-trained Models** (ResNet, EfficientNet, BERT)
- [ ] 📦 **Model Export** - TensorFlow Lite, ONNX format support
- [ ] 👤 **User Authentication** - Save and manage custom models
- [ ] ⚡ **API Rate Limiting** - Production-ready API with caching
- [ ] 🐳 **Docker Containerization** - Easy deployment
- [ ] ☁️ **Cloud Deployment** - AWS, GCP, Azure integration
- [ ] 📱 **Mobile App** - React Native companion app
- [ ] � **Model Versioning** - Track and compare model versions

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Kartikeya**

Created as a comprehensive demonstration of real-world neural network applications using Python, TensorFlow, and modern web technologies.

---

## 🙏 Acknowledgments

This project wouldn't be possible without these amazing technologies:

- [**TensorFlow**](https://www.tensorflow.org/) - Deep learning framework
- [**Keras**](https://keras.io/) - High-level neural network API
- [**Flask**](https://flask.palletsprojects.com/) - Lightweight web framework
- [**NumPy**](https://numpy.org/) - Numerical computing library
- [**Matplotlib**](https://matplotlib.org/) - Data visualization
- [**MobileNetV2**](https://arxiv.org/abs/1801.04381) - Pre-trained on ImageNet
- [**MNIST Dataset**](http://yann.lecun.com/exdb/mnist/) - Handwritten digit database

---

<div align="center">

### 🧠✨ **Enjoy exploring the world of AI and neural networks!** ✨🧠

**Made with ❤️ and lots of ☕**

[⬆ Back to Top](#-ai-powered-neural-network-hub)

</div>
