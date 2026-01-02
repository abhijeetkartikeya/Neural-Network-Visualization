# AI-Powered Neural Network Hub 🧠

A comprehensive web application showcasing real-world neural network applications built with Python (TensorFlow, NumPy, Matplotlib) backend and HTML/CSS/JavaScript frontend.

## 🌟 Features

### 1. Handwritten Digit Recognition ✍️
- **Interactive drawing canvas** with mouse and touch support
- **Real-time CNN prediction** using MNIST-trained model
- **Confidence scores** for all digits (0-9)
- **Network activation visualization**

### 2. Image Classification 🖼️
- **Upload any image** to classify objects and scenes
- **Pre-trained MobileNetV2** on ImageNet (1000+ categories)
- **Top-5 predictions** with confidence scores
- **Drag-and-drop** file upload

### 3. Sentiment Analysis 💬
- **Analyze emotions** in text, reviews, or social media posts
- **LSTM-based** sentiment classification
- **Emotion detection** (joy, sadness, anger, fear, surprise)
- **Real-time analysis** as you type

### 4. Predictive Analytics 📊
- **Train custom models** on your own CSV data
- **Support for regression and classification** tasks
- **Feature importance visualization**
- **Training history plots** (loss, accuracy curves)
- **Model performance metrics**

### 5. Neural Network Visualization 🎨
- **Live animated visualization** of neural network architecture
- **Particle effects** showing data flow through layers
- **Real-time activation updates** during inference

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd "/Users/kartikeya/Documents/coding/project NN"
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - TensorFlow 2.15.0
   - NumPy
   - Matplotlib
   - Flask & Flask-CORS
   - Pillow (image processing)
   - pandas & scikit-learn

### Running the Application

1. **Start the Flask server:**
   ```bash
   python app.py
   ```

   The server will:
   - Initialize all neural network models
   - Download pre-trained weights (first time only)
   - Start on `http://localhost:5000`

2. **Open your browser:**
   Navigate to `http://localhost:5000`

3. **Start using the AI features!**

## 📁 Project Structure

```
project NN/
├── app.py                      # Flask application server
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── models/                     # Neural network models
│   ├── __init__.py
│   ├── digit_recognition.py    # CNN for MNIST
│   ├── image_classifier.py     # MobileNetV2 for ImageNet
│   ├── sentiment_analyzer.py   # LSTM for sentiment
│   └── predictive_model.py     # Custom NN for predictions
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── data_processor.py       # Data preprocessing
│   └── visualization.py        # Matplotlib visualizations
├── static/                     # Frontend files
│   ├── index.html             # Main HTML page
│   ├── css/
│   │   └── styles.css         # Premium CSS styling
│   └── js/
│       ├── api.js             # API client
│       ├── app.js             # Main application logic
│       ├── drawing-canvas.js  # Canvas drawing
│       ├── file-uploader.js   # File upload handlers
│       ├── network-visualizer.js  # Network visualization
│       └── results-dashboard.js   # Results display
├── models/saved_models/        # Pre-trained model weights
├── data/samples/               # Sample datasets
└── uploads/                    # Temporary upload folder
```

## 🎯 Usage Guide

### Digit Recognition
1. Draw a digit (0-9) on the canvas using your mouse or touch
2. Click "Recognize" to get the prediction
3. View confidence scores for all digits
4. Click "Clear" to draw a new digit

### Image Classification
1. Click "Browse Files" or drag-and-drop an image
2. Preview the image
3. Click "Classify Image"
4. View top-5 predictions with confidence scores

### Sentiment Analysis
1. Enter text in the text area (reviews, tweets, feedback)
2. Click "Analyze Sentiment"
3. View sentiment classification (Positive/Negative/Neutral)
4. See detected emotions and confidence breakdown

### Predictive Analytics
1. Upload a CSV file with your data
2. Specify the target column name
3. Choose task type (Regression or Classification)
4. Click "Train Model"
5. View training results, metrics, and visualizations

## 🧪 Model Architectures

### Digit Recognition (CNN)
- **Input:** 28x28 grayscale images
- **Architecture:** Conv2D → MaxPooling → Conv2D → MaxPooling → Dense
- **Output:** 10 classes (digits 0-9)
- **Accuracy:** >98% on MNIST test set

### Image Classification (MobileNetV2)
- **Input:** 224x224 RGB images
- **Pre-trained:** ImageNet (1.4M images, 1000 classes)
- **Transfer Learning:** Fine-tuned for custom datasets
- **Top-5 Accuracy:** >90%

### Sentiment Analysis (LSTM)
- **Input:** Text sequences (up to 200 words)
- **Architecture:** Embedding → Bidirectional LSTM → Dense
- **Output:** 3 classes (Positive, Negative, Neutral)
- **Accuracy:** >85% on review datasets

### Predictive Model (Custom NN)
- **Customizable architecture** based on input features
- **Support for regression and classification**
- **Feature importance** using gradient-based methods
- **Early stopping** to prevent overfitting

## 🎨 Design Features

- **Premium dark theme** with AI-inspired gradients
- **Glassmorphism effects** for modern UI
- **Smooth micro-animations** for enhanced UX
- **Responsive design** for mobile, tablet, and desktop
- **Animated neural network background**
- **Real-time visualizations** with Matplotlib

## 🔧 API Endpoints

### Health Check
- `GET /api/health` - Check server status

### Digit Recognition
- `POST /api/digit/predict` - Predict handwritten digit

### Image Classification
- `POST /api/image/classify` - Classify uploaded image

### Sentiment Analysis
- `POST /api/sentiment/analyze` - Analyze text sentiment

### Predictive Analytics
- `POST /api/predict/train` - Train custom model on CSV data

### Visualizations
- `POST /api/visualize/network` - Generate network architecture diagram
- `POST /api/visualize/confusion-matrix` - Generate confusion matrix

## 📊 Performance

- **Digit Recognition:** ~50ms inference time
- **Image Classification:** ~200ms inference time
- **Sentiment Analysis:** ~100ms inference time
- **Model Training:** Varies based on dataset size

## 🛠️ Troubleshooting

### Server won't start
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)
- Verify port 5000 is not in use

### Models not loading
- First run downloads pre-trained weights (may take a few minutes)
- Check internet connection for downloading ImageNet weights
- Ensure sufficient disk space (~500MB for all models)

### Frontend not connecting
- Verify Flask server is running on `http://localhost:5000`
- Check browser console for errors
- Ensure CORS is enabled (Flask-CORS installed)

## 🚀 Future Enhancements

- [ ] Add more pre-trained models (ResNet, EfficientNet)
- [ ] Support for video classification
- [ ] Real-time webcam digit recognition
- [ ] Model export for deployment (TensorFlow Lite, ONNX)
- [ ] User authentication and model saving
- [ ] API rate limiting and caching
- [ ] Docker containerization
- [ ] Cloud deployment (AWS, GCP, Azure)

## 📝 License

This project is open source and available for educational purposes.

## 👨‍💻 Author

Created as a demonstration of real-world neural network applications using Python and modern web technologies.

## 🙏 Acknowledgments

- **TensorFlow** for deep learning framework
- **Keras** for high-level neural network API
- **MobileNetV2** pre-trained on ImageNet
- **MNIST** dataset for digit recognition
- **Flask** for web framework
- **NumPy** for numerical computing
- **Matplotlib** for visualizations

---

**Enjoy exploring the world of AI and neural networks! 🧠✨**
# Neural-Network-Visualization
