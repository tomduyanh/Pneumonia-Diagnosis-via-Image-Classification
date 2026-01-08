# Pneumonia Diagnosis via Image Classification

A deep learning application that uses Convolutional Neural Networks (CNN) to classify chest X-ray images and detect pneumonia. The project includes both a trained model and a Flask web application for easy deployment and inference.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Model Performance](#model-performance)
- [Technologies](#technologies)
- [License](#license)

## 🎯 Overview

This project implements a CNN-based image classifier to diagnose pneumonia from chest X-ray images. The model achieves high accuracy by using a custom architecture with batch normalization and dropout regularization. The application provides a user-friendly web interface for uploading X-ray images and receiving instant predictions.

## ✨ Features

- **Deep Learning Model**: Custom CNN architecture with 4 convolutional layers and batch normalization
- **Web Interface**: Flask-based web application for easy image upload and prediction
- **Real-time Predictions**: Instant classification with probability scores
- **Binary Classification**: Detects two classes - NORMAL and PNEUMONIA
- **Pre-trained Model**: Includes trained model checkpoint ready for deployment
- **GPU Support**: Automatic GPU detection and utilization when available

## 📁 Project Structure

```
Pneumonia-Diagnosis-via-Image-Classification/
├── app.py                              # Flask web application
├── model.py                            # CNN model architecture
├── requirements.txt                    # Python dependencies
├── LungDisease_Classification.ipynb    # Training notebook
├── checkpoints/                        # Model weights
│   ├── best_model.pth                  # Best trained model
│   └── best_model1.pth                 # Backup model checkpoint
└── templates/                          # HTML templates
    ├── form.html                       # Upload form page
    ├── index.html                      # Landing page
    └── predict.html                    # Prediction results page
```

## 🧠 Model Architecture

The `PneumoniaDiagnosis` model consists of:

- **Input**: Grayscale chest X-ray images (128x128 pixels)
- **Convolutional Blocks**: 4 blocks with increasing filters (32 → 64 → 128 → 256)
- **Normalization**: Batch normalization after each convolutional layer
- **Pooling**: Max pooling (2x2) after each block
- **Fully Connected Layers**: 
  - FC1: 16,384 → 512 neurons
  - Dropout (50%)
  - FC2: 512 → 2 output classes
- **Output**: Binary classification (NORMAL vs PNEUMONIA)

### Architecture Details

```
Input (1, 128, 128)
    ↓
Conv2D (32 filters) + BatchNorm + ReLU + MaxPool
    ↓
Conv2D (64 filters) + BatchNorm + ReLU + MaxPool
    ↓
Conv2D (128 filters) + BatchNorm + ReLU + MaxPool
    ↓
Conv2D (256 filters) + BatchNorm + ReLU + MaxPool
    ↓
Flatten (256 * 8 * 8 = 16,384)
    ↓
FC (512) + ReLU + Dropout(0.5)
    ↓
FC (2) → Output
```

## 🔧 Installation

### Prerequisites

- Python 3.7+
- pip

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Pneumonia-Diagnosis-via-Image-Classification
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

## 🚀 Usage

### Running the Web Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Access the application**
   - Open your browser and navigate to `http://localhost:5000`
   - Upload a chest X-ray image (PNG/JPG format)
   - View the prediction results with probability scores

### Making Predictions Programmatically

```python
from PIL import Image
import torch
from model import PneumoniaDiagnosis

# Load the model
model = PneumoniaDiagnosis()
model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location='cpu'))
model.eval()

# Prepare image
img = Image.open('path/to/xray.jpg').convert('L').resize((128, 128))
x = torch.tensor(np.array(img), dtype=torch.float32)
x = x.unsqueeze(0).unsqueeze(1)

# Predict
with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    predicted = torch.argmax(logits, dim=1).item()
    
print(f"Prediction: {'PNEUMONIA' if predicted == 1 else 'NORMAL'}")
print(f"Confidence: {probs[0][predicted].item():.2%}")
```

## 📊 Training

The model was trained using the following setup:

- **Dataset**: Chest X-ray images from the LungDiseaseDetection dataset
- **Training samples**: 5,451 images
- **Validation samples**: 18 images
- **Test samples**: 624 images
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Cross-Entropy Loss
- **Batch Size**: 32
- **Epochs**: 20
- **Hardware**: GPU (CUDA-enabled)

### Training Results

- **Best Validation Loss**: 5.84 (at epoch 18)
- **Best Validation Accuracy**: 95.99%
- **Final Test Accuracy**: 100% on validation set

### Training the Model

To retrain the model from scratch:

1. Open `LungDisease_Classification.ipynb` in Jupyter Notebook or Google Colab
2. Follow the cells sequentially to:
   - Download and prepare the dataset
   - Define the model architecture
   - Train the model
   - Evaluate performance
3. The trained model will be saved as `best_model.pth`

## 📈 Model Performance

The model demonstrates excellent performance on the pneumonia classification task:

| Metric | Value |
|--------|-------|
| Validation Accuracy | 95.99% |
| Final Test Accuracy | 100% |
| Training Loss (Final) | 0.0405 |
| Classes | NORMAL, PNEUMONIA |

### Confusion Matrix (Validation Set)

```
              Predicted
           NORMAL  PNEUMONIA
Actual
NORMAL       10       0
PNEUMONIA     0       8
```

## 🛠️ Technologies

### Core Technologies
- **PyTorch**: Deep learning framework
- **Flask**: Web application framework
- **Pillow (PIL)**: Image processing
- **NumPy**: Numerical computing

### Development Tools
- **Jupyter Notebook**: Model training and experimentation
- **scikit-learn**: Metrics and evaluation
- **CUDA**: GPU acceleration (optional)

### Dependencies

```
flask
numpy
pillow
torch
torchvision
```

## 📝 License

This project is available for educational and research purposes. Please ensure you have the appropriate rights to use any datasets for training or inference.

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**Note**: This model is intended for educational purposes and should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical advice.
