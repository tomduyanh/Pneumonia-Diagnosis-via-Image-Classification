# Pneumonia Diagnosis via Image Classification

A learning project that explores using Convolutional Neural Networks (CNN) to classify chest X-ray images for pneumonia detection. This includes a basic trained model and a simple Flask web application for demonstration purposes.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies](#technologies)

## Overview

This is an educational project that implements a CNN-based image classifier for pneumonia detection from chest X-ray images. The model uses a relatively simple architecture with batch normalization and dropout for regularization. A basic Flask web interface is provided for testing predictions.

> **Important Disclaimer**: This is a student/learning project and should NOT be used for actual medical diagnosis. The model has significant limitations and has only been tested on a small validation set. Always consult qualified medical professionals for health concerns.

## Features

- Custom CNN architecture with 4 convolutional layers
- Simple Flask web application for image upload
- Binary classification (NORMAL vs PNEUMONIA)
- Pre-trained model checkpoint included
- Basic GPU support when available

## Project Structure

```
Pneumonia-Diagnosis-via-Image-Classification/
├── app.py                              # Flask web application
├── model.py                            # CNN model architecture
├── requirements.txt                    # Python dependencies
├── LungDisease_Classification.ipynb    # Training notebook
├── checkpoints/                        # Model weights
│   ├── best_model.pth
│   └── best_model1.pth
└── templates/                          # HTML templates
    ├── form.html
    ├── index.html
    └── predict.html
```

## Model Architecture

The `PneumoniaDiagnosis` model is a straightforward CNN with:

- **Input**: Grayscale chest X-ray images (128x128 pixels)
- **4 Convolutional Blocks**: Filters increase from 32 → 64 → 128 → 256
- **Batch Normalization**: Applied after each convolutional layer
- **Max Pooling**: 2x2 pooling after each block
- **Fully Connected Layers**: 
  - FC1: 16,384 → 512 neurons with 50% dropout
  - FC2: 512 → 2 output classes

Architecture summary:
```
Input (1, 128, 128)
    ↓
[Conv2D(32) + BatchNorm + ReLU + MaxPool] × 4 blocks
    ↓
Flatten → FC(512) + Dropout(0.5) → FC(2)
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/tomduyanh/Competitive-Programming.git
   cd Pneumonia-Diagnosis-via-Image-Classification
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Web Application

1. Start the Flask development server:
   ```bash
   python app.py
   ```

2. Open your browser to `http://localhost:5000`
3. Upload a chest X-ray image to see the model's prediction

Note: The web application is intended for demonstration only and runs in debug mode.

The model was trained on a subset of chest X-ray images:

- **Optimizer**: Adam (learning rate: 0.001)
- **Loss**: Cross-Entropy
- **Batch Size**: 32
- **Epochs**: 20

To retrain the model, open `LungDisease_Classification.ipynb` in Jupyter or Google Colab and run the cells sequentially.

## Technologies

- **PyTorch** - Deep learning framework
- **Flask** - Web framework
- **Pillow** - Image processing
- **NumPy** - Numerical computing

## Acknowledgments

This project was created for educational purposes to learn about CNNs and medical image classification.

---

**Medical Disclaimer**: This project is strictly for educational and demonstration purposes. It is NOT a medical device and should NEVER be used to diagnose, treat, or make medical decisions. The model has significant limitations and has not been clinically validated. Always seek professional medical advice from qualified healthcare providers.
