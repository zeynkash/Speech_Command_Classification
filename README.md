# Speech Command Classification Using Deep Learning

A deep learning project for classifying spoken voice commands using multiple neural network architectures. This project compares the performance of MLP, Bi-LSTM, and CNN models on the Speech Commands dataset and includes a real-time microphone demo for live command recognition.

---

## Project Overview

The objective of this project is to classify spoken commands into one of eight categories:

- go
- stop
- up
- down
- left
- right
- yes
- no

Three different deep learning architectures were developed and evaluated:

1. Multi-Layer Perceptron (MLP)
2. Bidirectional Long Short-Term Memory (Bi-LSTM)
3. Convolutional Neural Network (CNN)

The project also includes a browser-based microphone interface for real-time speech command prediction.

---

## Dataset

### Source
Speech Commands Dataset (Kaggle)

### Dataset Characteristics

- Total recordings available: ~105,835
- Samples used in this project: 30,956
- Audio length: 1 second
- Sampling rate: 16 kHz
- Classes: 8 speech commands
- Background noise classes excluded

### Selected Commands

| Class ID | Command |
|----------|----------|
| 0 | go |
| 1 | stop |
| 2 | up |
| 3 | down |
| 4 | left |
| 5 | right |
| 6 | yes |
| 7 | no |

---

## Technologies Used

### Deep Learning Framework
- PyTorch

### Audio Processing
- Librosa

### Data Processing
- NumPy
- Pandas
- Scikit-learn

### Visualization
- Matplotlib
- Seaborn

### Deployment and Training
- Google Colab GPU

---

# Data Preprocessing

## Audio Loading

All audio files are:

- Converted to mono
- Resampled to 16 kHz
- Trimmed or padded to 1 second

---

## Feature Extraction

Two different feature representations were used.

### 1. MFCC Features

Used by:

- MLP
- Bi-LSTM

Parameters:

- 40 MFCC coefficients
- 101 time frames

#### MLP Input

For the MLP model, temporal information is summarized using:

- Mean of MFCC coefficients
- Standard deviation of MFCC coefficients

Final feature vector size:

80 dimensions

```
40 means + 40 standard deviations
```

---

### 2. Mel-Spectrogram

Used by:

- CNN

Parameters:

- 64 Mel bands
- 101 time frames

Input shape:

```
64 × 101
```

The Mel-Spectrogram transforms audio into an image-like representation, allowing CNNs to learn spatial patterns in speech signals.

---

# Model Architectures

## 1. Multi-Layer Perceptron (MLP)

### Input

```
80-dimensional MFCC feature vector
```

### Architecture

```text
Input (80)

↓
Linear (512)
BatchNorm
ReLU
Dropout

↓
Linear (256)
BatchNorm
ReLU
Dropout

↓
Linear (128)
BatchNorm
ReLU
Dropout

↓
Linear (8)

Output
```

### Advantages

- Fast training
- Lightweight
- Simple architecture

### Best Accuracy

Approximately:

```text
94% - 95%
```

---

## 2. Bidirectional LSTM

### Input

```text
101 time steps × 40 MFCC features
```

### Architecture

```text
MFCC Sequence

↓
Bi-LSTM Layer 1
(hidden size = 128)

↓
Bi-LSTM Layer 2
(hidden size = 128)

↓
Layer Normalization

↓
Mean Pooling

↓
MLP Classifier

↓
8 Outputs
```

### Advantages

- Captures temporal dependencies
- Uses information from both past and future frames

### Best Accuracy

Approximately:

```text
95% - 96%
```

---

## 3. Custom CNN (Best Model)

### Input

```text
Mel-Spectrogram
64 × 101
```

### Architecture

```text
Input

↓
Conv2D (32)
BatchNorm
ReLU
MaxPool

↓
Conv2D (64)
BatchNorm
ReLU
MaxPool

↓
Conv2D (128)
BatchNorm
ReLU
MaxPool

↓
Flatten

↓
Linear (256)
ReLU
Dropout

↓
Linear (8)

Output
```

### Advantages

- Learns local spectral patterns
- Excellent performance on spectrogram images
- Strong generalization

### Best Validation Accuracy

```text
96.77%
```

---

# Training Configuration

### Optimizer

```python
Adam
```

### Loss Function

```python
CrossEntropyLoss
```

### Batch Size

```python
32
```

### Epochs

```python
50
```

### Device

```python
CUDA GPU
```

### Regularization

- Dropout
- Batch Normalization
- Layer Normalization (Bi-LSTM)

---

# Evaluation Metrics

The following metrics were used:

- Accuracy
- Precision (Macro)
- Recall (Macro)
- F1 Score (Macro)

Additionally:

- Confusion Matrix
- Training Loss Curves
- Validation Loss Curves
- Training Accuracy Curves
- Validation Accuracy Curves

---

# Results

| Model | Accuracy |
|---------|---------|
| MLP | ~94-95% |
| Bi-LSTM | ~95-96% |
| CNN | 96.77% |

---

## Performance Comparison

### MLP

Pros:

- Fastest training
- Small model size

Cons:

- Loses temporal information

---

### Bi-LSTM

Pros:

- Captures sequential dependencies
- Better speech modeling

Cons:

- Higher computational cost

---

### CNN

Pros:

- Highest accuracy
- Strong feature extraction capability
- Best generalization

Cons:

- Slightly larger model size

---

# Confusion Matrix Analysis

The CNN model achieved the best class separation.

Most commands were classified correctly with minimal confusion between classes.

Classes with similar acoustic patterns showed occasional misclassifications but overall performance remained strong.

---

# Real-Time Microphone Demo

A browser-based interface was developed to demonstrate real-time speech command recognition.

## Pipeline

```text
Microphone Input

↓
Record Audio (1.5 sec)

↓
WebM Audio

↓
FFmpeg Conversion

↓
WAV File

↓
Mel-Spectrogram

↓
Normalization

↓
CNN Model

↓
Prediction
```

---

## Features

- Browser microphone recording
- Real-time prediction
- Confidence scores
- Audio playback
- Waveform visualization
- Spectrogram visualization

---

# Saved Outputs

## Trained Models

```text
cnn_best.pth
mlp_best.pth
lstm_best.pth
```

## Training Histories

```text
cnn_history.npz
mlp_history.npz
lstm_history.npz
```

## Evaluation Results

```text
model_comparison.csv
```

## Visualizations

```text
confusion_matrix.png
loss_curves.png
accuracy_curves.png
comparison_bar_chart.png
```

---

# Project Structure

```text
Speech-Command-Classification/
│
├── data/
│   ├── audio/
│   └── processed/
│
├── models/
│   ├── cnn_best.pth
│   ├── mlp_best.pth
│   └── lstm_best.pth
│
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── mlp_training.ipynb
│   ├── lstm_training.ipynb
│   └── cnn_training.ipynb
│
├── outputs/
│   ├── confusion_matrices/
│   ├── curves/
│   └── comparison/
│
├── demo/
│   └── microphone_demo.ipynb
│
├── requirements.txt
└── README.md
```

---

# Key Findings

1. CNN achieved the highest performance with 96.77% accuracy.
2. Mel-Spectrograms are highly effective for speech command classification.
3. MFCC-based approaches also provide strong results.
4. Bidirectional LSTM successfully captures temporal speech information.
5. Models converged around 50 epochs.
6. A real-time deployment pipeline was successfully implemented.

---

# Future Work

Potential improvements include:

- Using the complete Speech Commands dataset
- Data augmentation techniques
- Transfer learning with pretrained audio models
- Mobile deployment
- Transformer-based architectures
- Quantization for edge devices

---

# Authors

Deep Learning Course Project

Speech Command Classification Using Deep Learning

Developed using PyTorch and Google Colab.
