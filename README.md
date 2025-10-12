# Simple MNIST Neural Network from Scratch in NumPy

A minimal implementation of a two-layer neural network built entirely from scratch using only NumPy for MNIST digit classification. This project demonstrates the fundamental mathematics and algorithms behind neural networks without relying on high-level frameworks like TensorFlow or Keras.

## 📋 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Mathematical Foundation](#mathematical-foundation)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Code Structure](#code-structure)
- [Educational Value](#educational-value)
- [Key Functions](#key-functions)
- [Performance](#performance)
- [Visualization](#visualization)
- [Extensions and Improvements](#extensions-and-improvements)
- [References](#references)

## 🔍 Overview

This project implements a simple two-layer neural network from scratch to classify handwritten digits from the MNIST dataset. The implementation focuses on educational clarity, showing every mathematical step of forward propagation, backward propagation, and gradient descent without any machine learning libraries.

**Key Features:**
- Pure NumPy implementation (no TensorFlow/PyTorch)
- Complete mathematical derivations included
- Step-by-step gradient descent visualization
- Achieves ~84% accuracy on MNIST digit classification
- Educational focus with detailed explanations

## 🏗️ Architecture

The neural network consists of a simple two-layer architecture:

### Layer Configuration:
- **Input Layer** \\(A^{[0]}\\): 784 units (28×28 pixels)
- **Hidden Layer** \\(A^{[1]}\\): 10 units with ReLU activation
- **Output Layer** \\(A^{[2]}\\): 10 units with Softmax activation (digit classes 0-9)

### Network Dimensions:
```
Input:  784 × m (where m = number of training examples)
Layer 1: 10 × 784 weights + 10 × 1 biases
Layer 2: 10 × 10 weights + 10 × 1 biases
Output: 10 × m (probability distribution over 10 classes)
```

## 📊 Mathematical Foundation

### Forward Propagation

The forward pass equations are:

**Layer 1:**
\\[ Z^{[1]} = W^{[1]} X + b^{[1]} \\]
\\[ A^{[1]} = g_{ReLU}(Z^{[1]}) \\]

**Layer 2:**
\\[ Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]} \\]
\\[ A^{[2]} = g_{softmax}(Z^{[2]}) \\]

### Backward Propagation

The gradient computation follows:

**Output Layer Gradients:**
\\[ dZ^{[2]} = A^{[2]} - Y \\]
\\[ dW^{[2]} = \\frac{1}{m} dZ^{[2]} A^{[1]T} \\]
\\[ db^{[2]} = \\frac{1}{m} \\sum dZ^{[2]} \\]

**Hidden Layer Gradients:**
\\[ dZ^{[1]} = W^{[2]T} dZ^{[2]} \\cdot g^{[1]'}(Z^{[1]}) \\]
\\[ dW^{[1]} = \\frac{1}{m} dZ^{[1]} A^{[0]T} \\]
\\[ db^{[1]} = \\frac{1}{m} \\sum dZ^{[1]} \\]

### Parameter Updates

Gradient descent updates:
\\[ W^{[2]} := W^{[2]} - \\alpha dW^{[2]} \\]
\\[ b^{[2]} := b^{[2]} - \\alpha db^{[2]} \\]
\\[ W^{[1]} := W^{[1]} - \\alpha dW^{[1]} \\]
\\[ b^{[1]} := b^{[1]} - \\alpha db^{[1]} \\]

## 📈 Dataset

**MNIST Digit Recognition Dataset:**
- **Total samples**: 42,000 images
- **Training set**: 41,000 images
- **Development set**: 1,000 images  
- **Image size**: 28×28 pixels (784 features)
- **Classes**: 10 digits (0-9)
- **Data format**: Grayscale images normalized to [0, 1]

### Data Preprocessing:
```python
# Normalization
X_train = X_train / 255.0
X_dev = X_dev / 255.0

# One-hot encoding for labels
Y_one_hot = one_hot(Y_train)  # Shape: (10, m)
```

## 🛠️ Requirements

```
numpy
pandas
matplotlib
```

## 💾 Installation

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd simple-mnist-nn-numpy
```

2. **Install dependencies:**
```bash
pip install numpy pandas matplotlib
```

3. **For Kaggle environment:**
```python
# Data is loaded from Kaggle input directory
data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
```

## 🚀 Usage

### Quick Start:
```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Load and preprocess data
data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

# Split into train/dev sets
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n] / 255.0

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n] / 255.0

# Train the model
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)

# Make predictions
predictions = make_predictions(X_train, W1, b1, W2, b2)
accuracy = get_accuracy(predictions, Y_train)
print(f"Training Accuracy: {accuracy:.3f}")
```

## 📊 Results

### Training Performance:
- **Final Training Accuracy**: ~84%
- **Development Set Accuracy**: ~82%
- **Training Time**: ~40 seconds for 500 iterations
- **Learning Rate**: 0.10

### Training Progress:
```
Iteration 0:   Accuracy = 10.5%
Iteration 50:  Accuracy = 34.9%
Iteration 100: Accuracy = 54.1%
Iteration 200: Accuracy = 73.4%
Iteration 500: Accuracy = 84.0%
```

## 🗂️ Code Structure

### Core Components:

**1. Parameter Initialization:**
```python
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2
```

**2. Activation Functions:**
```python
def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
```

**3. Forward Propagation:**
```python
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2
```

**4. Backward Propagation:**
```python
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2
```

## 🎓 Educational Value

This implementation serves as an excellent learning resource for:

### Concepts Demonstrated:
- **Matrix Operations**: Understanding how neural networks process batches of data
- **Gradient Descent**: Step-by-step optimization algorithm implementation
- **Backpropagation**: Detailed gradient computation through chain rule
- **Activation Functions**: ReLU and Softmax implementations from scratch
- **One-Hot Encoding**: Converting categorical labels for multi-class classification
- **Vectorization**: Efficient NumPy operations for batch processing

### Mathematical Insights:
- How gradients flow backward through the network
- The role of activation functions in learning non-linear patterns
- Parameter initialization strategies
- Loss function optimization dynamics

## ⚙️ Key Functions

### Training Loop:
```python
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        if i % 10 == 0:
            predictions = get_predictions(A2)
            print(f"Iteration: {i}, Accuracy: {get_accuracy(predictions, Y):.3f}")
    
    return W1, b1, W2, b2
```

### Prediction and Evaluation:
```python
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size
```

### Utility Functions:
```python
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def ReLU_deriv(Z):
    return Z > 0

def get_predictions(A2):
    return np.argmax(A2, 0)

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2
```

## 📊 Performance Analysis

### Strengths:
- **Educational Clarity**: Every step is explicit and understandable
- **No Black Boxes**: Complete implementation visibility
- **Good Accuracy**: 84% on MNIST is reasonable for such a simple architecture
- **Fast Training**: Converges quickly due to simple architecture

### Limitations:
- **Simple Architecture**: Only two layers limit learning capacity
- **No Regularization**: Potential for overfitting on larger datasets
- **Basic Optimization**: Uses vanilla gradient descent
- **Limited Scope**: Designed specifically for MNIST-sized problems

## 🖼️ Visualization

The implementation includes visualization capabilities:
```python
def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(current_image, W1, b1, W2, b2)
    label = Y_train[index]
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
    
    print(f"Prediction: {prediction}")
    print(f"Label: {label}")

# Test multiple predictions
for i in range(3):
    test_prediction(i, W1, b1, W2, b2)
```

## 🔄 Extensions and Improvements

### Possible Enhancements:
- **Add More Layers**: Implement deeper architectures
- **Regularization**: Add L2 regularization or dropout
- **Advanced Optimizers**: Implement Adam or RMSprop
- **Better Initialization**: Use Xavier or He initialization
- **Data Augmentation**: Rotate/shift images for better generalization
- **Cross-Validation**: Implement k-fold validation

### Implementation Examples:

**L2 Regularization:**
```python
def backward_prop_regularized(Z1, A1, Z2, A2, W1, W2, X, Y, lambd):
    m = Y.size
    one_hot_Y = one_hot(Y)
    
    dZ2 = A2 - one_hot_Y
    dW2 = (1/m) * dZ2.dot(A1.T) + (lambd/m) * W2
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = (1/m) * dZ1.dot(X.T) + (lambd/m) * W1
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2
```

**Xavier Initialization:**
```python
def init_params_xavier():
    W1 = np.random.randn(10, 784) * np.sqrt(1/784)
    b1 = np.zeros((10, 1))
    W2 = np.random.randn(10, 10) * np.sqrt(1/10)
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2
```

## 📚 References

1. **Deep Learning Book**: Ian Goodfellow, Yoshua Bengio, Aaron Courville
2. **Neural Networks and Deep Learning**: Michael Nielsen
3. **MNIST Database**: [yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
4. **NumPy Documentation**: [numpy.org](https://numpy.org)
5. **Original MNIST Paper**: LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Code optimization and vectorization
- Additional activation functions
- More sophisticated optimizers
- Enhanced visualization features
- Documentation improvements

### How to Contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙋‍♂️ FAQ

### Q: Why only 10 hidden units?
A: This simple architecture is designed for educational purposes. While more units would improve performance, the current setup allows for clear understanding of the mathematics involved.

### Q: Can this be extended to other datasets?
A: Yes! The code can be modified for other classification tasks by adjusting the input dimensions, number of classes, and output layer size.

### Q: Why not use modern optimizers?
A: Vanilla gradient descent helps students understand the fundamental optimization process before moving to advanced techniques like Adam or RMSprop.

---

**Note**: This implementation prioritizes educational value and mathematical transparency over performance. It's designed to help understand the fundamental concepts behind neural networks rather than achieve state-of-the-art results.
'
