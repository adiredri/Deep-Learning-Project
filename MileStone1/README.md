Milestone 1 - Binary Classification of Sign Language Digits

Project Summary:
This milestone focuses on building a binary image classifier using a custom neural network implemented from scratch with NumPy. The task is to distinguish between two specific hand signs representing digits from the Sign Language Digits dataset. This forms the basis for a deeper understanding of how neural networks function without relying on high-level deep learning frameworks.

Dataset:
- Source: Sign Language Digits Dataset
- Size: 5,000 grayscale images, 28x28 pixels each
- Classes Selected: Digits 6 and 8 only (binary classification)
- Split: 80% for training, 20% for testing

Objective:
To build a neural network capable of classifying hand gestures between the two selected digits, using:
- Only NumPy (no TensorFlow, PyTorch, or scikit-learn for modeling)
- Manual implementation of all neural network components, including training via gradient descent

Model Architecture:
- Input Layer: 784 neurons (flattened 28x28 images)
- Hidden Layer: 64 neurons
- Output Layer: 1 neuron (binary classification)
- Activation Function: Sigmoid for both layers
- Loss Function: Binary Cross-Entropy
- Training: Mini-batch Gradient Descent with backpropagation

Training Details:
- Epochs: 40
- Learning Rate: 0.01
- Framework: All code written in Python using NumPy only
- Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- Visualization: Loss curve over epochs, confusion matrix

Results:
- Accuracy: 95.6%
- Precision: 94.8%
- Recall: 96.2%
- F1-Score: 95.5%
The model demonstrated stable training behavior and strong performance on the test set, with low overfitting.

Key Challenges Addressed:
- Careful reshaping and normalization of input data
- Manual matrix operations and dimension handling in forward and backward propagation
- Numerically stable implementation of loss calculation
- Interpretation and visualization of model evaluation

Colab Notebook:
https://colab.research.google.com/drive/1cuL95B1NBeBOSJPIXAhxGEqqECeJL914?usp=sharing
