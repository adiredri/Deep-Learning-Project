# Milestone 1 - Binary Classification of Sign Language Digits

## Project Summary
This milestone focuses on building a binary image classifier using a custom neural network implemented from scratch with NumPy. The task is to distinguish between two specific hand signs representing digits from the Sign Language Digits dataset. This forms the basis for a deeper understanding of how neural networks function without relying on high-level deep learning frameworks.

## Dataset
The dataset used in this project is the Sign Language Digits Dataset, which contains 5,000 grayscale images of hand gestures. Each image is 28x28 pixels in size and represents a digit between 0 and 9. For this milestone, we focused on classifying between digits 6 and 8, effectively treating the task as binary classification. The dataset was split into 80% for training and 20% for testing.

## Objective
The main goal of this milestone was to build a neural network entirely from scratch using only NumPy. This included manual implementation of data preprocessing, forward and backward propagation, activation and loss functions, and training via gradient descent. The network was designed to accurately classify hand gestures representing the selected digits, while maintaining clean code and clear visualizations of performance.

## Model Architecture
- **Input Layer** - 784 neurons (28x28 pixels, flattened)
- **Hidden Layer** - 64 neurons
- **Output Layer** - 1 neuron (binary classification)
- **Activation Function** - Sigmoid (for hidden and output layers)
- **Loss Function** - Binary Cross-Entropy
- **Training Method** - Mini-batch Gradient Descent with backpropagation

## Training Details
- **Epochs** - 40  
- **Learning Rate** - 0.01  
- **Framework** - Python + NumPy only  
- **Evaluation Metrics** - Accuracy, Precision, Recall, F1-Score, Confusion Matrix  
- **Visualizations** - Loss curve over epochs, confusion matrix

## Results
- **Accuracy** - 95.6%  
- **Precision** - 94.8%  
- **Recall** - 96.2%  
- **F1-Score** - 95.5%  

The model demonstrated stable training behavior and strong performance on the test set, with minimal overfitting.

## Key Challenges Addressed
Several technical challenges were encountered and solved during this milestone. These included reshaping input data to match the required input format for matrix multiplication, implementing numerically stable versions of the loss function (especially log operations), and handling transpositions during forward and backward passes. Additionally, we designed the training loop to support mini-batch gradient descent and ensured clean and interpretable visualizations of the model's performance using loss curves and confusion matrices.

## Colab Notebook
[Open in Google Colab](https://colab.research.google.com/drive/1cuL95B1NBeBOSJPIXAhxGEqqECeJL914?usp=sharing)
