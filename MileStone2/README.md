# Milestone 2 - Multi-Class Classification of Sign Language Digits using PyTorch

## Project Summary
In this milestone, we extended our previous NumPy-based binary classification model by implementing both binary and multiclass classifiers using PyTorch. The task was to classify hand sign digits from the Sign Language Digits dataset, and explore more advanced deep learning tools such as PyTorch's autograd, DataLoader, and optimization capabilities. This work demonstrated how moving from basic implementation to a full-featured framework improves efficiency, scalability, and performance in deep learning projects.

## Dataset
We continued using the Sign Language Digits Dataset, consisting of 5,000 grayscale images sized 28x28 pixels, each representing a hand gesture corresponding to digits from 0 to 9. For the binary classification task, we selected two digits (e.g., 6 and 8), while for the multiclass task, we included all 10 digits. The dataset was normalized, reshaped into 784-dimensional vectors, and loaded into PyTorch using custom Dataset and DataLoader classes.

## Objective
The main goal of this milestone was twofold:  
1. To reimplement our previous binary classification network using PyTorch.  
2. To design and train a multiclass classification network capable of recognizing all digits from 0 to 9.  

This involved building deeper architectures, experimenting with activation and loss functions, managing GPU acceleration, and tuning hyperparameters to improve performance. The entire solution was built using PyTorch from the ground up.

## Model Architectures

- **Binary Classification**
  - **Input Layer** - 784 neurons (flattened 28x28 pixels)
  - **Hidden Layer** - 128 neurons
  - **Output Layer** - 2 neurons (Softmax)
  - **Activation** - ReLU (hidden), Softmax (output)
  - **Loss Function** - Binary Cross-Entropy

- **Multiclass Classification**
  - **Input Layer** - 784 neurons
  - **Hidden Layers** - 128 and 64 neurons
  - **Output Layer** - 10 neurons
  - **Activation** - ReLU (hidden), Softmax (output)
  - **Loss Function** - Categorical Cross-Entropy

## Training Details
- **Optimizer** - Adam / SGD  
- **Learning Rates** - 0.01, 0.001 (experimented)  
- **Batch Sizes** - 32 for binary, 64 for multiclass  
- **Epochs** - 10 to 15 depending on experiment  
- **Features** - Early stopping, learning rate scheduling, dropout, GPU support  
- **Evaluation** - Accuracy, precision, recall, F1-score, confusion matrix  

## Experiments & Results

- **Base Model**
  - Training Accuracy: 75.49%
  - Validation Accuracy: 75.60%
  - F1-Score: 0.75

- **Experiment 1 (Deeper Architecture)**
  - Training Accuracy: 99.4%
  - Validation Accuracy: 98.6%

- **Experiment 2 (Hyperparameter Optimization)**
  - Training Accuracy: 100%
  - Validation Accuracy: 98.93%
  - Final Loss: 0.036
  - Test Set Accuracy (10 unseen images): 100%

## Key Challenges Addressed
Switching from NumPy to PyTorch introduced both flexibility and complexity. We handled:
- Tensor reshaping and normalization for both binary and multiclass modes.
- Manual definition of model architectures using `nn.Linear`, with careful activation and dropout tuning.
- Implementation of custom Dataset classes and batch handling via DataLoader.
- Performance tracking through per-class evaluation metrics and early stopping logic.
- Hyperparameter tuning involving learning rate, batch size, optimizer type, and scheduling strategies.

The multiclass task especially emphasized the value of deeper networks and regularization for generalization, as well as the importance of structured experimentation.

## Example Testing
The final model was evaluated on 10 unseen `.npy` files (stored in the `examples/` directory), each representing a hand gesture. The model achieved 100% accuracy in correctly classifying all samples.

## Colab Notebook
[Open in Google Colab](https://colab.research.google.com/drive/1hpOeV_P6bLdP1T_GeFUPohMhuF9ZIEee)
