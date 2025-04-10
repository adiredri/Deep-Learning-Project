# Milestone 1 - Binary Classification of Sign Language Digits

**Run the notebook**: [Open in Google Colab](https://colab.research.google.com/drive/1cuL95B1NBeBOSJPIXAhxGEqqECeJL914?usp=sharing)  
**View the full report**: [MS1_Report.pdf](./MS1_Report.pdf)

## Project Summary
This milestone focuses on building a binary image classifier using a custom neural network implemented from scratch with NumPy. The task is to distinguish between two specific hand signs representing digits from the Sign Language Digits dataset. This forms the basis for a deeper understanding of how neural networks function without relying on high-level deep learning frameworks.

## Dataset
The dataset used in this project is the Sign Language Digits Dataset, which contains 5,000 grayscale images of hand gestures. Each image is 28x28 pixels in size and represents a digit between 0 and 9. For this milestone, we focused on classifying between digits 6 and 8, effectively treating the task as binary classification. The dataset was split into 80% for training and 20% for testing.
- **Source**: Downloaded via [Google Drive link](https://drive.google.com/file/d/1-0fhqH8tXKPb60C_b4aUHT7f-J4O6Ezq), derived from the [Sign Language MNIST dataset on Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)

## Objective
The main goal of this milestone was to build a neural network entirely from scratch using only NumPy. This included manual implementation of data preprocessing, forward and backward propagation, activation and loss functions, and training via gradient descent. The network was designed to accurately classify hand gestures representing the selected digits, while maintaining clean code and clear visualizations of performance.

## Model Architecture

| **Input Layer** | **Hidden Layer**     | **Output Layer** | **Activation**              | **Loss Function**         | **Training Method**                       |
|-----------------|----------------------|------------------|-----------------------------|---------------------------|-------------------------------------------|
| 784 neurons     | 64 neurons           | 1 neuron         | Sigmoid (hidden & output)   | Binary Cross-Entropy      | Mini-batch Gradient Descent + Backprop    |

## Training Details

| **Epochs** | **Learning Rate** | **Framework**       | **Metrics**                                      | **Visualizations**                      |
|------------|-------------------|---------------------|--------------------------------------------------|------------------------------------------|
| 40         | 0.01              | Python + NumPy only | Accuracy, Precision, Recall, F1, Confusion Matrix | Loss curve, Confusion Matrix Heatmap     |

## Results

| **Accuracy** | **Precision** | **Recall** | **F1-Score** |
|--------------|---------------|------------|--------------|
| 95.6%        | 94.8%         | 96.2%      | 95.5%        |

The model demonstrated stable training behavior and strong performance on the test set, with minimal overfitting.

## Key Challenges Addressed
Several technical challenges were encountered and solved during this milestone. These included reshaping input data to match the required input format for matrix multiplication, implementing numerically stable versions of the loss function (especially log operations), and handling transpositions during forward and backward passes. Additionally, we designed the training loop to support mini-batch gradient descent and ensured clean and interpretable visualizations of the model's performance using loss curves and confusion matrices.
