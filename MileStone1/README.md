# Milestone 1 - Binary Classification of Sign Language Digits

- **Run the notebook** - [Open in Google Colab](https://colab.research.google.com/drive/1cuL95B1NBeBOSJPIXAhxGEqqECeJL914?usp=sharing)  
- **View the full report** - [MS1_Report.pdf](./MS1_Report.pdf)

---

## Project Summary

This milestone focuses on building a binary image classifier using a custom neural network implemented from scratch with NumPy. The task is to distinguish between two specific hand signs representing digits from the Sign Language Digits dataset. This forms the basis for a deeper understanding of how neural networks function without relying on high-level deep learning frameworks.

---

## Dataset

The dataset used in this project is the **Sign Language Digits Dataset**, which contains 5,000 grayscale images of hand gestures. Each image is 28×28 pixels in size and represents a digit between 0 and 9.  
For this milestone, we focused on classifying between digits **6 and 8**, effectively treating the task as binary classification. The dataset was split into 80% for training and 20% for testing.

- **Source** – [Sign Language MNIST dataset on Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)

---

## Objective

The main goal of this milestone was to build a neural network entirely from scratch using only NumPy. This included manual implementation of data preprocessing, forward and backward propagation, activation and loss functions, and training via gradient descent. The network was designed to accurately classify hand gestures representing the selected digits, while maintaining clean code and clear visualizations of performance.

---

## Model Architecture

| **Input Layer** | **Hidden Layer** | **Output Layer** | **Activation**            | **Loss Function**      |
|:---------------:|:----------------:|:----------------:|:-------------------------:|:----------------------:|
| 784 neurons     | 64 neurons       | 1 neuron         | Sigmoid (hidden & output) | Binary Cross-Entropy   |

The model was trained for **40 epochs** using a **learning rate of 0.01**, and optimized via **mini-batch gradient descent with backpropagation**.

---

## Results

| **Accuracy** | **Precision** | **Recall** | **F1-Score** |
|:------------:|:-------------:|:----------:|:------------:|
| 95.6%        | 94.8%         | 96.2%      | 95.5%        |

---

## Conclusions

This project served as an excellent introduction to deep learning fundamentals:

- We successfully implemented a full neural network pipeline using only NumPy, gaining a deeper understanding of forward/backward propagation, weight updates, and activation behaviors.
- The model showed strong generalization on test data, with minimal overfitting and performance exceeding 95% accuracy.
- Key technical challenges—such as matrix transpositions, numerical stability in log-based loss functions, and correct batching—were resolved through rigorous debugging and visual analysis.
- Visualization tools (loss curves, confusion matrix) played an important role in performance validation.

This milestone laid the groundwork for all future experiments in the course by giving us low-level insight into how neural networks truly operate.
