# Milestone 2 - Multi-Class Classification of Sign Language Digits using PyTorch

- **Run the notebook** – [Open in Google Colab](https://colab.research.google.com/drive/1hpOeV_P6bLdP1T_GeFUPohMhuF9ZIEee)  
- **View the full report** – [MS2_Report.pdf](./MS2_Report.pdf)

---

## Project Summary
In this milestone, we extended our previous NumPy-based binary classification model by implementing both binary and multiclass classifiers using PyTorch. The task was to classify hand sign digits from the Sign Language Digits dataset and explore more advanced deep learning tools such as PyTorch's `autograd`, `DataLoader`, and optimization utilities.

This milestone demonstrated how transitioning from low-level to high-level frameworks improves **efficiency**, **scalability**, and **experimentation** in deep learning workflows.

---

## Dataset
We used the **Sign Language Digits Dataset**, consisting of **5,000 grayscale 28×28 images**, where each image represents a digit from 0 to 9.  
- For the binary classification task, we focused on digits **6 vs. 8**.  
- For multiclass classification, we included **all 10 digits**.

The dataset was normalized, flattened, and loaded using **custom PyTorch Dataset and DataLoader classes**.

- **Source** – [Sign Language MNIST dataset on Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)

---

## Objective
The main goals were:

1. Re-implement the binary classifier using **PyTorch** (transition from NumPy).
2. Extend the model to handle **multiclass classification** across digits 0–9.

This included experimenting with deeper networks, dropout layers, learning rate scheduling, early stopping, GPU acceleration, and multiple optimizers.

---

## Model Architectures

| Model Type        | Input Layer | Hidden Layers     | Output Layer | Activation (Hidden) | Activation (Output) | Loss Function             |
|:-----------------:|:-----------:|:-----------------:|:------------:|:-------------------:|:--------------------:|:--------------------------:|
| Binary Classifier | 784         | 128               | 2            | ReLU                | Softmax              | Binary Cross-Entropy       |
| Multiclass        | 784         | 128, 64           | 10           | ReLU                | Softmax              | Categorical Cross-Entropy  |

---

## Training Details

| Optimizer    | Learning Rates  | Batch Sizes     | Epochs    | Techniques                               |
|:------------:|:---------------:|:---------------:|:--------:|:----------------------------------------:|
| Adam / SGD   | 0.01, 0.001     | 32 (binary), 64 (multi) | 10–15    | Dropout, Early Stopping, LR Scheduling  |

---

## Experiments & Results

- **Base Model (Shallow)**  
  - Training Accuracy: **75.49%**  
  - Validation Accuracy: **75.60%**  
  - F1 Score: **0.75**

- **Experiment 1 – Deeper Architecture**  
  - Training Accuracy: **99.4%**  
  - Validation Accuracy: **98.6%**

- **Experiment 2 – Hyperparameter Optimization (Best)**  
  - Training Accuracy: **100%**  
  - Validation Accuracy: **98.93%**  
  - Final Loss: **0.036**  
  - **Test Accuracy on 10 unseen .npy files**: **100%**

> 🔎 Full evaluation results, per-class metrics, and confusion matrices are provided in the [full report](./MS2_Report.pdf).

---

## Conclusions

This milestone showcased the **power and flexibility of PyTorch** for rapid experimentation and scalable model design:
- Binary classification was improved and stabilized using ReLU and Softmax activations.
- Moving to **multiclass classification** revealed the benefit of deeper networks and regularization (Dropout).
- PyTorch’s built-in tools like `autograd`, `nn.Module`, `DataLoader`, and GPU support significantly streamlined development.

The final model generalized well and achieved **perfect test performance** on unseen examples — proving the effectiveness of careful tuning, early stopping, and learning rate scheduling.

---
