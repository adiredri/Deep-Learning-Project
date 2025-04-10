# Milestone 2 – Multi-Class Classification of Sign Language Digits using PyTorch

- **Run the notebook**: [Open in Google Colab](https://colab.research.google.com/drive/1hpOeV_P6bLdP1T_GeFUPohMhuF9ZIEee)  
- **View the full report**: [MS2_Report.pdf](./MS2_Report.pdf)

---

## Project Summary

This milestone extended our previous work (Milestone 1) by implementing both binary and multiclass classifiers using **PyTorch**. The task remained the same — recognizing digits from hand sign images — but the approach shifted to a full-featured deep learning framework.

We built custom fully connected neural networks (FCNs) and leveraged PyTorch’s tools (DataLoader, autograd, CUDA acceleration, etc.) to explore how architectural depth and training optimizations impact model performance.

---

## Dataset

We continued using the **Sign Language Digits Dataset**, consisting of 5,000 grayscale 28×28 images representing hand gestures for digits 0–9.

- For binary classification: focused on digits **6 and 8**.
- For multiclass classification: used **all 10 digits (0–9)**.

- **Source**: [Sign Language MNIST on Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)

---

## Objective

1. Recreate the binary classification task using PyTorch  
2. Extend to multiclass classification for all digits  
3. Build a scalable, flexible training pipeline  
4. Tune architecture and hyperparameters for best generalization

---

## Model Architectures

| **Task**              | **Hidden Layers**    | **Output Layer**     | **Activation**                      | **Loss Function**           |
|:---------------------:|:--------------------:|:--------------------:|:-----------------------------------:|:---------------------------:|
| Binary Classification | 1 layer (128 units)  | 2 neurons (Softmax)  | ReLU (hidden), Softmax (output)     | Binary Cross-Entropy        |
| Multiclass Classification | 2 layers (128, 64)   | 10 neurons (Softmax) | ReLU (hidden), Softmax (output)     | Categorical Cross-Entropy   |

---

## Training Details

| **Optimizer** | **Learning Rate** | **Batch Size**       | **Epochs**      | **Features**                                        |
|:-------------:|:-----------------:|:--------------------:|:---------------:|:--------------------------------------------------:|
| Adam / SGD    | 0.001 / 0.01      | 32 (binary), 64 (multi) | 10–15           | Dropout, Early Stopping, LR Scheduling, GPU Accel. |

---

## Results & Experiments

| **Experiment**         | **Training Accuracy** | **Validation Accuracy** | **Test Accuracy** | **Loss**    | **F1 Score** |
|:----------------------:|:---------------------:|:------------------------:|:-----------------:|:-----------:|:------------:|
| Base Model             | 75.49%                | 75.60%                   | -                 | -           | 0.75         |
| Experiment 1 (Deeper)  | 99.4%                 | 98.6%                    | -                 | -           | -            |
| Experiment 2 (Tuned)   | 100%                  | 98.93%                   | **100%** (10 samples) | 0.036     | 0.99         |

The best model (Experiment 2) achieved **perfect test accuracy** on unseen `.npy` samples and excellent generalization on the validation set.

---

## Conclusions

This milestone highlighted the transition from scratch-built models to framework-based deep learning:

- **Model depth matters** – deeper architectures captured complex digit patterns better.
- **PyTorch utilities** like autograd and DataLoader greatly simplified training and data handling.
- **Hyperparameter tuning** (learning rate, batch size, dropout, scheduling) made a decisive difference in generalization and stability.
- **Multiclass tasks** require more regularization, but benefit significantly from dropout and scheduler-based learning.

Experimentation paid off — and we learned how to build real-world models with scalable, reusable code pipelines.
