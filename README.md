# Basics of Deep Learning - Final Project Portfolio

This repository presents a journey through three progressive milestones of the *"Basics of Deep Learning"* course. Each milestone explores different modeling approaches, frameworks, and levels of abstraction - from scratch implementations with NumPy to full-scale CNNs in PyTorch. The final milestone dives into a fine-grained visual classification task using the Stanford Cars dataset.

Each milestone has its own dedicated folder and `README.md` that includes full documentation, architectures, training processes, evaluation results, and conclusion

---

## Milestone 1 - NumPy Neural Network for Binary Classification

We began by building a feedforward neural network **entirely from scratch using NumPy** to classify between two hand gestures (digits 6 and 8) from the **Sign Language Digits Dataset**. The project involved:

- Manual implementation of forward and backward propagation.
- Use of the Sigmoid activation function and binary cross-entropy loss.
- Mini-batch gradient descent and clean performance visualization.

The model reached an impressive **95.6% accuracy** and taught us foundational deep learning concepts from the ground up.

📎 [Read full details here](MileStone1/README.md)

---

## Milestone 2 - Multi-Class Classification in PyTorch

In this stage, we transitioned to **PyTorch**, re-implementing the binary classifier and expanding it into a **multi-class classification model** capable of recognizing all ten digits (0–9). Highlights include:

- Structured training pipeline using DataLoader, GPU acceleration, and autograd.
- Development of deeper architectures with dropout and ReLU.
- Hyperparameter tuning, early stopping, and custom Dataset class.

The multiclass model achieved **98.93% validation accuracy** and **100% test accuracy** on unseen `.npy` examples.

📎 [Read full details here](MileStone2/README.md)

---

## Milestone 3 - Fine-Grained Car Classification & Image Retrieval

The final milestone tackled a **fine-grained classification task** using the **Stanford Cars Dataset** with 196 car categories. We explored three configurations:

### 1. Transfer Learning (ResNet-50 & DenseNet121)
- Fine-tuned pre-trained models on ImageNet.
- Best model: **ResNet-50 fine-tuned with augmentation and dropout**, achieving **73.78% accuracy**.

### 2. Image Retrieval Using Embeddings + KNN
- Extracted feature embeddings from CNNs.
- Performed top-k retrieval with Euclidean and Cosine distance.
- Best setup: **ResNet-50 + KNN (k=10)**, reaching **76.77% accuracy**.

### 3. End-to-End CNN from Scratch
- Custom architecture with 4 conv blocks, batchnorm, and augmentation.
- Final model reached **72.26% accuracy**, demonstrating competitive results even without pre-training.

📎 [Read full details here](MileStone3/README.md)

---

## Reflections and Insights

Through these milestones, we experienced the evolution of deep learning practices:

- **From scratch to frameworks** - We saw how using NumPy teaches the mechanics, while PyTorch boosts flexibility and power.
- **From binary to multiclass and fine-grained tasks** - We advanced from simple digit recognition to subtle distinctions between car models.
- **From classification to retrieval** - We broadened our thinking from prediction to semantic similarity.

This portfolio not only enhanced our technical skills but also strengthened our understanding of **model selection**, **architecture design**, **performance evaluation**, and **deployment considerations**.

---

## Skills & Tools

<!-- START TABLE -->

<div style="display: flex; align-items: flex-start; align: center">
<table align="center">
  <tr> 
    <td align="center" width="96"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/Google_Colab_Logo_2020.svg/1200px-Google_Colab_Logo_2020.svg.png" width="48" height="48" alt="Colab"/><br>Colab</td>
    <td align="center" width="96"><img src="https://upload.wikimedia.org/wikipedia/commons/4/4f/Overleaf_Logo_2020.svg" width="48" height="48" alt="Overleaf"/><br>Overleaf</td>
    <td align="center" width="96"><img src="https://upload.wikimedia.org/wikipedia/commons/d/d9/Kaggle_logo.png" width="48" height="48" alt="Kaggle"/><br>Kaggle</td>
    <td align="center" width="96"><img src="https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov5-logo.png" width="48" height="48" alt="YOLO"/><br>YOLO Model</td>
    <td align="center" width="96"><img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" width="48" height="48" alt="Python"/><br>Python</td>
  </tr>
  <tr>
    <td align="center" width="96"><img src="https://numpy.org/images/logo.svg" width="48" height="48" alt="NumPy"/><br>NumPy</td>
    <td align="center" width="96"><img src="https://pandas.pydata.org/static/img/pandas_logo.svg" width="48" height="48" alt="Pandas"/><br>Pandas</td>
    <td align="center" width="96"><img src="https://upload.wikimedia.org/wikipedia/commons/4/46/Matplotlib_logo.svg" width="48" height="48" alt="Matplotlib"/><br>Matplotlib</td>
    <td align="center" width="96"><img src="https://upload.wikimedia.org/wikipedia/commons/8/8c/PyTorch_logo.png" width="48" height="48" alt="PyTorch"/><br>PyTorch</td>
    <td align="center" width="96"><img src="https://upload.wikimedia.org/wikipedia/commons/e/ec/Scikit-learn_logo.svg" width="48" height="48" alt="Scikit-learn"/><br>sklearn</td>
  </tr>
</table>
</div>

<!-- END TABLE -->

---

## Project Authors

- [Adir Edri](https://github.com/adiredri)
- [Ofir Almog](https://github.com/Ofigu)

