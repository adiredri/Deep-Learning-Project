# Milestone 3 - Final Project: Fine-Grained Car Classification and Retrieval

## Introduction

In this final milestone of the "Basics of Deep Learning" course, we tackled the complex task of fine-grained image classification and retrieval using the Stanford Cars Dataset. This dataset includes 16,185 images labeled into 196 subtle car categories, with visual distinctions often limited to minor differences in shape, color, or model details.

To address this challenge, we implemented three distinct deep learning configurations:

1. **Transfer Learning with a pre-trained ResNet-50** - _[Link to Notebook](https://colab.research.google.com/drive/1pQqWT0t_fVY0rUHVP46eDusuXtT_uqXP)_

2. **Image Retrieval using feature embeddings and nearest neighbors** - _[Link to Notebook](https://colab.research.google.com/drive/1udo_D-PzcosCcCV9K5XtqfF2SUViLc8_)_

3. **End-to-End Convolutional Neural Network trained from scratch** - _[Link to Notebook](https://colab.research.google.com/drive/1kSFLQNswStkj4WQiSAg7eEbKwJ1kSzV3)_

Each configuration was explored through three experiments, allowing us to evaluate the impact of architectural depth, regularization, and data augmentation strategies. All implementations were done in PyTorch, with classification and retrieval tasks evaluated using standard metrics.

---

## Configuration 1 - Transfer Learning with ResNet-50 and DenseNet121


We fine-tuned pre-trained models on ImageNet to classify the 196 car categories. This configuration tested the power of transfer learning in adapting general-purpose features to a fine-grained classification task. We explored both shallow (frozen) and deep (fine-tuned/progressive) training strategies.

### Shared Settings

| Backbone     | Input Size | Optimizer | Learning Rate | Loss Function    | Batch Size | Scheduler                      | Epochs |
|:------------:|:----------:|:---------:|:-------------:|:----------------:|:----------:|:------------------------------:|:------:|
| ResNet-50 / DenseNet121 | 224×224    | Adam      | 0.001         | CrossEntropyLoss | 32         | StepLR (γ=0.1 every 7 epochs)  | 10     |

---

### Experiments

- **Experiment 1 - ResNet-50 (Frozen)** – Only the classification head was trained. The backbone was frozen, and no augmentation or regularization was used.

- **Experiment 2 - ResNet-50 (Fine-Tuned + Aug + Dropout)** – All layers were unfrozen. Data augmentation (RandomCrop, Flip, Rotation) and Dropout (p=0.5) were used.

- **Experiment 3 - DenseNet121 (Progressive Unfreezing)** – Used DenseNet121 with gradual unfreezing and same augmentations. Aimed to explore a more compact yet powerful architecture.

---

### Transfer Learning Summary

| Experiment |             Model Description              |   Loss   | Accuracy | F1 Score | Precision | Recall |
|:----------:|:------------------------------------------:|:--------:|:--------:|:--------:|:---------:|:------:|
|   Exp. 1   | ResNet-50 – Frozen                         |  2.3724  |  42.37%  |  41.96%  |  42.91%   | 42.37% |
|   Exp. 2   | ResNet-50 – Fine-Tuned + Aug + Dropout     |  0.9012  |  73.78%  |  73.63%  |  78.18%   | 73.78% |
|   Exp. 3   | DenseNet121 – Progressive Unfreezing       |  1.0800  |  71.69%  |  71.70%  |  75.43%   | 71.69% |

---

### Conclusions - Transfer Learning

This configuration confirmed the strength of transfer learning for fine-grained tasks:
- The frozen backbone alone produced weak results.
- Fine-tuning with regularization significantly improved performance.
- DenseNet121 performed well, though slightly behind the fine-tuned ResNet-50.

The best model in this configuration was **Experiment 2 – ResNet-50 Fine-Tuned**, which reached the highest accuracy and F1 Score. It was later used as the baseline for deployment.

---

## Configuration 2 - Image Retrieval Using Deep Embeddings

This configuration focused on visual similarity retrieval instead of direct classification. We trained CNN backbones (ResNet-50, DenseNet121) to generate deep embeddings, then used K-Nearest Neighbors (KNN) over these features to find the most visually similar images. The setup is useful for search engines, recommendation systems, and tasks requiring semantic similarity.

### Shared Settings

| Input Size | Backbone                | Optimizer | Learning Rate | Loss Function    | Batch Size | Epochs |
|:----------:|:------------------------:|:---------:|:-------------:|:----------------:|:----------:|:------:|
| 224×224    | ResNet-50 / DenseNet121 | Adam      | 0.001         | CrossEntropyLoss | 64         | 10     |

---

### Experiments

- **Experiment 1 - ResNet-50 + KNN (k = 3, Euclidean)** - Embeddings were extracted from a fine-tuned ResNet-50 and used with KNN (k = 3, Euclidean).

- **Experiment 2 - ResNet-50 + KNN (k = 10, Euclidean)** - Same ResNet-50 embeddings, but KNN was configured with a larger neighborhood (k = 10) to improve robustness.

- **Experiment 3 - DenseNet121 + KNN (k = 7, Cosine Similarity)** - Switched to DenseNet121 as backbone, with embeddings of size 1024 and cosine similarity instead of Euclidean distance.

---

### Image Retrieval Summary

| Experiment |             Model Description              | Accuracy | F1 Score | Precision | Recall | Predict Time |
|:----------:|:------------------------------------------:|:--------:|:--------:|:---------:|:------:|:------------:|
|   Exp. 1   | ResNet-50 + KNN (k = 3, Euclidean)          | 74.93%   | 75.23%   | 77.38%    | 74.93% |   1.1998s    |
|   Exp. 2   | ResNet-50 + KNN (k = 10, Euclidean)         | 76.77%   | 76.77%   | 78.81%    | 76.77% |   1.3873s    |
|   Exp. 3   | DenseNet121 + KNN (k = 7, Cosine Similarity) | 73.41%   | 73.49%   | 75.33%    | 73.41% |   1.0178s    |

---

### Conclusions - Image Retrieval

This configuration highlighted the potential of deep feature embeddings for image-to-image search:
- Even small neighborhoods (k = 3) yielded good results, but increasing k improved stability.
- Using cosine similarity with DenseNet provided competitive performance and the fastest prediction time.
- The retrieval pipeline is highly flexible, making it well-suited for applications requiring fast, interpretable, and semantic similarity.

The best model in this configuration was **Experiment 2**, which used ResNet-50 with KNN (k = 10, Euclidean) and achieved the highest accuracy and F1 score.

---

## Configuration 3 - End-to-End Convolutional Neural Network

In this configuration, we built and trained a CNN model entirely from scratch, with no pre-trained weights. This allowed full control over the architecture and learning process. The goal was to evaluate the viability of training a custom network on a complex fine-grained dataset like Stanford Cars.

### Shared Settings

| Input Size | Optimizer | Learning Rate | Loss Function    | Batch Size | Pooling   | Activation | Weight Init        | Max Epochs |
|:----------:|:---------:|:-------------:|:----------------:|:----------:|:---------:|:----------:|:------------------:|:----------:|
| 224×224    | Adam      | 0.001         | CrossEntropyLoss | 64         | MaxPooling | ReLU       | He Initialization  | 40         |

---

### Experiments

- **Experiment 1 - Basic CNN (No Regularization)** - A minimal CNN with 3 convolutional blocks, no normalization, and no data augmentation.

- **Experiment 2 - CNN + BatchNorm + Dropout** - Expanded to 4 convolutional blocks, added Batch Normalization and Dropout, still without augmentation.

- **Experiment 3 - Advanced CNN + Data Augmentation** - Same as Experiment 2, with added RandomCrop, HorizontalFlip, and ColorJitter augmentations. Trained for 40 epochs.

---

### End-to-End CNN Summary

| Experiment |            Model Description             | Accuracy | F1 Score | Precision | Recall |   Loss   |
|:----------:|:----------------------------------------:|:--------:|:--------:|:---------:|:------:|:--------:|
|   Exp. 1   | Basic CNN (3 blocks, no regularization)  |  8.88%   |  8.52%   |   9.80%   |  8.88% |  4.6955  |
|   Exp. 2   | CNN + BatchNorm + Dropout                |  8.16%   |  6.23%   |   7.17%   |  8.16% |  4.5436  |
|   Exp. 3   | CNN + Augmentation + Regularization      |  72.26%  |  72.17%  |  72.99%   | 72.26% |  1.1026  |

---

### Conclusions – End-to-End CNN

This configuration showed a steep learning curve but promising results:
- Basic models without augmentation or normalization failed to generalize.
- Adding BatchNorm and Dropout without augmentation was not enough.
- The full combination of regularization, deeper architecture, and data augmentation led to competitive results.

The best model in this configuration was **Experiment 3**, which demonstrated that with the right setup, custom CNNs can achieve results comparable to pre-trained solutions.

---

## Final Conclusions

This milestone provided a comprehensive comparison between three distinct approaches to solving a fine-grained visual classification task.

- **Transfer Learning** offered the best accuracy for classification with the least engineering effort. It leveraged powerful pre-trained features and benefited significantly from fine-tuning and dropout. This approach is ideal for fast deployment and works especially well when annotated data is limited but domain similarity is high.

- **Image Retrieval** proved effective for semantic similarity tasks. It outperformed classification in raw accuracy when evaluated using KNN over learned embeddings. This approach is especially useful for search interfaces and user-facing systems where interpretability of similarity matters.

- **End-to-End CNN** required more careful design and training but ultimately proved that custom networks can perform competitively when equipped with solid regularization and data augmentation. While resource-intensive, it gives full control over the learning process and is valuable for research and experimentation.

In summary, each configuration serves different practical needs:
- Use **Transfer Learning** for classification pipelines requiring strong accuracy with minimal compute.
- Use **Image Retrieval** for interactive tools or similarity-based search.
- Use **End-to-End CNN** when full customization or architectural research is needed.

This project deepened our understanding of deep learning design, training stability, and performance evaluation under different constraints - from off-the-shelf reuse to scratch-built solutions.
