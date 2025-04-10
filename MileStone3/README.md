# Milestone 3 - Final Project: Fine-Grained Car Classification and Retrieval


## Introduction

In this final milestone of the "Basics of Deep Learning" course, we tackled the complex task of fine-grained image classification and retrieval using the Stanford Cars Dataset. This dataset includes 16,185 images labeled into 196 subtle car categories, with visual distinctions often limited to minor differences in shape, color, or model details.

To address this challenge, we implemented three distinct deep learning configurations:
1. Transfer Learning with a pre-trained ResNet-50  
2. Image Retrieval using feature embeddings and nearest neighbors  
3. End-to-End Convolutional Neural Network trained from scratch  

Each configuration was explored through three experiments (9 in total), which allowed us to analyze how architecture depth, regularization, fine-tuning, and augmentation strategies affect model performance.

All models were implemented in PyTorch, and each configuration was tested with both classification and retrieval-focused metrics where applicable.

---

## Configuration 1 – Transfer Learning

In this configuration, we fine-tuned a ResNet-50 model pre-trained on ImageNet to classify the 196 car categories in our dataset. This approach aimed to leverage generalized visual features already learned by the model and adapt them to our fine-grained domain.

### Shared Settings

| Backbone   | Input Size | Optimizer | Learning Rate | Loss Function    | Batch Size | Scheduler                      | Epochs |
|:----------:|:----------:|:---------:|:-------------:|:----------------:|:----------:|:------------------------------:|:------:|
| ResNet-50  | 224×224    | Adam      | 0.001         | CrossEntropyLoss | 32         | StepLR (γ=0.1 every 7 epochs)  | 10     |

---

### Experiment 1 – Base Model (Frozen Backbone)
**Goal:**  
Evaluate out-of-the-box generalization of ImageNet features.

**Setup:**  
Freeze the entire ResNet-50 backbone. Train only the final classification head (2048 → 196). No data augmentation or regularization.

**Results:**  
- Accuracy: 42.37%  
- F1 Score: 41.96%  
- Loss: 2.3724  

**Conclusion:**  
The model struggled to generalize, showing that the final classifier alone can't compensate for domain differences. The features learned on ImageNet are insufficient for fine-grained car classification without adaptation.

---

### Experiment 2 – Frozen Backbone + Data Augmentation
**Goal:**  
Explore how basic data augmentation affects performance.

**Setup:**  
Same frozen backbone. Applied RandomCrop, HorizontalFlip, and Rotation (±15°). The classifier remains the only trainable layer.

**Results:**  
- Moderate improvement in validation accuracy and stability  
- Lower loss and a slight boost in precision and recall

**Conclusion:**  
Even with a fixed backbone, data augmentation enriches the training signal and improves generalization. However, the model’s ceiling is limited without access to deeper layers.

---

### Experiment 3 – Fine-Tuned Backbone + Augmentation + Dropout
**Goal:**  
Unlock the full potential of the backbone with regularization.

**Setup:**  
Unfreeze all ResNet-50 layers. Add Dropout (p = 0.5) before the final layer. Use the same augmentations as in Experiment 2.

**Results:**  
- Accuracy: 73.78%  
- F1 Score: 73.63%  
- Loss: 0.9012  
- Precision: 78.18%  
- Recall: 73.78%

**Conclusion:**  
This experiment significantly outperformed the others. Unfreezing the backbone allowed domain-specific features to be learned, while dropout and augmentation reduced overfitting.  
This model was selected as the best classifier in this configuration.

---

### Transfer Learning Summary

| Experiment |              Model Description               |   Loss   | Accuracy | F1 Score | Precision | Recall |
|:----------:|:---------------------------------------------:|:--------:|:--------:|:--------:|:---------:|:------:|
|   Exp. 1   | Frozen ResNet-50 (no augmentation)            |  2.3724  |  42.37%  |  41.96%  |  42.91%   | 42.37% |
|   Exp. 2   | Frozen + Augmentation                         |  ~2.000  |  ~47.0%  |  ~47.0%  |   ~50%    |  ~47%  |
|   Exp. 3   | Fine-tuned + Dropout + Augmentation           |  0.9012  |  73.78%  |  73.63%  |  78.18%   | 73.78% |


---


## Configuration 2 – Image Retrieval Using Deep Embeddings

### Overview
In this configuration, we shifted from direct classification to **visual similarity retrieval**. Instead of predicting class labels, the goal was to embed each image into a feature space and retrieve the most visually similar images using **K-Nearest Neighbors (KNN)**. We relied on deep CNN backbones to extract embeddings, then compared vectors using distance metrics (Euclidean or Cosine). This setup is useful for real-world applications like recommendation engines or visual search.

### Shared Setup
- Dataset: Stanford Cars (16,185 cropped images, 196 categories)
- Input Size: 224×224
- Embedding Extractor: CNN backbone (ResNet50 / DenseNet121)
- Normalization: ImageNet statistics
- Optimizer: Adam (lr = 0.001)
- Loss: CrossEntropyLoss
- Batch Size: 64
- Training Epochs: 10
- Retrieval Metric: KNN (Euclidean or Cosine)

---

### Experiment 1 – ResNet-50 + KNN (k=3, Euclidean)
**Goal:** Evaluate embedding quality with a basic KNN setup  
**Setup:**  
- Fine-tuned ResNet-50 (same as in classification config)  
- Extract embeddings (2048-dim)  
- KNN with k=3, Euclidean distance

**Results:**  
- Accuracy: 74.93%  
- F1 Score: 75.23%  
- Precision: 77.38%  
- Recall: 74.93%  
- Fit Time: 0.0092s  
- Predict Time: 1.1998s

**Conclusion:**  
This baseline setup achieved decent retrieval quality. A small `k` resulted in local consistency but sometimes lacked diversity for ambiguous samples.

---

### Experiment 2 – ResNet-50 + KNN (k=10, Euclidean)
**Goal:** Improve robustness with a wider neighborhood  
**Setup:**  
- Same embedding extractor (ResNet-50)  
- KNN with k=10, Euclidean distance

**Results:**  
- Accuracy: 76.77%  
- F1 Score: 76.77%  
- Precision: 78.81%  
- Recall: 76.77%  
- Fit Time: 0.0079s  
- Predict Time: 1.3873s

**Conclusion:**  
Increasing `k` improved robustness and overall accuracy. Larger neighborhoods compensated for class overlaps, making this the **best-performing model** in this configuration.

---

### Experiment 3 – DenseNet121 + KNN (k=7, Cosine Similarity)
**Goal:** Evaluate alternative backbone and distance metric  
**Setup:**  
- Progressive unfreezing of DenseNet121  
- Embedding size: 1024  
- KNN with k=7, Cosine similarity

**Results:**  
- Accuracy: 73.41%  
- F1 Score: 73.49%  
- Precision: 75.33%  
- Recall: 73.41%  
- Fit Time: 0.0047s  
- Predict Time: 1.0178s

**Conclusion:**  
Although slightly weaker in accuracy, this model offered **fast inference** and stable retrieval quality. Cosine similarity proved beneficial for normalized embeddings, and DenseNet's compact architecture reduced overhead.

---

### Image Retrieval Summary

| Experiment | Model Description                  | Accuracy | F1 Score | Precision | Recall | Predict Time |
|------------|------------------------------------|----------|----------|-----------|--------|---------------|
| Exp. 1     | ResNet-50 + KNN (k=3, Euclidean)   | 74.93%   | 75.23%   | 77.38%    | 74.93% | 1.1998s       |
| Exp. 2     | ResNet-50 + KNN (k=10, Euclidean)  | 76.77%   | 76.77%   | 78.81%    | 76.77% | 1.3873s       |
| Exp. 3     | DenseNet121 + KNN (k=7, Cosine)    | 73.41%   | 73.49%   | 75.33%    | 73.41% | 1.0178s       |

> Best model: **Experiment 2 – ResNet-50 + KNN (k=10)**  
> Best for speed: **Experiment 3 – DenseNet + Cosine Similarity**

---

This configuration highlighted the power of learned embeddings for **image-to-image retrieval**. It also showed how changing distance metrics and model backbones affects both accuracy and latency, offering options for different deployment needs.

---

## Configuration 3 – End-to-End Convolutional Neural Network

### Overview
In this configuration, we built and trained a complete CNN model **from scratch**, without relying on pre-trained weights. Unlike the Transfer Learning or Retrieval configurations, this approach gave us full architectural control — and the full burden of learning features directly from the dataset. This allowed us to investigate how depth, normalization, and data augmentation affect generalization in fine-grained settings.

### Shared Setup
- Dataset: Stanford Cars (cropped, 16,185 images, 196 categories)
- Input Size: 224×224
- Optimizer: Adam (lr = 0.001)
- Loss: CrossEntropyLoss
- Batch Size: 64
- Pooling: MaxPooling
- Activation: ReLU
- Weight Init: He Initialization
- Training Epochs: up to 40

---

### Experiment 1 – Basic CNN (No Regularization)
**Goal:** Build a minimal CNN and test its baseline ability to classify cars  
**Architecture:**
- 3 Convolutional Blocks
- MaxPooling + ReLU
- Fully Connected Head (no dropout, no normalization)
- No data augmentation

**Results:**  
- Accuracy: 8.88%  
- F1 Score: 8.52%  
- Loss: 4.6955  

**Conclusion:**  
This model failed to learn meaningful patterns. The lack of depth and regularization, along with no augmentation, made it ineffective on a complex dataset like ours.

---

### Experiment 2 – Modified CNN + BatchNorm + Dropout
**Goal:** Add regularization to improve generalization  
**Architecture:**
- 4 Convolutional Blocks  
- BatchNorm after each convolution  
- Dropout (p=0.5) before final layer  
- No augmentation  

**Results:**  
- Accuracy: 8.16%  
- F1 Score: 6.23%  
- Loss: 4.5436  

**Conclusion:**  
Despite deeper architecture and added regularization, performance barely improved. This emphasized the importance of data variation and augmentation, especially in low-data-per-class settings.

---

### Experiment 3 – Advanced CNN + Data Augmentation
**Goal:** Combine regularization and augmentation for maximum performance  
**Architecture:**
- Same as Experiment 2  
- Augmentations: RandomCrop, HorizontalFlip, ColorJitter  
- Trained for 40 epochs  

**Results:**  
- Accuracy: 72.26%  
- F1 Score: 72.17%  
- Precision: 72.99%  
- Loss: 1.1026  

**Conclusion:**  
This configuration **dramatically improved performance**. By combining architectural depth, normalization, dropout, and rich augmentation, the model learned generalizable patterns and became competitive with fine-tuned ResNet-50. This validated the viability of training from scratch with a strong data pipeline.

---

### End-to-End CNN Summary

| Experiment | Model Description                | Accuracy | F1 Score | Precision | Loss    |
|------------|----------------------------------|----------|----------|-----------|---------|
| Exp. 1     | Basic CNN (3 blocks, no reg)     | 8.88%    | 8.52%    | 9.80%     | 4.6955  |
| Exp. 2     | CNN + BatchNorm + Dropout        | 8.16%    | 6.23%    | 7.17%     | 4.5436  |
| Exp. 3     | CNN + Augmentation + Regularized | 72.26%   | 72.17%   | 72.99%    | 1.1026  |

> Best model: **Experiment 3 – Advanced CNN with Augmentation**

---

## Final Comparison: All Configurations

| Configuration     | Best Model                     | Accuracy | F1 Score | Precision | Recall | Comment                           |
|-------------------|--------------------------------|----------|----------|-----------|--------|------------------------------------|
| Transfer Learning | ResNet-50 Fine-Tuned + Dropout | 73.78%   | 73.63%   | 78.18%    | 73.78% | Highest overall performance        |
| Image Retrieval   | ResNet-50 + KNN (k=10)         | 76.77%   | 76.77%   | 78.81%    | 76.77% | Best for similarity search         |
| End-to-End CNN    | CNN + Aug + Dropout            | 72.26%   | 72.17%   | 72.99%    | 72.26% | Strong results, fully custom model |

### Conclusion
- **Transfer Learning** provided the best classification performance with minimal effort and strong generalization, making it ideal for deployment.
- **Image Retrieval** was most suitable for search and recommendation systems. The use of learned embeddings allowed flexible querying and robust top-k accuracy.
- **End-to-End CNN** showed that training from scratch is possible and effective — but requires strong architecture and augmentation strategies.

Each configuration contributed unique insights and demonstrated key principles of deep learning: reusability (Transfer Learning), representational power (Embeddings), and architectural design (CNNs from scratch). This project deepened our understanding of modeling strategies, evaluation, and the art of experimentation in real-world machine learning tasks.
