# Few-Shot Classification on Agricultural Data using Prototypical Networks

**Authors**: Bardia Akbari - Navid Naserazad

## Abstract

The accurate and timely identification of plant diseases is crucial for maintaining crop yield and ensuring food security. While deep learning has shown immense promise in this domain, its effectiveness is often hampered by the scarcity of labeled data for many diseases. Few-Shot Learning (FSL) offers a compelling solution by enabling models to generalize from a very limited number of examples. This report details a project aimed at applying FSL techniques to the PlantVillage dataset for plant disease classification. After an initial exploration of complex FSL models, we pivoted to a Prototypical Network, a robust and effective metric-based approach. We systematically evaluated this model with three distinct feature extraction backbones: a pre-trained Vision Transformer (ViT), a pre-trained ResNet50, and a custom four-layer convolutional network (Conv-4) trained from scratch. Our experiments, conducted in 5-way 1-shot and 5-way 5-shot settings, demonstrate that the pre-trained ViT backbone achieves superior performance, attaining a mean accuracy of 87.83% in the 1-shot scenario and 95.50% in the 5-shot scenario, highlighting the power of transfer learning and modern architectures in the FSL domain.

## 1. Introduction

Precision agriculture relies heavily on the automated detection of plant diseases to facilitate early intervention and minimize economic losses. Conventional deep learning models, particularly Convolutional Neural Networks (CNNs), have excelled at image classification tasks but typically require vast amounts of labeled data for training. In agriculture, collecting and annotating such large datasets for every possible plant disease is often impractical, as some diseases are rare or emerge unexpectedly.

This data bottleneck motivates the exploration of Few-Shot Learning (FSL), a paradigm that trains models to learn new classes from only a handful of examples. Our project initially aimed to implement the model proposed in "Few-Shot Classification with Feature Map Reconstruction Networks," a promising but complex approach. However, due to unforeseen implementation challenges and time constraints, we adopted a more direct and well-established FSL method: the Prototypical Network.

The primary objective of this project shifted to implementing and rigorously evaluating a Prototypical Network for classifying plant diseases from the PlantVillage dataset. To understand the impact of the feature extractor on performance, we tested three different backbone architectures:

- **Vision Transformer (ViT)**: A state-of-the-art, pre-trained transformer model.
- **ResNet50**: A widely-used, pre-trained deep residual network.
- **Conv-4**: A standard shallow CNN, trained from scratch.

This report presents our methodology, experimental setup, results, and a detailed discussion of the findings, ultimately providing a clear comparison of these architectures within a few-shot learning context.

## 2. Background and Methodology

### 2.1. Few-Shot Learning and Prototypical Networks

Few-Shot Learning aims to solve a classification task where only a few labeled examples (the support set) are available for each new class. The model's ability to generalize is then tested on unseen examples (the query set). This is often framed as an "N-way K-shot" problem, where N is the number of classes and K is the number of support examples per class.

Our approach is based on **Prototypical Networks**. This is a metric learning method that operates on a simple but powerful principle: there exists an embedding space where images from the same class cluster together. The algorithm proceeds as follows:

1. **Embedding**: A neural network backbone, or encoder, maps all input images (both support and query) into a high-dimensional feature space.
2. **Prototype Calculation**: For each of the N classes, a single prototype vector is computed by taking the element-wise mean of the embeddings of its K support images. This prototype represents a central point for that class in the embedding space.
3. **Classification**: A query image is classified by calculating its distance (e.g., Euclidean distance) to each of the N class prototypes. The query image is assigned the label of the class corresponding to the nearest prototype.

The network is trained episodically, where each episode simulates a few-shot task, and the loss function (Negative Log-Likelihood) is optimized to minimize the distance between query embeddings and their correct class prototypes while maximizing the distance to incorrect ones.

### 2.2. Backbone Architectures

The effectiveness of a Prototypical Network is highly dependent on the quality of its encoder. We experimented with three diverse backbones:

- **ViT (Vision Transformer)**: Based on the transformer architecture, ViT processes images by splitting them into a sequence of fixed-size patches. We used the `vit_small_patch16_224` model, pre-trained on ImageNet. As implemented in our `ViT.py` script, the final representation for each image is extracted from the output corresponding to the special `[CLS]` token.
- **ResNet50**: A 50-layer deep CNN that utilizes residual connections to enable effective training of very deep networks. We used a standard ResNet50 model pre-trained on ImageNet, as shown in `resnet50.py`, which has proven to be a strong baseline for many computer vision tasks.
- **Conv-4 (Custom CNN)**: To establish a baseline without the benefits of pre-training, we implemented a classic four-layer CNN, often referred to as Conv-4 in FSL literature. As defined in `CNN.py`, our implementation consists of four sequential blocks, each containing a 3x3 convolution, batch normalization, a ReLU activation, and max-pooling. This network was trained entirely from scratch on the PlantVillage meta-training set.

### 2.3. Dataset and Preprocessing

We used the **PlantVillage dataset**, which contains a large number of images of healthy and diseased plant leaves. To properly evaluate the FSL models, it is crucial that the classes seen during training, validation, and testing are disjoint. Our `PlantVillageDatasetPreparer` class handles this by splitting the 38 available classes into three distinct sets:

- **Meta-Train (26 classes)**: Used for training the model episodically.
- **Meta-Validation (6 classes)**: Used to monitor training progress and save the best-performing model.
- **Meta-Test (6 classes)**: A completely held-out set of classes used for final performance evaluation.

All images were resized to 224x224 pixels. For training, we applied random horizontal flipping and color jittering. All images were normalized using standard ImageNet statistics.

### 2.4. Experimental Setup

The core of our training logic is managed by the `MetaLearner` class, which handles the episodic training loop, optimization, and evaluation.

**Configurations**: We evaluated all models under two standard FSL scenarios:
- **5-way 1-shot**: 5 classes, 1 support image per class.
- **5-way 5-shot**: 5 classes, 5 support images per class.

**Training Details**:
- **Optimizer**: AdamW with a learning rate of 1e-4.
- **Scheduler**: A StepLR scheduler that reduces the learning rate by a factor of 0.5 every 10 epochs.
- **Epochs**: The pre-trained ViT and ResNet50 models were trained for 10 epochs, while the from-scratch Conv-4 model was trained for 30 epochs to allow for sufficient convergence.
- **Episodes**: Each training epoch consisted of 100 episodes, and validation was performed on 50 episodes.
- **Evaluation**: Final performance was measured on the meta-test set over 1,000 randomly generated episodes. We report the mean accuracy, standard deviation, and the 95% confidence interval to ensure a robust evaluation.

## 3. Results and Discussion

The performance of the three backbone architectures in both the 1-shot and 5-shot configurations is summarized in the tables below.

### 3.1. 5-Way 1-Shot Results

This configuration represents the most challenging scenario, where the model must generalize from a single example.

| **Backbone**           | **Best Validation Accuracy** | **Meta-Test Accuracy (Mean ± Std Dev)** | **95% Confidence Interval** |
|-------------------------|-----------------------------|-----------------------------------------|-----------------------------|
| ViT (Pre-trained)       | 91.36%                     | 87.83% ± 8.44%                         | [87.30%, 88.35%]           |
| ResNet50 (Pre-trained)  | 81.76%                     | 74.68% ± 10.74%                        | [74.01%, 75.35%]           |
| Conv-4 (From Scratch)   | 65.36%                     | 61.07% ± 11.96%                        | [60.33%, 61.81%]           |

**Discussion**:  
In the 1-shot setting, the pre-trained ViT backbone is the clear winner, outperforming ResNet50 by over 13 percentage points and the from-scratch Conv-4 by nearly 27 points. This significant margin underscores the value of pre-training on a large dataset like ImageNet, which provides the model with a rich feature representation that can be adapted to new tasks. The Conv-4 model struggles significantly, indicating that learning a versatile feature space from scratch is extremely difficult under the data constraints of meta-learning.

### 3.2. 5-Way 5-Shot Results

Increasing the number of support examples to five provides the model with more information to form robust class prototypes.

| **Backbone**           | **Best Validation Accuracy** | **Meta-Test Accuracy (Mean ± Std Dev)** | **95% Confidence Interval** |
|-------------------------|-----------------------------|-----------------------------------------|-----------------------------|
| ViT (Pre-trained)       | 97.68%                     | 95.50% ± 4.41%                         | [95.22%, 95.77%]           |
| ResNet50 (Pre-trained)  | 93.60%                     | 90.94% ± 6.24%                         | [90.55%, 91.33%]           |
| Conv-4 (From Scratch)   | 82.16%                     | 78.96% ± 9.54%                         | [78.37%, 79.55%]           |

**Discussion**:  
As expected, all models show a substantial improvement in the 5-shot setting. With more support examples, the calculated prototypes are more representative of their respective classes, leading to more accurate classification. The ViT model continues to lead, achieving an impressive 95.50% accuracy. The performance gap between the models persists, but the Conv-4 model benefits the most from the additional data, with its accuracy increasing by nearly 18 percentage points. This suggests that while shallow models trained from scratch are weak in extreme low-data regimes, they can become more effective as the number of shots increases.

## 4. Conclusion and Future Work

This project successfully implemented and evaluated a Prototypical Network for few-shot plant disease classification. Our findings lead to two main conclusions:

1. **Prototypical Networks are an effective method** for this task, providing a simple yet powerful framework for learning from limited data.
2. **The choice of backbone architecture is paramount**. Pre-trained models, particularly the Vision Transformer, provide a significant advantage by leveraging knowledge transferred from large-scale datasets. The ViT backbone consistently outperformed both Res