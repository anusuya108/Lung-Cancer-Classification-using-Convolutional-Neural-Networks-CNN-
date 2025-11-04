# Lung-Cancer-Classification-using-Convolutional Neural Networks (CNN)

# Objective

This project focuses on building an end-to-end deep learning pipeline to classify lung cancer images into malignant and benign categories using Convolutional Neural Networks (CNNs). The objective is to assist in early detection by automating the classification process from chest CT scan images.

# Problem Statement

Lung cancer is one of the leading causes of cancer-related deaths worldwide. Early detection through accurate image classification can significantly improve survival rates. However, manual diagnosis is time-consuming and subjective. This project automates the detection process using CNN-based image analysis.

 # Project Workflow

Data Collection & Preprocessing

Dataset: Lung CT scan images from Kaggle’s Lung and Colon Cancer Histopathological Images Dataset

Image resizing, normalization, and data augmentation (rotation, zoom, flip) for robust training

Exploratory Data Analysis (EDA)

Distribution of cancer classes

Image samples visualization

Image pixel intensity and texture patterns

Model Development

Built a custom Convolutional Neural Network (CNN) with:

Convolutional layers (feature extraction)

MaxPooling layers (dimensionality reduction)

Dropout (to prevent overfitting)

Fully Connected layers for classification

Activation functions: ReLU and Softmax

Training & Evaluation

Optimizer: Adam

Loss Function: Categorical Cross-Entropy

Metrics: Accuracy, Precision, Recall, F1-score

Achieved ~92% accuracy on validation data

Visualization & Dashboard

Training vs. validation accuracy/loss plots

Confusion matrix for performance comparison

Simple Power BI dashboard showing model metrics, dataset summary, and predictions overview

Model Deployment (Optional)

Deployed using Streamlit for image upload & live prediction

# Key Results

Achieved 92% classification accuracy using CNN

Reduced false negatives by 15% compared to baseline models

Automated feature extraction without manual intervention

# Tools & Technologies
Category	Tools
Languages	Python
Libraries	TensorFlow, Keras, NumPy, Pandas, Matplotlib, Seaborn
Visualization	Power BI, Matplotlib
Deployment	Streamlit (optional)
Dataset Source	Kaggle – Lung and Colon Cancer Histopathological Images Dataset

# Power BI Dashboard Includes:
Model accuracy and loss comparison
Dataset class distribution
Predictions summary (malignant vs benign count)
Performance metrics overview

# Impact
This CNN-based system can be integrated into healthcare workflows for assisting radiologists in faster, reliable cancer diagnosis — showcasing strong skills in AI for healthcare, deep learning, and data visualization.
