# Logistic Regression - Binary Classification
Overview
This project demonstrates binary classification using logistic regression on the breast cancer dataset from scikit-learn. The implementation includes comprehensive model evaluation, threshold tuning, and visualization of key concepts including the ROC curve and sigmoid function.
Technologies and Libraries

Python 3.x
Scikit-learn: Machine learning algorithms and evaluation metrics
Pandas: Data manipulation and analysis
NumPy: Numerical computations
Matplotlib: Data visualization

Dataset
The project utilizes the Wisconsin Breast Cancer Dataset from scikit-learn, which contains 569 samples with 30 features each. The target variable represents tumor classification where 0 indicates malignant and 1 indicates benign tumors.
Implementation Details
Data Preprocessing
The implementation includes proper data preprocessing steps including train-test split with an 80-20 ratio and feature standardization using StandardScaler. Standardization is crucial for logistic regression as it ensures all features contribute equally to the model training process.
Model Training
The logistic regression model is trained using scikit-learn's LogisticRegression class with default parameters. The model learns to classify tumors as malignant or benign based on the provided features.
Evaluation Metrics
The project implements comprehensive model evaluation including:

Confusion Matrix: Provides detailed breakdown of true positives, true negatives, false positives, and false negatives
Precision: Measures the accuracy of positive predictions
Recall: Measures the model's ability to identify all positive instances
ROC-AUC Score: Evaluates the model's performance across all classification thresholds

Visualization Components
The implementation includes two key visualizations:

ROC Curve: Displays the trade-off between true positive rate and false positive rate across different thresholds
Sigmoid Function: Illustrates the mathematical foundation of logistic regression probability calculations

Threshold Tuning
The project demonstrates threshold adjustment capabilities by implementing custom threshold evaluation at 0.3, showing how different thresholds affect precision and recall metrics.
Key Features
The implementation showcases several important aspects of binary classification:

Proper data splitting and preprocessing techniques
Comprehensive evaluation using multiple metrics
Probability prediction capabilities for threshold tuning
Visual representation of model performance and underlying mathematical concepts
Custom threshold implementation for optimizing specific business requirements

Usage
Execute the script to train the logistic regression model and generate evaluation metrics along with visualization plots. The implementation provides both default threshold results and demonstrates custom threshold tuning for specific use cases.
Results Interpretation
The confusion matrix and evaluation metrics provide insights into model performance, while the ROC curve visualization helps understand the trade-off between sensitivity and specificity. The sigmoid function plot illustrates the probability transformation mechanism underlying logistic regression predictions.
This implementation serves as a comprehensive example of binary classification best practices, including proper evaluation techniques and visualization methods for effective model interpretation.

Author :
Sreejesh M S
