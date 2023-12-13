# heart_disease_detection

## Table of contents

- [Introduction](#introduction)
- [Architecture](#architecture)
- [Algorithms](#algorithms)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Code Explanation](#code-explanation)
- [Example Output](#example-output)
- [Note](#note)
- [License](#license)

## Introduction

Heart disease remains the leading cause of death globally. World Health Organization (WHO) has estimated 12 million deaths occur worldwide, every year due to heart diseases. Half the deaths in the United States and other developed countries are due to cardio vascular diseases. Thus preventing heart diseases has become more than necessary. The implementation of data-driven systems for predicting heart diseases is crucial to advancing research and prevention efforts, ultimately promoting healthier lives for a larger population. 
 
  In this context, the application of machine learning emerges as a crucial tool. Machine learning plays a vital role in accurately predicting heart diseases. 

  The project involved a analysis of a dataset containing heart disease patient information, including data processing techniques. Following, various models were trained using diverse algorithms, including Logistic Regression, SVM (Support Vector Machine), KNN (K-Nearest Neighbors), Decision Tree, Random Forest, XGBoost, and Neural Netowrks. Lastly, trained model was deployed into the GUI (Graphical User Interface) platform for better communication between users.

  This project intends to predict whether the heart disease detection is present or not.

## Architecture

![Image](https://github.com/manisankar29/heart_disease_detection_using_machine_learning/assets/138246745/88403914-41f4-4dcf-8fb6-3197c6d34892)


- Creating an architecture diagram for a heart disease detection project involves illustrating the key components, their interactions, and the flow of data within the system.

- The heart disease detection system is designed with a modular and scalable architecture to efficiently handle the complexity of detecting heart disease.

- The core components include data acquistion, preprocessing, modeling, training, testing, and predicting.

- At the base of the architecture, data acquistion modules gather heart disease data from various sources, such as ECG and historical records.

## Algorithms

1. Logistic Regression:

   - A probabilistic algoithm used for binary classification problems.
  
   - Works well for binary classification problems, is relatively simple and easy to interpret.
  
   - However, it may not work well for datasets with high dimensionality.

2. SVM:

   - A linear or non-linear model used for binary classification, regression, and even outlier detection.
  
   - Work well for both binary and multi-class classification problems, is also relatively simple and easy to interpret.
  
   - However, it may not work well for datasets with high dimensionality or complex non-linear patterns.

3. KNN:

   - A non-parametric, lazy learning algorithm used for binary classification, regression, and even pattern recognition.
  
   - Works well for classification problems with complex patterns or high dimensionality.
  
   - However, it may be slow for large datasets, as it involves computing the distance between each instance and every other instance.
  
4. Decision Tree:

   - A flowchart-like model used for classification and regression.
  
   - Works well for both classification and regression problems, can handle high dimensionality and complex non-linear patterns.
  
   - However, it may overfit the training data, resulting in poor generalization to new data.

5. Random Forest:

   - An ensemble method that uses multiple decision trees for better performance.
  
   - It works well for both classification and regression problems, can handle high dimensionality and complex non-linear patterns.
  
   - However, it may still overfit the training data.

6. XGBoost:

   - A gradient boosting library designed for speed and performance.
  
   - A fast and powerful ensemble method that works well for both classification and regression problems. It can handle high dimensionality and complex non-linear patterns.
  
   - However, it may still overfit the training data.
  
7. Neural Networks:

   - A collection of slgorithms that can be used for a variety of tasks, including pattern recognition, time series prediction, and optimization.
  
   - Works well for pattern recognition and optimization problems. They can handle high dimensionality and complex non-linear patterns.
  
   - However, they may require a large amount of training data, can be computationality expensive, and may have difficulty providing an interpretable model.
