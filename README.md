# HEART DISEASE DETECTION USING MACHINE LEARNING

## Table of contents

- [Introduction](#introduction)
- [Architecture](#architecture)
- [Algorithms](#algorithms)
- [Features](#features)
- [Dataset](#dataset)
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

**1. Logistic Regression:**

   - A probabilistic algoithm used for binary classification problems.
  
   - Works well for binary classification problems, is relatively simple and easy to interpret.
  
   - However, it may not work well for datasets with high dimensionality.

**2. SVM:**

   - A linear or non-linear model used for binary classification, regression, and even outlier detection.
  
   - Work well for both binary and multi-class classification problems, is also relatively simple and easy to interpret.
  
   - However, it may not work well for datasets with high dimensionality or complex non-linear patterns.

**3. KNN:**

   - A non-parametric, lazy learning algorithm used for binary classification, regression, and even pattern recognition.
  
   - Works well for classification problems with complex patterns or high dimensionality.
  
   - However, it may be slow for large datasets, as it involves computing the distance between each instance and every other instance.
  
**4. Decision Tree:**

   - A flowchart-like model used for classification and regression.
  
   - Works well for both classification and regression problems, can handle high dimensionality and complex non-linear patterns.
  
   - However, it may overfit the training data, resulting in poor generalization to new data.

**5. Random Forest:**

   - An ensemble method that uses multiple decision trees for better performance.
  
   - It works well for both classification and regression problems, can handle high dimensionality and complex non-linear patterns.
  
   - However, it may still overfit the training data.

**6. XGBoost:**

   - A gradient boosting library designed for speed and performance.
  
   - A fast and powerful ensemble method that works well for both classification and regression problems. It can handle high dimensionality and complex non-linear patterns.
  
   - However, it may still overfit the training data.
  
**7. Neural Networks:**

   - A collection of slgorithms that can be used for a variety of tasks, including pattern recognition, time series prediction, and optimization.
  
   - Works well for pattern recognition and optimization problems. They can handle high dimensionality and complex non-linear patterns.
  
   - However, they may require a large amount of training data, can be computationality expensive, and may have difficulty providing an interpretable model.

## Features

The model considers multiple features related to an individual's health, such as:

- Age
- Sex
- Cerebral Palsy (CP)
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting ECG
- Thalach (Maximum heart rate achieved)
- Exang (Excercise induced angina)
- Oldpeak
- Slope
- Cardiac arrest
- Thalassemia

## Dataset

The dataset, obtained from Kaggle, includes records with associated [features](#features) and labels indicating the presence or absence of heart disease.

## Prerequisites

Before using this code, ensure that you have the following prerequisites:

- **Python**: The script is written in python and requires a python environment.
- **Scikit-learn**: It is a free open source machine learning library for python.
- **Keras**: It is a deep learning API developed by Google for implementing neural networks.
- **Joblib**: It can be used to save and load machine learning models.
- **Ipywidgets**: These are interactive HTML widgets for Jupyter notebooks and the IPython kernel.
- **IPython**: It is an enhanced interactive environment for Python with many functionalities compared to the standard Python shell.

## Getting Started

1. Clone this repository or create a new python script.
2. Place your dataset in the same directory as the script.
3. Train and test the machine learning model.
4. Deploy the trained model in Graphical User Interface (GUI).

## Code Explanation

The code is divided into the following sections:

### I. Importing required libraries

- The code begins by importing necessary libraries, including `Numpy`, `Pandas`, `Matplotlib`, `Seaborn`, and `OS`.
- `%matplotlib inline` is a magic command for Jupyter Notebooks to display plots inline.

```bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
%matplotlib inline
print(os.listdir())
warnings.filterwarnings('ignore')
```
### II. Importing and understanding the dataset

- The dataset (`heart.csv`) is loaded using pandas.
  
 ```bash
data = pd.read_csv("/content/heart.csv")
```

- Basic exploration of the dataset is performed, including checking its shape, displaying the first rows, generating descriptive statistics, and obtaining information on data types.
  
```bash
data.shape
data.head(5)
data.describe()
data.info()
```
- Information about each column is provided based on the dataset, explaining the meaning of each feature.

```bash
info = ["age",
        "1: male, 0: female",
        "chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic",
        "resting blood pressure",
        "serum cholestoral in mg/dl",
        "fasting blood sugar > 120 mg/dl",
        "resting ECG values(0,1,2)",
        "max heartrate achieved",
        "exercise induced angina",
        "oldpeak = ST depression induced by exercise relative to rest",
        "the slope of the peak exercise ST segment",
        "no. of major vessels (0-3) colored by flourosopy",
        "thal: 3 = normal, 6 = fixed defect, 7 = revrsable defect"]
for i in range(len(info)):
  print(data.columns[i]+":\t\t"+info[i])
```

- Descriptive statistics and unique values of the target variable (`"target"`) are displayed.

```bash
data["target"].describe()
data["target"].unique()
```

### III. Exploratory Data Analysis (EDA)

- Percentage of the patients with and without heart problems are displayed by analyzing the target variable.

```bash
y = data["target"]
target_temp = data.target.value_counts()
print(target_temp)
print("Percentage of patience without heart problems: "+str(round(target_temp[0]*100/303,2)))
print("Percentage of patience with heart problems: "+str(round(target_temp[1]*100/303,2)))
```

### IV. Train test split

- The dataset is split into training and testing sets using `train_test_split`.

```bash
from sklearn.model_selection import train_test_split
X = data.drop("target", axis=1)
Y = data["target"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
```

### V. Model fitting

- Imported `accuracy_score` from `Scikit-learn` library to get various models performance in the means of accuracy.

```bash
from sklearn.metrics import accuracy_score
```

- Various machine learning models are trained and evaluated for classification using accuracy scores.

#### [Logistic Regression](#logistic-regression)
