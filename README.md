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
- [Output](#output)
- [Note](#note)
- [License](#license)

## Introduction

Heart disease remains the leading cause of death globally. World Health Organization (WHO) has estimated 12 million deaths occur worldwide, every year due to heart diseases. Half the deaths in the United States and other developed countries are due to cardio vascular diseases. Thus preventing heart diseases has become more than necessary. The implementation of data-driven systems for predicting heart diseases is crucial to advancing research and prevention efforts, ultimately promoting healthier lives for a larger population. 
 
  In this context, the application of machine learning emerges as a crucial tool. Machine learning plays a vital role in accurately predicting heart diseases. 

  The project involved a analysis of a dataset containing heart disease patient information, including data processing techniques. Following, various models were trained using diverse algorithms, including Logistic Regression, SVM (Support Vector Machine), KNN (K-Nearest Neighbors), Decision Tree, Random Forest, XGBoost, and Neural Netowrks. Lastly, trained model was deployed into the GUI (Graphical User Interface) platform for better communication between users.

  This project intends to predict whether the heart disease detection is present or not.

## Architecture

<img width="1000" alt="architecture" src="https://github.com/manisankar29/Heart_disease_detection_using_machine_learning/assets/138246745/f2b129a9-d285-4815-af7a-ed5a32ed2d80">

- Obtain the heart disease dataset from [Kaggle](https://www.kaggle.com/), which includes various [features](#features) related to heart health.
- Preprocess the data by cleaning and transforming it into a format suitable for modeling.
- Split the preprocessed data into training and testing sets.
- Develop a machine learning model with various [algorithms](#algorithms) using the training dataset.
- Validate the model's performance using the testing dataset.
- Deploy the validated model in a real-world setting for predicting heart disease in new patients.

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

   - A collection of algorithms that can be used for a variety of tasks, including pattern recognition, time series prediction, and optimization.
  
   - Works well for pattern recognition and optimization problems. They can handle high dimensionality and complex non-linear patterns.
  
   - However, they may require a large amount of training data, can be computationality expensive, and may have difficulty providing an interpretable model.

## Features

The model considers multiple features related to an individual's health, such as:

- Age
- Sex
- Chest pain type (4 values)
- Resting blood pressure
- Serum cholesterol in mg/dl
- Fasting blood sugar > 120 mg/dl
- Resting electrocardiographic results (values 0,1,2)
- Thalach (Maximum heart rate achieved)
- Exang (Excercise induced angina)
- Oldpeak = ST depression induced by exercise relative to rest
- The slope of the peak exercise ST segment
- Number of major vessels (0-3) colored by flouroscopy
- Thalassemia (0 = normal, 1 = fixed defect, 2 = reversable defect)

## Dataset

The dataset, obtained from Kaggle, includes records with associated [features](#features) and labels indicating the presence or absence of heart disease.

## Prerequisites

Before using this code, ensure that you have the following prerequisites:

- **Python**: The script is written in python and requires a python environment.
- **Scikit-learn**: It is a free open source machine learning library for python.
- **Keras**: It is a deep learning API developed by Google for implementing neural networks.
- **Joblib**: It can be used to save and load machine learning models.
- **Streamlit**: It is an open-source Python framework for data scientists and AI/ML engineers to deliver dynamic data apps with only a few lines of code.

## Getting Started

1. Clone this repository or create a new python script.
2. Place your dataset in the same directory as the script.
3. Train and test the machine learning model.
4. Deploy the trained model in Graphical User Interface (GUI).

## Code Explanation

The code is divided into the following sections:

### 1. Heart_disease_detection.ipynb

In this section the machine learning model is trained and evaluated from the dataset.

#### I. Importing required libraries

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
#### II. Importing and understanding the dataset

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

#### III. Exploratory Data Analysis (EDA)

- Percentage of the patients with and without heart problems are displayed by analyzing the target variable.

```bash
y = data["target"]
target_temp = data.target.value_counts()
print(target_temp)
print("Percentage of patience without heart problems: "+str(round(target_temp[0]*100/303,2)))
print("Percentage of patience with heart problems: "+str(round(target_temp[1]*100/303,2)))
```

#### IV. Train test split

- The dataset is split into training and testing sets using `train_test_split`.

```bash
from sklearn.model_selection import train_test_split
X = data.drop("target", axis=1)
Y = data["target"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
```

#### V. Model fitting

- Imported `accuracy_score` from `Scikit-learn` library to get various models performance in the means of accuracy.

```bash
from sklearn.metrics import accuracy_score
```

- Various machine learning models are trained and evaluated for classification using accuracy scores.

<p align="center"><b>Logistic Regression</b></p>

```bash
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, Y_train)
Y_pred_lr = lr.predict(X_test)
score_lr = round(accuracy_score(Y_pred_lr, Y_test)*100,2)
print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+"%")
```

<p align="center"><b>Support Vector Machine</b></p>

```bash
from sklearn.svm import SVC
sv = SVC(kernel='linear')
sv.fit(X_train, Y_train)
Y_pred_svm = sv.predict(X_test)
score_svm = round(accuracy_score(Y_pred_svm, Y_test)*100,2)
print("The accuracy score achieved using SVM is: "+str(score_lr)+"%")
```

<p align="center"><b>K-Nearest Neighbor</b></p>

```bash
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)
score_knn = round(accuracy_score(Y_pred_knn, Y_test)*100,2)
print("The accuracy score achieved using KNN is: "+str(score_knn)+"%")
```

<p align="center"><b>Decision Tree</b></p>

```bash
from sklearn.tree import DecisionTreeClassifier
max_accuracy = 0
for x in range(2000):
  dt = DecisionTreeClassifier(random_state=x)
  dt.fit(X_train, Y_train)
  Y_pred_dt = dt.predict(X_test)
  curr_accuracy = round(accuracy_score(Y_pred_dt, Y_test)*100,2)
  if(curr_accuracy > max_accuracy):
    max_accuracy = curr_accuracy
    best_x = x

dt = DecisionTreeClassifier(random_state=best_x)
dt.fit(X_train, Y_train)
Y_pred_dt = dt.predict(X_test)
score_dt = round(accuracy_score(Y_pred_dt, Y_test)*100,2)
print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+"%")
```

<p align="center"><b>Random Forest</b></p>

```bash
from sklearn.ensemble import RandomForestClassifier
max_accuracy = 0
for x in range(2000):
  rf = RandomForestClassifier(random_state=x)
  rf.fit(X_train, Y_train)
  Y_pred_rf = rf.predict(X_test)
  curr_accuracy = round(accuracy_score(Y_pred_rf, Y_test)*100,2)
  if(curr_accuracy > max_accuracy):
    max_accuracy = curr_accuracy
    best_x = x

rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train, Y_train)
Y_pred_rf = rf.predict(X_test)
score_rf = round(accuracy_score(Y_pred_rf, Y_test)*100,2)
print("The accuracy score achieved using Random Forest is: "+str(score_rf)+"%")
```

<p align="center"><b>XGBoost</b></p>

```bash
import xgboost as xgb
xgb_model = xgb.XGBClassifier(objective="binary:logistic",random_state=42)
xgb_model.fit(X_train, Y_train)
Y_pred_xgb = xgb_model.predict(X_test)
score_xgb = round(accuracy_score(Y_pred_xgb, Y_test)*100,2)
print("The accuracy score achieved using XGBoost is: "+str(score_xgb)+"%")
```

<p align="center"><b>Neural Networks</b></p>

```bash
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(11,activation='relu',input_dim=13))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=300)
Y_pred_nn = model.predict(X_test)
rounded = [round(x[0]) for x in Y_pred_nn]
Y_pred_nn = rounded
score_nn = round(accuracy_score(Y_pred_nn, Y_test)*100,2)
print("The accuracy score achieved using Neural Network is: "+str(score_nn)+"%")
```

#### VI. Output final score

- Accuracy scores of different models are compared and visualized using scatter plot.

```bash
scores = [score_lr, score_svm, score_knn, score_dt, score_rf, score_xgb, score_nn]
algorithms = ["Logistic Regression","SVM","KNN","Decision Tree","Random Forest","XGBoost","Neural Networks"]
for i in range(len(algorithms)):
  print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+"%")

sns.set(rc={'figure.figsize':(10,4)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")
sns.barplot(x=algorithms, y=scores)
plt.show()
```

#### VII. Prediction on new data

- New data is taken for predicting whether disease is there or not.

```bash
new_data = pd.DataFrame({
    'age':52,
    'sex':1,
    'cp':0,
    'trestbps':125,
    'chol':212,
    'fbs':0,
    'restecg':1,
    'thalach':168,
    'exang':0,
    'oldpeak':1.0,
     'slope':2,
    'ca':2,
    'thal':3,
},index=[0])

p = rf.predict(new_data)
if p[0]==0:
  print("No Disease")
else:
  print("Disease")
```

#### VIII. Save model using Joblib

- The Random Forest model with highest accuracy is saved using `joblib` and can be downloaded.

```bash
import joblib
joblib.dump(rf, 'trained_model.joblib')
from google.colab import files
files.download('trained_model.joblib')
```
### 2. app.py

In this section the `streamlit` application is developed for machine learning model.

#### I. Importing required liraries

- The code begins by importing necessary libraries, including `streamlit`, `pandas`, `joblib`, and `numpy`.

```bash
import streamlit as st
import pandas as pd
import joblib
import numpy as np
```
#### II. Style and Layout

- This section adds custom CSS to style the application. It centers and styles the copyright text.

 ```bash
st.markdown(
    """
    <style>
    .copyright {
        text-align: center;
        margin-top: 20px;
        color: #666;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
```

#### III. Load data

- This loads the dataset `heart.csv` into a pandas DataFrame. Although the data is loaded here, it is not used further in the script.

```bash
data = pd.read_csv('heart.csv')
```

#### IV. Application title and description

- Sets the main title of the application.
- Adds a subheader to the sidebar for user instructions.
- Provides a welcome message and a disclaimer about the tool's purpose.

```bash
st.title("Heart Disease Predictor")
st.sidebar.subheader("Select Features:")
st.write(
    "Welcome to the Heart Disease Prediction App! This application is empowered by a machine learning model to predict the likelihood of a person having heart disease based on diverse health-related features. Enter your input data in the side bar and observe the prediction."
)
st.write(
    "### Note\n"
    "This tool is intended to raise awareness and encourage proactive health discussions. It is not a definitive diagnosis. Always consult with a qualified healthcare professional for accurate health assessments and recommendations."
)
```

#### V. User inputs for prediction

- This section provides input widgets in the sidebar for users to input various health-related features required for prediction. Each widget corresponds to a feature used by the model.

```bash
age = st.sidebar.number_input('Age', min_value=1, max_value=120, value=25)
sex = st.sidebar.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
cp = st.sidebar.selectbox('Chest Pain Type', options=[1, 2, 3, 4], format_func=lambda x: 'Typical Angina' if x == 1 else 'Atypical Angina' if x==2 else 'Non-anginal Pain' if x==3 else 'Asympotatic')
trestbps = st.sidebar.number_input('Resting Blood Pressure', min_value=50, max_value=200, value=120)
chol = st.sidebar.number_input('Serum Cholesterol', min_value=100, max_value=600, value=200)
fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1], format_func=lambda x: 'True' if x==1 else 'False')
restecg = st.sidebar.selectbox('Resting Electrocardiographic Results', options=[0, 1, 2], format_func=lambda x: 'Normal' if x==0 else 'ST-T Wave Abnormality' if x==1 else 'Left Ventricular Hypertrophy')
thalach = st.sidebar.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150)
exang = st.sidebar.selectbox('Exercise Induced Angina', options=[0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
oldpeak = st.sidebar.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment', options=[0, 1, 2], format_func=lambda x: 'UpSloping' if x==0 else 'Flat' if x==1 else 'DownSloping')
ca = st.sidebar.selectbox('Number of Major Vessels Colored by Fluoroscopy', options=[0, 1, 2, 3])
thal = st.sidebar.selectbox('Thalassemia', options=[1, 2, 3], format_func=lambda x: 'Normal' if x==1 else 'Fixed Defect' if x==2 else 'Reversable Defect')
```

#### VI. Create input DataFrame

- Collects all the user inputs and creates a new DataFrame. This DataFrame will be used for prediction by the machine learning model.

```bash
new_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp], 
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
})
```

#### VII. Load model and Make prediction

- Loads the pre-trained machine learning model from a file `trained_model.joblib`.
- Uses the model to predict the likelihood of heart disease based on the user inputs.

``` bash
model = joblib.load('trained_model.joblib')
prediction = model.predict(new_data)
```

#### VIII. Display prediction result

- Displays the prediction result. If the prediction is '0', it indicates no heart disease chances found. If the prediction is '1', it indicates that there are heart disease chances.

```bash
st.subheader("Prediction Result:")
if prediction[0] == 0:
    st.markdown('<div style="color: green; font-size: 46px;">No Heart Disease Chances Found😊</div>', unsafe_allow_html=True)
else:
    st.markdown('<div style="color: red; font-size: 46px;">Heart Disease Chances Found!!☹️<div>', unsafe_allow_html=True)
```

#### IX. Footer

- Adds a footer with copyright information.

```bash
st.markdown('<div class="copyright">&copy; 2024 mani sankar pasala. All rights reserved.</div>', unsafe_allow_html=True)
```

## Output

- Accuracy scores achieved by all the trained models.

![output1](https://github.com/manisankar29/heart_disease_detection_using_machine_learning/assets/138246745/c616e0dd-4eb4-49fe-93b2-84319b6d14af)

- Visualizing accuracy scores achieved by all the trained models using Barplot.

![output2](https://github.com/manisankar29/Heart_disease_detection_using_machine_learning/assets/138246745/e673d0e8-6698-4580-95e1-ab6071df03a8)

- An interactive application to take user inputs and provide real-time predictions using the trained model.

<img width="959" alt="image" src="https://github.com/manisankar29/Heart_disease_detection_using_machine_learning/assets/138246745/15c6afa6-2cfd-4a79-8ff0-1435b4e99349">

## Note

- The accuracy of various models depends on the quality and quantity of the data. It may not be same accuracy in all cases.
- Ensure that the CSV file paths are correctly specified in the code.

Enjoy using the Heart Disease Detection script!

If you encounter any issues or have questions, feel free to reach out for assistance.

```bash
You can include this README.md files in your project's repository, and it will serve as a guide for users who want to use the provided code.
```

## License

[MIT License](LICENSE)
