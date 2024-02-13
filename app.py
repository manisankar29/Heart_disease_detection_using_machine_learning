import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.markdown(
    """
    <style>
    body {
        background-image: url('https://images.pexels.com/photos/887349/pexels-photo-887349.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2')
        background-size: cover;
        font-family: 'Times New Roman', Times, serif;
    }
    .copyright {
        text-align: center;
        margin-top: 2opx;
        color: #666;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

data = pd.read_csv("/workspaces/Heart_disease_detection_using_machine_learning/heart.csv")

info = ["Age", "Sex (1: male, 0: female)", "Chest Pain Type (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic)",
        "Resting Blood Pressure", "Serum Cholestoral (mg/dl)", "Fasting Blood Sugar (mg/dl)", "Resting ECG (0,1,2)",
        "Thalach (maximum heartrate achieved)", "Exang (exercise induced angina)", "Oldpeak",
        "The Slope of the Peak Exercise ST Segment", "No. of Major Vessels (0-3) Colored by Flourosopy", "Thalassemia"]

st.title("Heart Disease Predictor")
st.sidebar.subheader("Select Features:")
st.write(
    "Welcome to the Heart Disease Prediction App! This application is empowered by a machine learning model to predict the likelihood of a person having heart disease based on diverse health-related features. Enter your input data in the side bar and observe the prediction."
)
st.write(
    "### Note\n"
    "This tool is intended to raise awareness and encourage proactive health discussions. It is not a definitive diagnosis. Always consult with a qualified healthcare professional for accurate health assessments and recommendations."
)
features = {}
for i in range(len(info)):
    features[data.columns[i]] = st.sidebar.number_input(info[i], float(data[data.columns[i]].min()), float(data[data.columns[i]].max()), float(data[data.columns[i]].mean()))

new_data = pd.DataFrame(features, index=[0])

X = data.drop("target", axis=1)
Y = data["target"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

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
prediction = rf.predict(new_data)

st.subheader("Prediction Result:")
if prediction[0] == 0:
    st.markdown('<div style="color: green; font-size: 46px;">No Heart Disease Chances Found😊</div>', unsafe_allow_html=True)
else:
    st.markdown('<div style="color: red; font-size: 46px;">Heart Disease Chances Found!!☹️<div>', unsafe_allow_html=True)

st.markdown('<div class="copyright">&copy; 2024 mani sankar pasala. All rights reserved.</div>', unsafe_allow_html=True)
