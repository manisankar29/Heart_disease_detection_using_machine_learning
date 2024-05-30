import streamlit as st
import pandas as pd
import joblib
import numpy as np

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

data = pd.read_csv('heart.csv')

st.title("Heart Disease Predictor")
st.sidebar.subheader("Select Features:")
st.write(
    "Welcome to the Heart Disease Prediction App! This application is empowered by a machine learning model to predict the likelihood of a person having heart disease based on diverse health-related features. Enter your input data in the side bar and observe the prediction."
)
st.write(
    "### Note\n"
    "This tool is intended to raise awareness and encourage proactive health discussions. It is not a definitive diagnosis. Always consult with a qualified healthcare professional for accurate health assessments and recommendations."
)

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

model = joblib.load('trained_model.joblib')
prediction = model.predict(new_data)

st.subheader("Prediction Result:")
if prediction[0] == 0:
    st.markdown('<div style="color: green; font-size: 46px;">No Heart Disease Chances Found😊</div>', unsafe_allow_html=True)
else:
    st.markdown('<div style="color: red; font-size: 46px;">Heart Disease Chances Found!!☹️<div>', unsafe_allow_html=True)

st.markdown('<div class="copyright">&copy; 2024 mani sankar pasala. All rights reserved.</div>', unsafe_allow_html=True)
