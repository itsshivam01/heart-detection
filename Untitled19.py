#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Title and Introduction
st.title("Heart Disease Prediction App")
st.write("""
This application uses Logistic Regression to predict the likelihood of heart disease based on user-provided data.
""")

# Sidebar for user input
st.sidebar.header("Input Features")
def user_input_features():
    age = st.sidebar.slider("Age", 18, 100, 50)
    sex = st.sidebar.selectbox("Sex (0 = Female, 1 = Male)", (0, 1))
    cp = st.sidebar.selectbox("Chest Pain Type (0-3)", (0, 1, 2, 3))
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.sidebar.slider("Cholesterol (mg/dL)", 100, 600, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL (1 = Yes, 0 = No)", (0, 1))
    restecg = st.sidebar.selectbox("Resting ECG (0 = Normal, 1 = Abnormal, 2 = Hypertrophy)", (0, 1, 2))
    thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise-Induced Angina (1 = Yes, 0 = No)", (0, 1))
    oldpeak = st.sidebar.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment (0-2)", (0, 1, 2))
    ca = st.sidebar.slider("Number of Major Vessels (0-4)", 0, 4, 0)
    thal = st.sidebar.selectbox("Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)", (0, 1, 2))

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

# User input
input_df = user_input_features()

# Load dataset (replace this with your actual dataset)
  # Replace with your actual dataset
data = pd.read_csv("heart.csv")
X, Y = data.data, data.target

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train Logistic Regression
model = LogisticRegression(max_iter=200)
model.fit(X_train, Y_train)

# Prediction and Evaluation
st.write("### User Input Features")
st.write(input_df)

prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.write("### Prediction")
st.write("Heart Disease Present" if prediction[0] == 1 else "No Heart Disease")

st.write("### Prediction Probability")
st.write(f"Probability of Heart Disease: {prediction_proba[0][1]*100:.2f}%")
st.write(f"Probability of No Heart Disease: {prediction_proba[0][0]*100:.2f}%")

# Display model accuracy
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
st.write(f"### Model Accuracy: {accuracy * 100:.2f}%")

