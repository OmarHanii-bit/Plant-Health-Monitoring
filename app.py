import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load Model and Preprocessors
@st.cache_resource
def load_resources():
    try:
        model = load_model("plant_health_model.h5")
        preprocessors = joblib.load("preprocessors.pkl")
        return model, preprocessors
    except FileNotFoundError:
        return None, None

model, preprocessors = load_resources()

st.title("Plant Health Predictor")

if model is None or preprocessors is None:
    st.error("Model or preprocessors not found. Please run 'train_model.py' first.")
    st.stop()

scaler = preprocessors['scaler']
label_encoders = preprocessors['label_encoders']
feature_columns = preprocessors['columns']
cat_columns = preprocessors['cat_cols']

st.write("Enter sensor values to predict plant health:")

user_inputs = []

# Dynamic UI generation based on feature columns
for col in feature_columns:
    if col in cat_columns:
        le = label_encoders[col]
        val = st.selectbox(f"Select {col}", options=le.classes_)
        val_encoded = le.transform([val])[0]
        user_inputs.append(val_encoded)
    else:
        # Heuristic for default values or ranges could be added here
        # Using 0.0 as default
        val = st.number_input(f"Enter {col}", value=0.0)
        user_inputs.append(val)

# Prediction function
def predict_health(*inputs):
    arr = np.array(inputs).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    # Get probability of Class 1
    pred_prob = model.predict(arr_scaled)[0][0]
    
    # Logic based on data analysis:
    # High Health_Score (>70) correlates with Health_Status = 1
    # Therefore, we interpret Class 1 as "Healthy".
    threshold = 0.5
    is_healthy = pred_prob > threshold
    
    status = "Healthy (Status 1)" if is_healthy else "Unhealthy (Status 0)"
    return status, pred_prob

if st.button("Predict"):
    status, prob = predict_health(*user_inputs)
    
    st.write("---")
    if "Healthy" in status and "Unhealthy" not in status:
         st.success(f"Predicted Status: {status}")
         st.info(f"Confidence: {prob:.2%}")
         st.caption("Note: Based on data, Status 1 correlates with higher Health Scores.")
    else:
         st.error(f"Predicted Status: {status}")
         st.info(f"Confidence: {1-prob:.2%}")
         st.caption("Note: Based on data, Status 0 correlates with lower Health Scores.")
