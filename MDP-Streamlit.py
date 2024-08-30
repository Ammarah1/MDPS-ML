# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 13:20:12 2024

@author: ammar
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
import warnings
from streamlit_option_menu import option_menu

# Paths to the models and scaler files
MODEL_PATHS = {
    "diabetes_model": "C:/Users/ammar/OneDrive - Teesside University/Dissertation/Spyder/Multiple Disease Prediction System/Final Final/diabetes_best_model_KNN.sav",
    "diabetes_scaler": "C:/Users/ammar/OneDrive - Teesside University/Dissertation/Spyder/Multiple Disease Prediction System/Final Final/diabetes_scaler.sav",
    "heart_model": "C:/Users/ammar/OneDrive - Teesside University/Dissertation/Spyder/Multiple Disease Prediction System/Final Final/heart_best_model_RandomForest.sav",
    "heart_scaler": "C:/Users/ammar/OneDrive - Teesside University/Dissertation/Spyder/Multiple Disease Prediction System/Final Final/heart_scaler.sav",
    "parkinsons_model": "C:/Users/ammar/OneDrive - Teesside University/Dissertation/Spyder/Multiple Disease Prediction System/Final Final/parkinsons_best_model.sav",
    "parkinsons_scaler": "C:/Users/ammar/OneDrive - Teesside University/Dissertation/Spyder/Multiple Disease Prediction System/Final Final/parkinsons_scaler.sav"
}

# Verify files exist
for name, path in MODEL_PATHS.items():
    if not os.path.isfile(path):
        st.error(f"File not found: {path}")

# Load the saved models and scaler
diabetes_model = pickle.load(open(MODEL_PATHS["diabetes_model"], 'rb'))
diabetes_scaler = pickle.load(open(MODEL_PATHS["diabetes_scaler"], 'rb'))
heart_disease_model = pickle.load(open(MODEL_PATHS["heart_model"], 'rb'))
heart_scaler = pickle.load(open(MODEL_PATHS["heart_scaler"], 'rb'))
parkinsons_model = pickle.load(open(MODEL_PATHS["parkinsons_model"], 'rb'))
parkinsons_scaler = pickle.load(open(MODEL_PATHS["parkinsons_scaler"], 'rb'))

# Sidebar for Navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# Prediction functions
def diabetes_prediction(input_data):
    input_data_df = pd.DataFrame([input_data], columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ])
    # Standardize the input data using the previously fitted scaler
    std_data = diabetes_scaler.transform(input_data_df)
    warnings.filterwarnings('ignore', message="X has feature names, but .* was fitted without feature names")
    prediction = diabetes_model.predict(std_data)
    return 'The person is diabetic' if prediction[0] == 1 else 'The person is not diabetic'

def heart_disease_prediction(inputs):
    input_data_df = pd.DataFrame([inputs], columns=[
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ])
    # Standardize the input data using the previously fitted scaler
    std_data = heart_scaler.transform(input_data_df)
    warnings.filterwarnings('ignore', message="X has feature names, but .* was fitted without feature names")
    prediction = heart_disease_model.predict(std_data)
    return 'The person has Heart Disease' if prediction[0] == 1 else 'The person does not have Heart Disease'

def parkinsons_prediction(inputs):
    columns = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP',
        'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
        'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
    ]

    input_data_df = pd.DataFrame([inputs], columns=columns)
    input_data_scaled = parkinsons_scaler.transform(input_data_df)
    prediction = parkinsons_model.predict(input_data_scaled)
    return 'The person has Parkinsons' if prediction[0] == 1 else 'The person does not have Parkinsons'

# Pages for different predictions
def diabetes_prediction_page():
    st.title('Diabetes Prediction using ML')
    
    # User input
    inputs = {
        'Pregnancies': st.text_input('Number of Pregnancies'),
        'Glucose': st.text_input('Glucose Level'),
        'BloodPressure': st.text_input('Blood Pressure Value'),
        'SkinThickness': st.text_input('Skin Thickness Value'),
        'Insulin': st.text_input('Insulin Level'),
        'BMI': st.text_input('BMI Value'),
        'DiabetesPedigreeFunction': st.text_input('Diabetes Pedigree Function Value'),
        'Age': st.text_input('Age of the Person')
    }
    
    # Prediction
    if st.button('Diabetes Test Result'):
        try:
            input_data = [float(inputs[key]) if inputs[key] else 0 for key in inputs]
            result = diabetes_prediction(input_data)
            st.success(result)
        except Exception as e:
            st.error(f"Error: {e}")

def heart_disease_prediction_page():
    st.title('Heart Disease Prediction using ML')
    
    # User input
    inputs = {
        'age': st.text_input('Age'),
        'sex': st.text_input('Sex (1 = male, 0 = female)'),
        'cp': st.text_input('Chest Pain Type (0-3)'),
        'trestbps': st.text_input('Resting Blood Pressure'),
        'chol': st.text_input('Cholesterol Level'),
        'fbs': st.text_input('Fasting Blood Sugar (1 = true, 0 = false)'),
        'restecg': st.text_input('Resting ECG (0-2)'),
        'thalach': st.text_input('Maximum Heart Rate'),
        'exang': st.text_input('Exercise Induced Angina (1 = yes, 0 = no)'),
        'oldpeak': st.text_input('Oldpeak'),
        'slope': st.text_input('Slope of the Peak Exercise ST Segment'),
        'ca': st.text_input('Major Vessels Colored by Fluoroscopy'),
        'thal': st.text_input('Thal (1 = normal; 2 = fixed defect; 3 = reversible defect)')
    }
    
    # Prediction
    if st.button('Heart Disease Test Result'):
        try:
            input_data = [float(inputs[key]) if inputs[key] else 0 for key in inputs]
            result = heart_disease_prediction(input_data)
            st.success(result)
        except Exception as e:
            st.error(f"Error: {e}")

def parkinsons_prediction_page():
    st.title("Parkinson's Disease Prediction using ML")
    
    # User input
    inputs = {
        'MDVP:Fo(Hz)': st.text_input('MDVP:Fo(Hz)'),
        'MDVP:Fhi(Hz)': st.text_input('MDVP:Fhi(Hz)'),
        'MDVP:Flo(Hz)': st.text_input('MDVP:Flo(Hz)'),
        'MDVP:Jitter(%)': st.text_input('MDVP:Jitter(%)'),
        'MDVP:Jitter(Abs)': st.text_input('MDVP:Jitter(Abs)'),
        'MDVP:RAP': st.text_input('MDVP:RAP'),
        'MDVP:PPQ': st.text_input('MDVP:PPQ'),
        'Jitter:DDP': st.text_input('Jitter:DDP'),
        'MDVP:Shimmer': st.text_input('MDVP:Shimmer'),
        'MDVP:Shimmer(dB)': st.text_input('MDVP:Shimmer(dB)'),
        'Shimmer:APQ3': st.text_input('Shimmer:APQ3'),
        'Shimmer:APQ5': st.text_input('Shimmer:APQ5'),
        'MDVP:APQ': st.text_input('MDVP:APQ'),
        'Shimmer:DDA': st.text_input('Shimmer:DDA'),
        'NHR': st.text_input('NHR'),
        'HNR': st.text_input('HNR'),
        'RPDE': st.text_input('RPDE'),
        'DFA': st.text_input('DFA'),
        'spread1': st.text_input('spread1'),
        'spread2': st.text_input('spread2'),
        'D2': st.text_input('D2'),
        'PPE': st.text_input('PPE')
    }
    
    # Prediction
    if st.button("Parkinson's Test Result"):
        try:
            input_data = [float(inputs[key]) if inputs[key] else 0 for key in inputs]
            result = parkinsons_prediction(input_data)
            st.success(result)
        except Exception as e:
            st.error(f"Error: {e}")

# Main page rendering based on selection
if selected == 'Diabetes Prediction':
    diabetes_prediction_page()
elif selected == 'Heart Disease Prediction':
    heart_disease_prediction_page()
elif selected == 'Parkinsons Prediction':
    parkinsons_prediction_page()
