# -*- coding: utf-8 -*-
"""Gas Pipeline Maintenance Predictor"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import streamlit as st

# Load local dataset
dataset = pd.read_csv("pipe_thickness_loss_dataset.csv", encoding='ascii', delimiter=',')

# Data exploration (optional, for debugging)
print("Dataset columns:", dataset.columns)
print("Missing values per column:\n", dataset.isna().sum())
if dataset.isna().any().any():
    print("\nThe dataset contains missing values.")
else:
    print("\nNo missing values found in the dataset.")

# Prepare data
X = dataset.drop(columns=['Condition'])  # Features
y = dataset['Condition']  # Target
X = pd.get_dummies(X, columns=['Material', 'Grade'], drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeClassifier(max_depth=3, min_samples_split=20, random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Set page configuration
st.set_page_config(page_title="Gas Pipeline Maintenance Predictor", layout="wide", page_icon=":factory:")

# Custom CSS to ensure label visibility and form blending
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stNumberInput, .stSelectbox {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 5px;
        margin-bottom: 15px;
    }
    .stNumberInput label, .stSelectbox label {
        color: #2c3e50 !important;
        font-weight: bold;
        display: block;
        margin-bottom: 5px;
    }
    .stSelectbox div[role="listbox"] {
        color: #000000;
        background-color: #ffffff;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        border: 2px solid #ccc;
    }
    .header {
        color: #2c3e50;
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    /* Google Form iframe styling */
    .google-form-container {
        width: 100%;
        max-width: 640px;
        margin: 20px auto;
        overflow: hidden; /* Prevent overlap */
    }
    .google-form-iframe {
        width: 100%;
        height: 600px; /* Adjustable height */
        border: none;
        background-color: #1a1a1a !important; /* Force dark mode background */
        overflow-y: auto; /* Scrollbar for overflow */
        border-radius: 10px;
    }
    .google-form-iframe * {
        color: #dcdcdc !important; /* Force dark mode text */
        background-color: #1a1a1a !important; /* Ensure consistent background */
    }
    /* Dark mode adjustments (enforced) */
    @media (prefers-color-scheme: dark) {
        .main {
            background-color: #1a1a1a;
        }
        .stNumberInput, .stSelectbox {
            background-color: #2e2e2e;
        }
        .stNumberInput label, .stSelectbox label {
            color: #dcdcdc !important;
        }
        .stSelectbox div[role="listbox"] {
            color: #ffffff;
            background-color: #2e2e2e;
        }
        .result-box {
            border-color: #555;
        }
        .header {
            color: #dcdcdc;
        }
        .google-form-iframe {
            background-color: #1a1a1a !important;
        }
        .google-form-iframe * {
            color: #dcdcdc !important;
        }
    }
    @media (prefers-color-scheme: light) {
        .google-form-iframe {
            background-color: #f0f2f6 !important;
        }
        .google-form-iframe * {
            color: #2c3e50 !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown('<div class="header">Gas Pipeline Maintenance Predictor</div>', unsafe_allow_html=True)

# Input Section with Columns
col1, col2 = st.columns(2)
with col1:
    pressure = st.number_input("Max Pressure (psi)", min_value=0.0, step=1.0, key="pressure")
    temp = st.number_input("Temperature (°C)", min_value=0.0, step=1.0, key="temp")
    corrosion = st.number_input("Corrosion Impact (%)", min_value=0.0, max_value=100.0, step=1.0, key="corrosion")
with col2:
    loss = st.number_input("Thickness Loss (mm)", min_value=0.0, step=0.1, key="loss")
    years = st.number_input("Time Years", min_value=0.0, step=1.0, key="years")
    material_options = dataset['Material'].dropna().unique().tolist() if 'Material' in dataset.columns else ['Unknown']
    grade_options = dataset['Grade'].dropna().unique().tolist() if 'Grade' in dataset.columns else ['Unknown']
    if not material_options:
        material_options = ['No Material Data']
    if not grade_options:
        grade_options = ['No Grade Data']
    material = st.selectbox("Material", options=material_options, key="material")
    grade = st.selectbox("Grade", options=grade_options, key="grade")

# Predict Button
if st.button("Predict Condition", key="predict"):
    pressure = st.session_state.get("pressure", 0.0)
    temp = st.session_state.get("temp", 0.0)
    corrosion = st.session_state.get("corrosion", 0.0)
    loss = st.session_state.get("loss", 0.0)
    years = st.session_state.get("years", 0.0)
    input_data = pd.DataFrame([[pressure, temp, corrosion, loss, 0, years, 0, 0, material, grade]], 
                              columns=['Pipe_Size_mm', 'Thickness_mm', 'Max_Pressure_psi', 
                                       'Temperature_C', 'Corrosion_Impact_Percent', 'Thickness_Loss_mm', 
                                       'Material_Loss_Percent', 'Time_Years', 'Material', 'Grade'])
    input_data = pd.get_dummies(input_data, columns=['Material', 'Grade'], drop_first=True)
    input_data = input_data.reindex(columns=X.columns, fill_value=0)
    prediction = model.predict(input_data)[0]
    color = "#ff0000" if "Critical" in prediction else "#00ff00" if "Normal" in prediction else "#ffff00"
    st.markdown(f'<div class="result-box" style="background-color: {color}; color: #000000;">Condition: {prediction}</div>', unsafe_allow_html=True)

# Footer
st.markdown("<p style='text-align: center; color: #7f8c8d;'>Powered by AGIS Hackathon 2025</p>", unsafe_allow_html=True)

# Google Form for User Feedback
st.subheader("Share Your Experience")
st.markdown(
    '<div class="google-form-container">'
    '<iframe class="google-form-iframe" src="https://docs.google.com/forms/d/e/1FAIpQLSez8lf2na9RlgvMZk4IFNrU3_sLhI6oyGZ127ihbeSs2dMblA/viewform?embedded=true" '
    'frameborder="0" marginheight="0" marginwidth="0">Loading…</iframe>'
    '</div>',
    unsafe_allow_html=True
)
