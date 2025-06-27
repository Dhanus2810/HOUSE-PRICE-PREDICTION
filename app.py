# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model components
model = joblib.load("models/house_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

st.set_page_config(page_title="🏠 House Price Predictor", layout="centered")

# 🎨 Set Background Image and Styling
def set_bg():
    st.markdown(
        """
        <style>
        .stApp {
            background: url('https://www.zingbus.com/blog/wp-content/uploads/2023/07/Best-tourist-Places-in-Bangalore.jpg') no-repeat center center fixed;
            background-size: cover;
            color: black;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg()

st.markdown("<h1 style='text-align: center;'>🏠 Bengaluru House Price Prediction</h1>", unsafe_allow_html=True)

# Simplified location list for dropdown (top 10 common)
locations = [
    '1st Block Jayanagar', '1st Phase JP Nagar', '2nd Stage Nagarbhavi', '5th Block Hbr Layout',
    'Electronic City', 'HSR Layout', 'Marathahalli', 'Rajaji Nagar',
    'Sarjapur Road', 'Whitefield', 'other'
]

# User inputs
location = st.selectbox("Select Location", sorted(locations))
total_sqft = st.number_input("Total Sqft Area", min_value=100.0, value=1000.0)
bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
bhk = st.number_input("Number of Bedrooms (BHK)", min_value=1, max_value=10, value=2)

# Predict button
if st.button("Predict Price"):
    # Create input DataFrame
    input_df = pd.DataFrame([[location, total_sqft, bath, bhk]], columns=["location", "total_sqft", "bath", "bhk"])
    input_df['location'] = input_df['location'].apply(lambda x: x.strip())

    # One-hot encode location
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)

    # Scale and predict
    scaled_input = scaler.transform(input_encoded)
    prediction = model.predict(scaled_input)[0]
    st.success(f"💰 Estimated Price: ₹ {prediction:.2f} Lakhs")
