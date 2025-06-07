import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

# ✅ Page configuration MUST be the first Streamlit command
st.set_page_config(page_title="Work Suitability Predictor", layout="centered")

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv("weather_data_extended_with_5000_rows.csv")
    df['Work Suitability'] = df['Work Suitability'].map({'Yes': 1, 'No': 0})
    return df

df = load_data()

# Define features and target
X = df[['Temperature', 'Wind speed', 'Rain in mm', 'Location']]
y = df['Work Suitability']

# Preprocessor for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Location'])
    ],
    remainder='passthrough'
)

# Model pipeline
model = make_pipeline(preprocessor, LogisticRegression(max_iter=1000))

# Train-test split & model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# 🔷 Streamlit UI
# -------------------------------

st.title("🌤️ Weather Work Suitability Predictor")
st.markdown("Enter weather conditions to check if it's suitable for work.")

# Input form
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.number_input("🌡️ Temperature (°C)", min_value=-20.0, max_value=60.0, value=30.0)
        wind_speed = st.number_input("🌬️ Wind Speed (km/h)", min_value=0.0, max_value=150.0, value=10.0)
    with col2:
        rain = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, max_value=200.0, value=0.0)
        location = st.selectbox("📍 Location", sorted(df['Location'].unique()))

    submitted = st.form_submit_button("🔍 Predict")

# Predict
if submitted:
    input_data = pd.DataFrame([{
        "Temperature": temperature,
        "Wind speed": wind_speed,
        "Rain in mm": rain,
        "Location": location
    }])

    probability = model.predict_proba(input_data)[0][1]  # Probability of "suitable"
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success(f"✅ The weather is suitable for work! (Confidence: {probability:.2%})")
    else:
        st.error(f"⚠️ The weather is NOT suitable for work. (Confidence: {probability:.2%})")
