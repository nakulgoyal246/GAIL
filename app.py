import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

st.set_page_config(page_title="Work Suitability Predictor", layout="centered")

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv("weather_data_extended_with_5000_rows.csv")
    df['Work Suitability'] = df['Work Suitability'].map({'Yes': 1, 'No': 0})
    return df

df = load_data()

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

with st.expander("📊 Explore Weather Data (Click to Expand)", expanded=False):
    st.markdown("### 🔍 Work Suitability Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Suitable for Work", int(df['Work Suitability'].sum()))

    st.divider()

    # Work Suitability Distribution
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ✅ Suitability Count")
        fig1, ax1 = plt.subplots()
        sns.countplot(data=df, x='Work Suitability', ax=ax1)
        ax1.set_xticklabels(['No', 'Yes'])
        ax1.set_title('Work Suitability Count')
        st.pyplot(fig1)

    with col2:
        st.markdown("#### 📍 Suitability by Location")
        location_stats = df.groupby("Location")["Work Suitability"].mean().sort_values(ascending=False)
        st.bar_chart(location_stats)

    st.divider()

    st.markdown("### 📈 Weather Features vs Suitability")

    # Temperature & Wind Speed
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("##### 🌡️ Temperature")
        fig2, ax2 = plt.subplots()
        sns.histplot(data=df, x="Temperature", hue="Work Suitability", multiple="stack", ax=ax2)
        ax2.set_title("Temperature vs Suitability")
        st.pyplot(fig2)

    with col4:
        st.markdown("##### 💨 Wind Speed")
        fig3, ax3 = plt.subplots()
        sns.histplot(data=df, x="Wind speed", hue="Work Suitability", multiple="stack", ax=ax3)
        ax3.set_title("Wind Speed vs Suitability")
        st.pyplot(fig3)

    # Rainfall
    col5, col6 = st.columns(2)
    with col5:
        st.markdown("##### 🌧️ Rainfall")
        fig4, ax4 = plt.subplots()
        sns.histplot(data=df, x="Rain in mm", hue="Work Suitability", multiple="stack", ax=ax4)
        ax4.set_title("Rainfall vs Suitability")
        st.pyplot(fig4)

    with col6:
        st.markdown("##### 💡 Tip:")
        st.info("Use this data to understand what conditions impact work suitability. Extreme temperatures, high wind, and heavy rainfall usually reduce suitability.")

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

# Streamlit UI

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
