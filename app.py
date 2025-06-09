import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ────────────────────────────────────────────────────────────────────────────────
# ⚙️  APP CONFIG
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Weather Work‑Suitability", layout="centered", page_icon="🌦️")

# ────────────────────────────────────────────────────────────────────────────────
# 📥 DATA LOADING & PREPARATION
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data …")
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["Location"] = df["Location"].str.strip()
    df["Date_Time"] = pd.to_datetime(df["Date_Time"], errors="coerce")
    df["Work_Suitability"] = df["Work_Suitability"].map({"Yes": 1, "No": 0})
    return df

df = load_data("updated_weather_data.csv")

# ────────────────────────────────────────────────────────────────────────────────
# 🧠 MODEL TRAINING (LOGISTIC REGRESSION)
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models …", ttl=24*3600)
def train_models(data: pd.DataFrame):
    X_w = data[["Temperature_C", "Humidity_pct", "Precipitation_mm", "Wind_Speed_kmh"]]
    le_desc = LabelEncoder()
    y_w = le_desc.fit_transform(data["Weather_Description"].astype(str))

    weather_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, multi_class="multinomial", n_jobs=-1)),
    ])

    Xw_train, Xw_test, yw_train, yw_test = train_test_split(X_w, y_w, test_size=0.2, random_state=42)
    weather_pipe.fit(Xw_train, yw_train)
    desc_acc = accuracy_score(yw_test, weather_pipe.predict(Xw_test))

    data["Weather_Desc_Cat"] = le_desc.transform(data["Weather_Description"].astype(str))
    X_s = data[["Temperature_C", "Humidity_pct", "Precipitation_mm", "Wind_Speed_kmh", "Weather_Description"]]
    y_s = data["Work_Suitability"]

    preproc = ColumnTransformer([
        ("weather_desc", OneHotEncoder(handle_unknown="ignore"), ["Weather_Description"])
    ], remainder="passthrough")

    suit_pipe = Pipeline([
        ("prep", preproc),
        ("clf", LogisticRegression(max_iter=2000, n_jobs=-1, class_weight="balanced")),
    ])

    Xs_train, Xs_test, ys_train, ys_test = train_test_split(X_s, y_s, test_size=0.2, random_state=42)
    suit_pipe.fit(Xs_train, ys_train)
    suit_acc = accuracy_score(ys_test, suit_pipe.predict(Xs_test))

    return weather_pipe, suit_pipe, le_desc, desc_acc, suit_acc

weather_model, work_model, le_desc, weather_acc, work_acc = train_models(df)

for k in ("manual_strict", "realtime_strict"):
    if k not in st.session_state:
        st.session_state[k] = False

# ────────────────────────────────────────────────────────────────────────────────
# 📋 MAIN APP CONTENT (VERTICAL LAYOUT)
# ────────────────────────────────────────────────────────────────────────────────

st.title("🌦️ Work Suitability & Weather Predictor")
st.markdown("Enter weather details to forecast conditions and work suitability.")

locs = sorted(df["Location"].dropna().unique())
sel_loc = st.selectbox("📍 Choose a Location", locs)
st.dataframe(df[df["Location"] == sel_loc].tail(5), use_container_width=True, height=150)

temp = st.slider("🌡️ Temperature (°C)", -20.0, 50.0, 25.0)
hum = st.slider("💧 Humidity (%)", 0.0, 100.0, 50.0)
precip = st.slider("🌧️ Precipitation (mm)", 0.0, 100.0, 10.0)
wind = st.slider("💨 Wind Speed (km/h)", 0.0, 100.0, 20.0)

if st.button("🔍 Predict"):
    input_df = pd.DataFrame([{ "Temperature_C": temp, "Humidity_pct": hum, "Precipitation_mm": precip, "Wind_Speed_kmh": wind }])
    weather_code = weather_model.predict(input_df)[0]
    weather_label = le_desc.inverse_transform([weather_code])[0]
    st.info(f"🌤️ **Predicted Weather:** {weather_label}")

    input_df["Weather_Description"] = weather_label

    strict = (
        temp < 5 or temp > 40 or
        precip > 8 or wind > 30 or
        weather_label.lower() in ["heavy rainfall", "flood", "storm", "cyclone"]
    )
    st.session_state.manual_strict = strict

    if strict:
        st.warning("⚠️ Harsh conditions – work not suitable (rule override).")
        work_pred = 0
    else:
        work_pred = work_model.predict(input_df)[0]

    if work_pred == 1:
        st.success("✅ Work is Suitable under these conditions.")
    else:
        st.error("❌ Work is NOT Suitable under these conditions.")

    if st.session_state.realtime_strict and strict:
        st.warning("🚨 Harsh weather detected in BOTH manual and real‑time inputs!")

# ────────────────────────────────────────────────────────────────────────────────
# 📊 REAL-TIME WEATHER & VISUALS
# ────────────────────────────────────────────────────────────────────────────────

st.header("📊 Insights & Real‑Time Forecast")
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Records", len(df))
kpi2.metric("Weather‑Desc Acc", f"{weather_acc*100:.1f}%")
kpi3.metric("Work‑Suit Acc", f"{work_acc*100:.1f}%")

st.markdown("### 🌐 Real‑Time Weather")
city_coords = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Agra": (27.1767, 78.0081),
    "Bangalore": (12.9716, 77.5946),
    "Shillong": (25.5788, 91.8933),
}
city_sel = st.selectbox("City", list(city_coords.keys()))
api_key = "6OtYQy1ODeUOOH2UJJYwnglbW1AidxPr"

def get_realtime(lat, lon, key):
    url = "https://api.tomorrow.io/v4/timelines"
    params = {
        "location": f"{lat},{lon}",
        "fields": ["temperature", "humidity", "precipitationIntensity", "windSpeed"],
        "units": "metric", "timesteps": "current"
    }
    headers = {"apikey": key}
    r = requests.get(url, params=params, headers=headers, timeout=20)
    if r.status_code != 200:
        st.error("API error ❌")
        return None
    v = r.json()["data"]["timelines"][0]["intervals"][0]["values"]
    return {
        "Temperature_C": v["temperature"],
        "Humidity_pct": v["humidity"],
        "Precipitation_mm": v.get("precipitationIntensity", 0.0),
        "Wind_Speed_kmh": v["windSpeed"],
    }

if st.button("🚀 Fetch & Predict", key="rt"):
    lat, lon = city_coords[city_sel]
    rt = get_realtime(lat, lon, api_key)
    if rt:
        st.json(rt)
        rt_df = pd.DataFrame([rt])
        w_code = weather_model.predict(rt_df)[0]
        w_label = le_desc.inverse_transform([w_code])[0]
        st.info(f"🌤️ **Predicted Weather:** {w_label}")

        rt_df["Weather_Description"] = w_label

        strict_rt = (
            rt["Temperature_C"] < 5 or rt["Temperature_C"] > 40 or
            rt["Precipitation_mm"] > 8 or rt["Wind_Speed_kmh"] > 30 or
            w_label.lower() in ["heavy rainfall", "flood", "storm", "cyclone"]
        )
        st.session_state.realtime_strict = strict_rt

        if strict_rt:
            st.warning("⚠️ Harsh real‑time conditions – work not suitable (rule override).")
            rt_pred = 0
        else:
            rt_pred = work_model.predict(rt_df)[0]

        if rt_pred == 1:
            st.success("✅ Work is Suitable now.")
        else:
            st.error("❌ Work is NOT Suitable now.")

        if st.session_state.manual_strict and strict_rt:
            st.warning("🚨 Harsh weather detected in BOTH manual and real‑time inputs!")

with st.expander("📈 Historical Visualisations"):
    loc_filtered = df[df["Location"] == sel_loc]
    if loc_filtered.empty:
        st.info("No data for selected location.")
    else:
        st.line_chart(loc_filtered.set_index("Date_Time")["Temperature_C"], height=200)
        st.line_chart(loc_filtered.set_index("Date_Time")["Humidity_pct"], height=200)
        st.line_chart(loc_filtered.set_index("Date_Time")["Precipitation_mm"], height=200)
        st.line_chart(loc_filtered.set_index("Date_Time")["Wind_Speed_kmh"], height=200)
        st.line_chart(loc_filtered.set_index("Date_Time")["Work_Suitability"], height=150)

# ────────────────────────────────────────────────────────────────────────────────
# 📜 FOOTER
# ────────────────────────────────────────────────────────────────────────────────
st.markdown("""
---
<center><small>Developed by Nakul Goyal | © 2025</small></center>
""", unsafe_allow_html=True)
