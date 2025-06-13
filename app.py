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
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# ────────────────────────────────────────────────────────────────────────────────
# ⚙️  PDF
# ────────────────────────────────────────────────────────────────────────────────

def generate_pdf_report(forecast_data, city_name):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(temp_file.name, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, f"3-Day Weather & Work Suitability Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 70, f"Location: {city_name}")
    
    y = height - 120
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Day")
    c.drawString(100, y, "Date")
    c.drawString(200, y, "Weather")
    c.drawString(350, y, "Work Suitability")
    c.setFont("Helvetica", 12)

    for i, day in enumerate(forecast_data, start=1):
        y -= 25
        values = day["values"]
        date = day["startTime"][:10]
        data = {
            "Temperature_C": values["temperature"],
            "Humidity_pct": values["humidity"],
            "Precipitation_mm": values.get("precipitationIntensity", 0.0),
            "Wind_Speed_kmh": values["windSpeed"]
        }

        df_input = pd.DataFrame([data])
        w_code = weather_model.predict(df_input)[0]
        w_label = le_desc.inverse_transform([w_code])[0]
        df_input["Weather_Description"] = w_label

        strict = (
            data["Temperature_C"] < 5 or data["Temperature_C"] > 50 or
            data["Precipitation_mm"] > 8 or data["Wind_Speed_kmh"] > 30 or
            w_label.lower() in ["heavy rainfall", "flood", "storm", "cyclone"]
        )

        suitability = "Not Suitable" if strict or work_model.predict(df_input)[0] == 0 else "Suitable"

        c.drawString(50, y, f"Day {i}")
        c.drawString(100, y, date)
        c.drawString(200, y, w_label)
        c.drawString(350, y, suitability)

    c.save()
    return temp_file.name

# ────────────────────────────────────────────────────────────────────────────────
# ⚙️  REPORT
# ────────────────────────────────────────────────────────────────────────────────

def send_email_report(recipient_email, forecast_data, city_name):
    sender_email = st.secrets["email"]["sender"]
    sender_password = st.secrets["email"]["password"]


    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = f"📄 3-Day Forecast Report for {city_name}"

    body = f"Hello,\n\nPlease find attached the 3-day weather and work suitability forecast report for {city_name}.\n\nRegards,\nWeather App"
    msg.attach(MIMEText(body, "plain"))

    # Generate and attach PDF
    pdf_path = generate_pdf_report(forecast_data, city_name)
    from email.mime.application import MIMEApplication

    with open(pdf_path, "rb") as f:
        part = MIMEApplication(f.read(), _subtype="pdf")
        part.add_header("Content-Disposition", "attachment", filename=f"Forecast_{city_name}.pdf")
        msg.attach(part)


    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        os.remove(pdf_path)
        return True
    except Exception as e:
        st.error(f"Email sending failed: {e}")
        return False


# ────────────────────────────────────────────────────────────────────────────────
# ⚙️  APP CONFIG
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Weather Work‑Suitability", layout="wide", page_icon="🌦️")

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
        temp < 5 or temp > 50 or
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
    "GAIL_Vijaipur_Madhya_Pradesh": (24.4840, 77.1570),
    "GAIL_Pata_Uttar_Pradesh": (26.6944, 79.3883),
    "GAIL_Gandhar_Gujarat": (21.7265, 72.9150),
    "GAIL_Vaghodia_Gujarat": (22.3051, 73.4002),
    "GAIL_Lakwa_Assam": (27.0127, 94.8896),
    "GAIL_Usar_Maharashtra": (18.2911, 73.1317),
    "GAIL_Dibrugarh_Assam": (27.4728, 94.9120),
    "GAIL_Dahej_Gujarat": (21.7129, 72.5820),
    "GAIL_Mangalore_Karnataka": (12.9153, 74.8560),
    "GAIL_Ranchi_Jharkhand": (23.3441, 85.3096),
    "GAIL_Nashik_Maharashtra": (19.9975, 73.7898),
    "GAIL_Meerut_Uttar_Pradesh": (28.9845, 77.7064),
    "GAIL_Guna_Madhya_Pradesh": (24.6476, 77.3111),
    "GAIL_Khagaria_Bihar": (25.4726, 86.4721),
    "GAIL_Kutch_Gujarat": (23.7333, 68.9667),
    "GAIL_Chitradurga_Karnataka": (14.2306, 76.4023),
    "GAIL_Tirunelveli_Tamil_Nadu": (8.7139, 77.7564),
    "GAIL_Pata_Solar_Uttar_Pradesh": (26.6944, 79.3883),
}
city_sel = st.selectbox("City", list(city_coords.keys()))
api_key = "tc9mfVHKkts32FTlnGx5cwXobWW7ZOBo"

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

# ────────────────────────────────────────────────────────────────────────────────
# 📊 AUTO MAIL
# ────────────────────────────────────────────────────────────────────────────────

def send_auto_email(location, weather_label=None, work_suitability=None, forecast_summary=None):
    location_email_map = {
        "GAIL_Vijaipur_Madhya_Pradesh": "nakulgoyal298@gmail.com, nakulgoyal1303@gmail.com",
        "GAIL_Pata_Uttar_Pradesh": "jsdeshwal85@gmail.com",
        "GAIL_Gandhar_Gujarat": "gandhar@gail.co.in",
        "GAIL_Vaghodia_Gujarat": "vaghodia@gail.co.in",
        "GAIL_Lakwa_Assam": "lakwa@gail.co.in",
        "GAIL_Usar_Maharashtra": "usar@gail.co.in",
        "GAIL_Dibrugarh_Assam": "dibrugarh@gail.co.in",
        "GAIL_Dahej_Gujarat": "dahej@gail.co.in",
        "GAIL_Mangalore_Karnataka": "mangalore@gail.co.in",
        "GAIL_Ranchi_Jharkhand": "ranchi@gail.co.in",
        "GAIL_Nashik_Maharashtra": "nashik@gail.co.in",
        "GAIL_Meerut_Uttar_Pradesh": "meerut@gail.co.in",
        "GAIL_Guna_Madhya_Pradesh": "guna@gail.co.in",
        "GAIL_Khagaria_Bihar": "khagaria@gail.co.in",
        "GAIL_Kutch_Gujarat": "kutch@gail.co.in",
        "GAIL_Chitradurga_Karnataka": "chitradurga@gail.co.in",
        "GAIL_Tirunelveli_Tamil_Nadu": "tirunelveli@gail.co.in",
        "GAIL_Pata_Solar_Uttar_Pradesh": "patasolar@gail.co.in",
    }

    receiver_email = location_email_map.get(location)
    if not receiver_email:
        st.warning(f"No email configured for {location}")
        return

    sender_email = st.secrets["email"]["sender"]
    sender_password = st.secrets["email"]["password"]

    if forecast_summary:
        subject = f"📬 3-Day Forecast Summary for {location}"
        message = f"""Hello Team,

Here is the 3-day weather and work suitability forecast for your location:


{forecast_summary}

Regards,
Weather Forecast System
"""
    else:
        subject = f"Weather Work Suitability Alert - {location}"
        message = f"""Hello Team,

Here is the real-time weather and work suitability report for your location:

📍 Location: {location}
🌤️ Weather: {weather_label}
✅ Work Suitability: {"Suitable" if work_suitability else "Not Suitable"}

Regards,
Weather Forecast System
"""

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(message, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        st.success(f"📩 Email sent to {receiver_email}")
    except Exception as e:
        st.error(f"Email sending failed: {e}")

def get_forecast_3day(lat, lon, key):
    url = "https://api.tomorrow.io/v4/timelines"
    params = {
        "location": f"{lat},{lon}",
        "fields": ["temperature", "humidity", "precipitationIntensity", "windSpeed"],
        "units": "metric",
        "timesteps": "1d",  # daily forecast
        "startTime": "nowPlus1d",
        "endTime": "nowPlus3d"
    }
    headers = {"apikey": key}
    r = requests.get(url, params=params, headers=headers, timeout=20)
    if r.status_code != 200:
        st.error("❌ 3‑Day Forecast API error")
        return None
    return r.json()["data"]["timelines"][0]["intervals"]

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
            rt["Temperature_C"] < 5 or rt["Temperature_C"] > 50 or
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

        send_auto_email(city_sel, w_label, rt_pred)

        if st.session_state.manual_strict and strict_rt:
            st.warning("🚨 Harsh weather detected in BOTH manual and real‑time inputs!")

st.markdown("### 📅 3‑Day Forecast & Suitability")

if st.button("🔮 Get 3‑Day Forecast & Predict"):
    st.markdown(f"📅 **Today:** {datetime.now().date().strftime('%A, %d %B %Y')}")
    lat, lon = city_coords[city_sel]
    forecast_data = get_forecast_3day(lat, lon, api_key)

    if forecast_data:
        st.session_state["forecast_data"] = forecast_data
        cols = st.columns(3)

        for i, (day, col) in enumerate(zip(forecast_data, cols), start=1):
            with col:
                v = day["values"]
                day_data = {
                    "Temperature_C": v["temperature"],
                    "Humidity_pct": v["humidity"],
                    "Precipitation_mm": v.get("precipitationIntensity", 0.0),
                    "Wind_Speed_kmh": v["windSpeed"]
                }
                dt = day["startTime"][:10]

                st.markdown(f"### 📆 Day {i}<br><small>{dt}</small>", unsafe_allow_html=True)
                st.write(day_data)

                df_input = pd.DataFrame([day_data])
                w_code = weather_model.predict(df_input)[0]
                w_label = le_desc.inverse_transform([w_code])[0]
                st.info(f"🌤️ **{w_label}**")

                df_input["Weather_Description"] = w_label
                strict = (
                    day_data["Temperature_C"] < 5 or day_data["Temperature_C"] > 50 or
                    day_data["Precipitation_mm"] > 8 or day_data["Wind_Speed_kmh"] > 30 or
                    w_label.lower() in ["heavy rainfall", "flood", "storm", "cyclone"]
                )
                if strict:
                    st.warning("⚠️ Not Suitable")
                    suit_pred = 0
                else:
                    suit_pred = work_model.predict(df_input)[0]

                if suit_pred == 1:
                    st.success("✅ Suitable")
                else:
                    st.error("❌ Not Suitable")

        # ✅ Build summary for auto email
        summary = f"\n📍 Location: {city_sel}\n\n"

        for i, day in enumerate(forecast_data, 1):
            values = day["values"]
            date = day["startTime"][:10]
            data = {
                "Temperature_C": values["temperature"],
                "Humidity_pct": values["humidity"],
                "Precipitation_mm": values.get("precipitationIntensity", 0.0),
                "Wind_Speed_kmh": values["windSpeed"]
            }

            df_input = pd.DataFrame([data])
            w_code = weather_model.predict(df_input)[0]
            w_label = le_desc.inverse_transform([w_code])[0]
            df_input["Weather_Description"] = w_label

            strict = (
                data["Temperature_C"] < 5 or data["Temperature_C"] > 50 or
                data["Precipitation_mm"] > 8 or data["Wind_Speed_kmh"] > 30 or
                w_label.lower() in ["heavy rainfall", "flood", "storm", "cyclone"]
            )

            suit = "Not Suitable" if strict or work_model.predict(df_input)[0] == 0 else "Suitable"
            summary += f"Day {i} ({date}):\n🌤 Weather: {w_label}\n{'✅' if suit == 'Suitable' else '❌'} Work Suitability: {suit}\n\n"


        send_auto_email(city_sel, forecast_summary=summary)

st.markdown("### 📤 Send Report via Email")
email = st.text_input("Enter recipient email")

if st.button("📩 Send Email Report"):
    if not email or "@" not in email:
        st.error("Please enter a valid email address.")
    else:
        # Retrieve cached forecast or fetch a fresh one
        data_for_pdf = st.session_state.get("forecast_data")
        if data_for_pdf is None:
            lat, lon = city_coords[city_sel]
            data_for_pdf = get_forecast_3day(lat, lon, api_key)
            if data_for_pdf is None:
                st.error("❌ Could not fetch forecast data.")
                st.stop()

        with st.spinner("Sending email…"):
            success = send_email_report(email, data_for_pdf, city_sel)
            if success:
                st.success("✅ Email sent successfully!")


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