import requests
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
import warnings

# ----------------------------------------------------------
# 🔧 CONFIGURATION
# ----------------------------------------------------------
warnings.filterwarnings("ignore")  # hide minor sklearn warnings

# <-- Replace with your real API key
API_KEY = "1314dd76846285b8137e47d9ac8e4786"
CITY = "Nashik,IN"
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

# ----------------------------------------------------------
# 🌤 FUNCTION: Fetch live weather
# ----------------------------------------------------------


def get_weather(city):
    """Fetch live weather data from OpenWeatherMap API."""
    params = {"q": city, "appid": API_KEY, "units": "metric"}
    try:
        r = requests.get(WEATHER_URL, params=params, timeout=10)
        data = r.json()
        if r.status_code == 200:
            return {
                "Temperature (°C)": data["main"]["temp"],
                "Humidity (%)": data["main"]["humidity"],
                "Pressure (hPa)": data["main"]["pressure"],
                "Description": data["weather"][0]["description"].capitalize(),
            }
        else:
            st.error(
                f"Error fetching weather: {data.get('message', 'Unknown error')}")
            return None
    except requests.RequestException as e:
        st.error(f"Network error: {e}")
        return None


# ----------------------------------------------------------
# 🧠 FUNCTION: Train simple yield prediction model
# ----------------------------------------------------------
def train_yield_model():
    """Train a sample Linear Regression model using dummy farm data."""
    df = pd.DataFrame({
        "temp": [20, 25, 30, 35, 40],
        "humidity": [50, 60, 70, 80, 90],
        "moisture": [20, 40, 60, 80, 100],
        "pH": [5.5, 6.0, 6.5, 7.0, 7.5],
        "yield": [1.5, 2.0, 2.3, 2.1, 1.8]  # tons per acre
    })
    X = df[["temp", "humidity", "moisture", "pH"]]
    y = df["yield"]
    model = LinearRegression().fit(X, y)
    return model


# ----------------------------------------------------------
# 🌾 FUNCTION: Crop suggestion based on pH & temperature
# ----------------------------------------------------------
def suggest_crops(pH, temp):
    """Return crop suggestions based on pH and temperature ranges."""
    if pH < 6:
        return ["Rice", "Potato", "Maize"]
    elif 6 <= pH <= 7:
        if temp < 25:
            return ["Wheat", "Barley", "Soybean"]
        else:
            return ["Sugarcane", "Corn", "Sunflower"]
    else:
        return ["Cotton", "Sorghum", "Groundnut"]


# ----------------------------------------------------------
# 🚀 STREAMLIT DASHBOARD
# ----------------------------------------------------------
st.set_page_config(page_title="Smart Farming Dashboard",
                   page_icon="🌾", layout="centered")

st.title("🌾 Smart Farming Assistant")
st.write("Analyze **live weather**, input **soil data**, and get **yield predictions + crop suggestions** using machine learning.")

# --- Fetch and display live weather ---
st.divider()
st.subheader("☀ Live Weather Data")

weather_data = get_weather(CITY)
if weather_data:
    st.json(weather_data)
else:
    st.warning(
        "Unable to fetch live weather data. Please check API key or internet connection.")

# --- Soil data input ---
st.divider()
st.subheader("🧪 Soil Data Input")
moisture = st.slider("Soil Moisture (%)", 0, 100, 50)
ph = st.slider("Soil pH", 4.0, 9.0, 6.5)

# --- Train yield model ---
model = train_yield_model()

# --- Predict yield if weather available ---
if weather_data:
    temp = weather_data["Temperature (°C)"]
    humidity = weather_data["Humidity (%)"]

    # ✅ Use DataFrame instead of NumPy array to keep feature names (no warning)
    X_new = pd.DataFrame([{
        "temp": temp,
        "humidity": humidity,
        "moisture": moisture,
        "pH": ph
    }])

    predicted_yield = model.predict(X_new)[0]

    st.divider()
    st.subheader("🌱 Predicted Crop Yield")
    st.success(f"Estimated yield: **{predicted_yield:.2f} tons/acre**")

    # --- Crop suggestions ---
    st.subheader("🌾 Suggested Crops for Current Conditions")
    suggestions = suggest_crops(ph, temp)
    st.write(", ".join(suggestions))

# --- Footer ---
st.divider()
st.caption(
    "Developed by 🌱 Smart Farming Project | Powered by OpenWeatherMap & Machine Learning")
