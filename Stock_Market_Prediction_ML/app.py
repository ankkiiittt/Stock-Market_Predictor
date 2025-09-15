import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ======================
# Page Configuration
# ======================
st.set_page_config(
    page_title="Stock Market Predictor",
    page_icon="📊",
    layout="wide"
)

# ======================
# Sidebar Inputs
# ======================
st.sidebar.header("⚙️ Settings")

theme = st.sidebar.radio("🎨 Theme", ["☀️ Light", "🌙 Dark"])
stock = st.sidebar.text_input("Enter Stock Symbol", "GOOG")
start = st.sidebar.date_input("📅 Start Date", pd.to_datetime("2012-01-01"))
end = st.sidebar.date_input("📅 End Date", pd.to_datetime("2022-12-31"))
predict_button = st.sidebar.button("🚀 Run Prediction")

# ======================
# Theme Styles
# ======================
def apply_theme(theme):
    if theme == "🌙 Dark":
        st.markdown("""
            <style>
            .stApp { background-color: #1E1E1E; color: #E0E0E0; }
            h1, h2, h3, h4 { color: #F5F5F5; }
            .css-18e3th9, .css-1d391kg {
                background: #2C2C2C;
                color: #F5F5F5;
                border-radius: 10px;
                box-shadow: 0px 4px 10px rgba(0,0,0,0.4);
            }
            </style>
        """, unsafe_allow_html=True)
        plt.style.use("dark_background")
    else:
        st.markdown("""
            <style>
            .stApp { background-color: #f9fafc; color: #2C3E50; }
            h1, h2, h3, h4 { color: #2C3E50; }
            .css-18e3th9, .css-1d391kg {
                background: #ffffff;
                border-radius: 10px;
                box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
            }
            </style>
        """, unsafe_allow_html=True)
        plt.style.use("default")

apply_theme(theme)

# ======================
# App Header
# ======================
st.title("📊 Stock Market Prediction Dashboard")
st.markdown("Predict stock prices using a trained deep learning model.")

# ======================
# Load Model
# ======================
@st.cache_resource
def load_trained_model():
    return load_model("C:\\Users\\Asus\\Downloads\\Stock_Market_Prediction_ML\\model.keras")

model = load_trained_model()

# ======================
# Fetch Stock Data
# ======================
@st.cache_data
def load_stock_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

if predict_button:
    data = load_stock_data(stock, start, end)

    st.subheader(f"📈 Latest Stock Data for **{stock}**")
    st.dataframe(data.tail(10).style.highlight_max(axis=0, color="lightgreen"))

    # Train / Test Split
    data_train = pd.DataFrame(data.Close[0:int(len(data) * 0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):])

    # Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data_train)

    past_100_days = data_train.tail(100)
    data_test = pd.concat([past_100_days, data_test], ignore_index=True)
    data_test_scale = scaler.transform(data_test)

    # ======================
    # Moving Average Charts
    # ======================
    st.subheader("📊 Price Trends with Moving Averages")

    ma_50 = data.Close.rolling(50).mean()
    ma_100 = data.Close.rolling(100).mean()
    ma_200 = data.Close.rolling(200).mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.Close, label="Closing Price", color="#2E86C1", linewidth=2)
    ax.plot(ma_50, label="MA50", color="#E74C3C", linestyle="--", linewidth=1.5)
    ax.plot(ma_100, label="MA100", color="#27AE60", linestyle="--", linewidth=1.5)
    ax.plot(ma_200, label="MA200", color="#F39C12", linestyle="--", linewidth=1.5)
    ax.set_title(f"{stock} Price with Moving Averages", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend(frameon=True, facecolor="white", shadow=True)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # ======================
    # Prepare Data for Prediction
    # ======================
    x_test, y_test = [], []
    for i in range(100, data_test_scale.shape[0]):
        x_test.append(data_test_scale[i - 100:i])
        y_test.append(data_test_scale[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    # Predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # ======================
    # Prediction Plot
    # ======================
    st.subheader("🔮 Predicted vs Original Stock Price")

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(y_test, label="Original Price", color="#27AE60", linewidth=2)
    ax2.plot(predictions, label="Predicted Price", color="#E74C3C", linewidth=2)
    ax2.set_title(f"{stock} Prediction Results", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Price (USD)")
    ax2.legend(frameon=True, facecolor="white", shadow=True)
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

    st.success("✅ Prediction Complete!")

else:
    st.info("👉 Enter stock details in the sidebar and click **Run Prediction** to start.")
