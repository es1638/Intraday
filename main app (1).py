
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from datetime import datetime, timedelta
import lightgbm as lgb

# Set Streamlit page config
st.set_page_config(layout="wide")
st.title("📉 Intraday Breakout Prediction Dashboard")

# Load the LightGBM model
try:
    model = joblib.load("lightgbm_model_converted.pkl")
except Exception as e:
    st.error(f"Failed to load LightGBM model: {e}")
    st.stop()

# Daily screening conditions with detailed logging
def passes_screening(ticker):
    try:
        hist = yf.download(ticker, period="1y", interval="1d", progress=False)
        if hist.empty or "Close" not in hist.columns or "Volume" not in hist.columns or "High" not in hist.columns:
            st.info(f"{ticker}: Missing required historical columns.")
            return False

        avg_volume = hist["Volume"].tail(30).mean()
        if pd.isna(avg_volume) or avg_volume < 10_000_000:
            st.info(f"{ticker}: Avg volume too low or NaN.")
            return False

        high_52w = hist["High"].rolling(window=252).max().iloc[-1]
        current_price = hist["Close"].iloc[-1]

        if pd.isna(high_52w) or pd.isna(current_price):
            st.info(f"{ticker}: Missing current price or 52w high.")
            return False

        high_52w = float(high_52w)
        current_price = float(current_price)

        if current_price < 0.6 * high_52w:
            st.info(f"{ticker}: Price {current_price} < 60% of 52w high {high_52w}")
            return False

        return True
    except Exception as e:
        st.warning(f"⚠️ Error with {ticker}: {e}")
        return False

@st.cache_data(show_spinner=False)
def get_sp500_tickers():
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    return table['Symbol'].tolist()

@st.cache_data(show_spinner=False)
def get_screened_tickers():
    tickers = get_sp500_tickers()
    screened = []
    for ticker in tickers:
        if passes_screening(ticker):
            screened.append(ticker)
    return screened

# Live feature engineering
def get_live_features(ticker):
    data = yf.download(ticker, period="2d", interval="1m")
    if data.empty:
        raise ValueError("No intraday data available")
    data.index = pd.to_datetime(data.index)
    volume_series = data["Volume"]
    data["momentum_10min"] = data["Close"].pct_change(periods=10)
    data["price_change_5min"] = data["Close"].pct_change(periods=5)
    data["rolling_volume"] = volume_series.rolling(window=5).mean()
    data["rolling_volume_ratio"] = volume_series / data["rolling_volume"]
    features = data[["momentum_10min", "price_change_5min", "rolling_volume", "rolling_volume_ratio"]]
    return features.dropna().iloc[-1:]

# Load screened tickers
if "screened_tickers" not in st.session_state:
    st.session_state.screened_tickers = []

if st.button("🦁 Refresh Daily Screen"):
    with st.spinner("Running daily screener..."):
        st.session_state.screened_tickers = get_screened_tickers()
        st.success(f"Screened {len(st.session_state.screened_tickers)} tickers.")

# Buy signal threshold input
threshold = st.slider("Buy Signal Threshold", 0.90, 1.00, 0.98, step=0.01)

# Evaluation output
if st.session_state.screened_tickers:
    results = []
    for ticker in st.session_state.screened_tickers:
        try:
            X = get_live_features(ticker)
            if X.empty:
                raise ValueError("No intraday data available")

            # Check if model has predict_proba (e.g., classification model)
            # Otherwise use predict() directly (e.g., regression model)
            prob = model.predict(X).item()       # Regression prediction

            results.append({
                "Ticker": ticker,
                "Buy Signal": "✅ Buy" if prob >= threshold else "❌ No",
                "Probability": "N/A"
            })
        except Exception as e:
            results.append({
                "Ticker": ticker,
                "Buy Signal": "⚠️ Error",
                "Probability": str(e)
            })

    df_results = pd.DataFrame(results)
    st.dataframe(df_results)
else:
    st.info("Please run the daily screen to populate tickers.")

