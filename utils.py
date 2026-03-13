import re
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# -------------------------
# TEXT CLEANING
# -------------------------
def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# -------------------------
# FETCH LATEST MARKET WINDOW
# -------------------------
def get_latest_market_window():

    df = yf.download("^GSPC", period="10d", progress=False)

    df = df[["Open","High","Low","Close","Volume"]]

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(df)

    window = scaled[-5:]

    window = np.expand_dims(window, axis=0)

    return window
