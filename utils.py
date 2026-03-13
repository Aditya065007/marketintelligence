import re
import numpy as np
import yfinance as yf

def clean_text(text):

    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def get_latest_market_window(lookback=5):

    df = yf.download("^GSPC", period="10d")

    cols = ["Open","High","Low","Close","Volume"]
    data = df[cols].values

    return np.expand_dims(data[-lookback:], axis=0)