import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
import os
import sys
import nltk

from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import clean_text, get_latest_market_window

# ------------------------------------------------
# Fix keras / tf.keras pickle compatibility
# ------------------------------------------------
import tensorflow.keras as keras
sys.modules['keras'] = keras

# ------------------------------------------------
# NLTK setup
# ------------------------------------------------
nltk.download("punkt")
nltk.download("stopwords")

st.title("Financial Market Intelligence AI")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------
# Load model + assets
# ------------------------------------------------
@st.cache_resource
def load_assets():

    model_path = os.path.join(BASE_DIR, "market_model_saved")

    tokenizer_path = os.path.join(BASE_DIR, "tokenizer.pkl")
    lda_path = os.path.join(BASE_DIR, "lda_model.pkl")
    vec_path = os.path.join(BASE_DIR, "lda_vectorizer.pkl")

    model = tf.keras.models.load_model(model_path)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    with open(lda_path, "rb") as f:
        lda = pickle.load(f)

    with open(vec_path, "rb") as f:
        vec = pickle.load(f)

    return model, tokenizer, lda, vec


model, tokenizer, lda, vec = load_assets()

# ------------------------------------------------
# UI
# ------------------------------------------------
headline = st.text_input("Enter financial news headline")

if st.button("Predict Market"):

    cleaned = clean_text(headline)

    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=100)

    topic_vec = lda.transform(vec.transform([cleaned]))

    ts_window = get_latest_market_window()

    reg, cls = model.predict(
        {
            "ts_input": ts_window,
            "text_input": padded,
            "topic_input": topic_vec
        }
    )

    price = float(reg[0][0])
    prob = float(cls[0][0])

    direction = "UP" if prob > 0.5 else "DOWN"

    st.metric("Market Direction", direction)
    st.metric("Confidence", f"{prob:.2f}")
    st.metric("Predicted Price", f"${price:.2f}")
