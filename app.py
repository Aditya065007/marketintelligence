import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
import os
import nltk

from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import clean_text, get_latest_market_window

nltk.download("punkt")
nltk.download("stopwords")

st.title("Financial Market Intelligence AI")

# ---------------------------------------
# Base directory (ensures Streamlit finds files)
# ---------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------
# Load model and assets
# ---------------------------------------
@st.cache_resource
def load_assets():

    model = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "market_model_saved")
    )

    tokenizer = pickle.load(open(
        os.path.join(BASE_DIR, "tokenizer.pkl"), "rb"))

    lda = pickle.load(open(
        os.path.join(BASE_DIR, "lda_model.pkl"), "rb"))

    vec = pickle.load(open(
        os.path.join(BASE_DIR, "lda_vectorizer.pkl"), "rb"))

    return model, tokenizer, lda, vec


model, tokenizer, lda, vec = load_assets()

# ---------------------------------------
# User input
# ---------------------------------------
headline = st.text_input("Enter financial news headline")

# ---------------------------------------
# Prediction
# ---------------------------------------
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
