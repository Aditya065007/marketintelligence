import streamlit as st
import tensorflow as tf
import pickle
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import clean_text, get_latest_market_window

st.title("Financial Market Intelligence AI")

@st.cache_resource
def load_assets():

    model = tf.keras.models.load_model("model_assets/market_model.keras")

    tokenizer = pickle.load(open("model_assets/tokenizer.pkl","rb"))
    lda = pickle.load(open("model_assets/lda_model.pkl","rb"))
    vec = pickle.load(open("model_assets/lda_vectorizer.pkl","rb"))

    return model, tokenizer, lda, vec

model, tokenizer, lda, vec = load_assets()

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