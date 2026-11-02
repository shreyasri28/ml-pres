from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.predict import load_model, predict_message

MODEL_PATH = Path(__file__).resolve().parent / "models" / "spam_detector.joblib"

st.set_page_config(page_title="Spam Detection System", page_icon="📧", layout="centered")


@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Please train the model first by running: python src/train.py --data data/sms_spam_sample.csv"
        )
    return load_model(MODEL_PATH)


st.title("Email/SMS Spam Detection System")
st.subheader("NLP-powered spam classifier")

try:
    model = get_model()
except FileNotFoundError as exc:
    st.warning(str(exc))
    st.info("Use the terminal to train the model before launching predictions.")
    st.stop()

message = st.text_area(
    "Enter a message or email to classify:",
    height=180,
    placeholder="Example: Congratulations! You've won a free prize. Claim now!",
)

if st.button("Classify Message", type="primary"):
    if not message.strip():
        st.warning("Please enter some text before classifying.")
    else:
        result = predict_message(message, model=model)
        label = result["label"]
        confidence = result["confidence"]
        ham_prob = result["probabilities"]["ham"]
        spam_prob = result["probabilities"]["spam"]

        if label == "Spam":
            st.error(f"Prediction: {label} ({confidence:.2%} confidence)")
        else:
            st.success(f"Prediction: {label} ({confidence:.2%} confidence)")

        st.write("Probability breakdown:")
        st.progress(int(confidence * 100))
        st.write(f"Ham probability: {ham_prob:.2%}")
        st.write(f"Spam probability: {spam_prob:.2%}")

st.caption("This model uses TF-IDF + Logistic Regression for robust spam detection.")
