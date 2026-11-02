from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.predict import load_model, predict_message

MODEL_PATH = Path(__file__).resolve().parent / "models" / "spam_detector.joblib"

st.set_page_config(
    page_title="Email & SMS Spam Detector",
    page_icon="📩",
    layout="centered",
)

st.title("📩 Email & SMS Spam Detector")
st.write("Check whether a message looks like spam using a TF-IDF and Logistic Regression model.")

@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


try:
    model = get_model()
except (FileNotFoundError, RuntimeError, ValueError, TypeError) as exc:
    st.warning(str(exc))
    st.code("python src/train.py", language="bash")
    st.stop()

sample = st.selectbox(
    "Try an example (optional)",
    [
        "Write or paste your own message",
        "Hi, are we still meeting at the library at 5?",
        "Congratulations! You won a free prize. Click the link to claim now!",
    ],
)
default_message = "" if sample == "Write or paste your own message" else sample
message = st.text_area(
    "Message to check",
    value=default_message,
    height=160,
    placeholder="Paste an SMS or email message here...",
)

if st.button("Check message", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        try:
            result = predict_message(message, model=model)
        except (ValueError, TypeError) as exc:
            st.error(str(exc))
        else:
            if result["label"] == "Spam":
                st.error(f"🚨 Likely spam — {result['confidence']:.1%} confidence")
            else:
                st.success(f"✅ Likely legitimate (ham) — {result['confidence']:.1%} confidence")

            st.subheader("Model scores")
            left, right = st.columns(2)
            left.metric("Ham", f"{result['probabilities']['ham']:.1%}")
            right.metric("Spam", f"{result['probabilities']['spam']:.1%}")
            st.caption("Scores are model estimates, not a guarantee. Avoid opening links in suspicious messages.")

st.divider()
st.caption("Educational machine-learning demonstration • TF-IDF + Logistic Regression")
