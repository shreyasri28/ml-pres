from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib

from src.preprocess import preprocess_text

MODEL_PATH = PROJECT_ROOT / "models" / "spam_detector.joblib"


def load_model(model_path: str | Path = MODEL_PATH):
    """Load a trained model and fail with an actionable message if unavailable."""
    path = Path(model_path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Trained model is missing or empty: {path}. "
            "Train it first with: python src/train.py"
        )

    try:
        model = joblib.load(path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read model file: {path}. Retrain it with: python src/train.py"
        ) from exc

    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        raise TypeError("The saved model must support predict() and predict_proba().")
    if not hasattr(model, "classes_") or set(model.classes_) != {0, 1}:
        raise ValueError("The saved model does not contain both ham (0) and spam (1) classes.")
    return model


def predict_message(message: str, model=None) -> dict:
    """Classify one message and return its label and class probabilities."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Please provide a non-empty message.")

    if model is None:
        model = load_model()

    cleaned = preprocess_text(message)
    probabilities = model.predict_proba([cleaned])[0]
    class_probabilities = {
        int(class_label): float(probability)
        for class_label, probability in zip(model.classes_, probabilities)
    }
    predicted_class = int(model.predict([cleaned])[0])
    label = "Spam" if predicted_class == 1 else "Ham"

    return {
        "label": label,
        "confidence": class_probabilities[predicted_class],
        "probabilities": {
            "ham": class_probabilities.get(0, 0.0),
            "spam": class_probabilities.get(1, 0.0),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict whether a message is spam or ham.")
    parser.add_argument("message", nargs="+", help="Message text to classify.")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="Path to model file.")
    args = parser.parse_args()

    try:
        model = load_model(args.model)
        result = predict_message(" ".join(args.message), model=model)
    except Exception as exc:
        print(f"Prediction failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Prediction: {result['label']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Ham probability: {result['probabilities']['ham']:.4f}")
    print(f"Spam probability: {result['probabilities']['spam']:.4f}")


if __name__ == "__main__":
    main()
