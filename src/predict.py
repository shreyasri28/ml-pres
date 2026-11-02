from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib

from src.preprocess import preprocess_text

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "spam_detector.joblib"


def load_model(model_path: str | Path = MODEL_PATH):
    """Load a serialized spam detection pipeline."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at: {path}. Train the model first.")

    try:
        model = joblib.load(path)
        if hasattr(model, "named_steps"):
            classifier = model.named_steps.get("classifier")
            if classifier is None or not hasattr(classifier, "predict_proba"):
                raise TypeError("Saved model is missing a compatible classifier.")
        elif not hasattr(model, "predict_proba"):
            raise TypeError("Saved model does not expose predict_proba().")
        return model
    except Exception as exc:
        try:
            from src.train import train_model

            retrained = train_model()
            return retrained["model"]
        except Exception:
            raise RuntimeError(
                f"Unable to load a compatible model from {path}. Please retrain the model with: python src/train.py --data data/sms_spam_sample.csv"
            ) from exc


def predict_message(message: str, model=None):
    """Predict whether the given message is spam or ham."""
    if model is None:
        model = load_model()

    cleaned = preprocess_text(message)
    prediction = model.predict([cleaned])[0]
    probabilities = model.predict_proba([cleaned])[0]

    label = "Spam" if prediction == 1 else "Ham"
    confidence = max(probabilities)

    return {
        "label": label,
        "confidence": float(confidence),
        "probabilities": {
            "ham": float(probabilities[0]),
            "spam": float(probabilities[1]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict if a message is spam or ham.")
    parser.add_argument("message", nargs="?", default="", help="Message text to classify.")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="Path to the serialized model file.")
    args = parser.parse_args()

    if not args.message:
        print("Please provide a message to classify.", file=sys.stderr)
        raise SystemExit(1)

    try:
        model = load_model(args.model)
        result = predict_message(args.message, model=model)
    except Exception as exc:
        print(f"Prediction failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Prediction: {result['label']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Ham probability: {result['probabilities']['ham']:.4f}")
    print(f"Spam probability: {result['probabilities']['spam']:.4f}")


if __name__ == "__main__":
    main()
