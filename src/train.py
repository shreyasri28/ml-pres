from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import preprocess_series

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "spam_detector.joblib"


def load_dataset(data_path: str | Path | None = None) -> pd.DataFrame:
    """Load the SMS dataset, falling back to bundled sample data if needed."""
    if data_path is None:
        candidate_paths = [
            DATA_DIR / "SMSSpamCollection",
            DATA_DIR / "SMSSpamCollection.csv",
            DATA_DIR / "sms_spam_sample.csv",
        ]
        for path in candidate_paths:
            if path.exists():
                data_path = path
                break

    if data_path is None:
        raise FileNotFoundError(
            "No dataset found. Place a CSV or text file with columns ['label', 'text'] in the data/ folder."
        )

    data_path = Path(data_path)
    if data_path.suffix.lower() == ".csv":
        df = pd.read_csv(data_path, encoding="utf-8")
    else:
        # Standard SMS Spam Collection format: tab-separated, with no header
        df = pd.read_csv(data_path, sep="\t", header=None, names=["label", "text"], encoding="utf-8")

    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError("Dataset must contain 'label' and 'text' columns.")

    df = df[["label", "text"]].dropna().copy()
    df["label"] = df["label"].str.strip().str.lower()
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    return df


def build_model() -> Pipeline:
    """Return a complete text classification pipeline."""
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
            )),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="liblinear",
                multi_class="ovr",
            )),
        ]
    )


def train_model(data_path: str | Path | None = None) -> dict:
    """Train and evaluate the spam detector model."""
    df = load_dataset(data_path)

    if df.empty:
        raise ValueError("Dataset is empty after cleaning.")

    X = preprocess_series(df["text"].tolist())
    y = df["label"].astype(int).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_model()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "predict_proba"):
        raise TypeError("Trained classifier does not support probability output.")
    if not hasattr(classifier, "multi_class"):
        raise AttributeError("Classifier is missing multi_class metadata.")

    # Validate the trained pipeline before saving it.
    test_vector = model.named_steps["tfidf"].transform([X_train[0]])
    _ = classifier.predict_proba(test_vector)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "classification_report": classification_report(y_test, y_pred, target_names=["ham", "spam"]),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return {
        "model": model,
        "metrics": metrics,
        "model_path": str(MODEL_PATH),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a SMS/Email spam detection model.")
    parser.add_argument("--data", type=str, default=None, help="Path to dataset CSV or tab-delimited file.")
    args = parser.parse_args()

    try:
        result = train_model(args.data)
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Model trained successfully and saved to: {result['model_path']}")
    print(f"Accuracy: {result['metrics']['accuracy']:.4f}")
    print(f"Precision: {result['metrics']['precision']:.4f}")
    print(f"Recall: {result['metrics']['recall']:.4f}")
    print(f"F1: {result['metrics']['f1_score']:.4f}")
    print("Confusion matrix:")
    print(result['metrics']['confusion_matrix'])
    print("\nClassification report:\n")
    print(result['metrics']['classification_report'])


if __name__ == "__main__":
    main()
