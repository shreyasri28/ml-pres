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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import preprocess_series

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "spam_detector.joblib"


def load_dataset(data_path: str | Path | None = None) -> pd.DataFrame:
    """Load a CSV (label,text) or the tab-separated UCI SMS dataset."""
    if data_path is None:
        candidates = [
            DATA_DIR / "SMSSpamCollection",
            DATA_DIR / "SMSSpamCollection.csv",
            DATA_DIR / "sms_spam_sample.csv",
        ]
        data_path = next((path for path in candidates if path.exists()), None)

    if data_path is None:
        raise FileNotFoundError(
            "No dataset found. Run 'python data/download_data.py' or provide --data."
        )

    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, encoding="utf-8-sig")
    else:
        df = pd.read_csv(
            path, sep="\t", header=None, names=["label", "text"],
            encoding="utf-8-sig",
        )

    if not {"label", "text"}.issubset(df.columns):
        raise ValueError("Dataset must contain columns named 'label' and 'text'.")

    df = df[["label", "text"]].dropna().copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].ne("")]
    df = df[df["label"].isin(["ham", "spam"])]
    df["label"] = df["label"].map({"ham": 0, "spam": 1}).astype(int)
    df = df.drop_duplicates(subset=["label", "text"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("Dataset has no usable ham/spam rows.")
    if df["label"].nunique() != 2:
        raise ValueError("Dataset must contain both ham and spam messages.")
    return df


def build_model() -> Pipeline:
    """Create a complete TF-IDF + Logistic Regression pipeline."""
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                max_df=1.0,
                sublinear_tf=True,
                strip_accents="unicode",
                max_features=50000,
            )),
            ("classifier", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear",
                random_state=42,
            )),
        ]
    )


def train_model(data_path: str | Path | None = None) -> dict:
    """Train, evaluate, and save the spam detector."""
    df = load_dataset(data_path)
    X = preprocess_series(df["text"].tolist())
    y = df["label"].tolist()

    class_counts = pd.Series(y).value_counts()
    if class_counts.min() < 2:
        raise ValueError(
            "At least two examples of each class are required for a stratified split. "
            "Use the full SMS Spam Collection dataset."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=labels).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        },
        "classification_report": classification_report(
            y_test, y_pred, labels=labels, target_names=["ham", "spam"],
            zero_division=0,
        ),
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return {"model": model, "metrics": metrics, "model_path": str(MODEL_PATH)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SMS/email spam detector.")
    parser.add_argument("--data", type=str, default=None, help="CSV or tab-separated dataset path.")
    args = parser.parse_args()

    try:
        result = train_model(args.data)
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    metrics = result["metrics"]
    print(f"Model saved to: {result['model_path']}")
    print(f"Dataset rows: {metrics['dataset_rows']}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1_score']:.4f}")
    print("Confusion matrix (true labels: ham, spam):")
    print(metrics["confusion_matrix"])
    print("\nClassification report:\n")
    print(metrics["classification_report"])


if __name__ == "__main__":
    main()
