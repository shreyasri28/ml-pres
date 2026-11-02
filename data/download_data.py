from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"


def download_dataset() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "smsspamcollection.zip"
    dataset_path = DATA_DIR / "SMSSpamCollection"

    if dataset_path.exists():
        print(f"Dataset already exists at: {dataset_path}")
        return dataset_path

    print(f"Downloading dataset from {ZIP_URL}...")
    urlretrieve(ZIP_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)

    print(f"Dataset downloaded to: {dataset_path}")
    return dataset_path


if __name__ == "__main__":
    try:
        download_dataset()
    except Exception as exc:  # pragma: no cover
        print(f"Failed to download dataset: {exc}", file=sys.stderr)
        raise SystemExit(1)
