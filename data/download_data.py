from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATASET_NAME = "SMSSpamCollection"


def download_dataset() -> Path:
    """Download and safely extract the UCI SMS Spam Collection dataset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = DATA_DIR / DATASET_NAME
    zip_path = DATA_DIR / "smsspamcollection.zip"

    if dataset_path.is_file() and dataset_path.stat().st_size > 0:
        print(f"Dataset already exists: {dataset_path}")
        return dataset_path

    print("Downloading the UCI SMS Spam Collection dataset...")
    urlretrieve(ZIP_URL, zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        member = next(
            (name for name in archive.namelist()
             if Path(name).name == DATASET_NAME),
            None,
        )
        if member is None:
            raise FileNotFoundError(f"{DATASET_NAME} was not found in the downloaded archive.")
        # Read the known dataset member directly instead of extracting arbitrary archive paths.
        dataset_path.write_bytes(archive.read(member))

    if dataset_path.stat().st_size == 0:
        raise ValueError("The downloaded dataset is empty.")

    print(f"Dataset ready: {dataset_path}")
    return dataset_path


if __name__ == "__main__":
    try:
        download_dataset()
    except Exception as exc:
        print(f"Dataset download failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
