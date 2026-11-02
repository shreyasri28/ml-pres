from __future__ import annotations

import re
from typing import Iterable

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Clean raw text for spam detection."""
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text: str) -> list[str]:
    """Tokenize and lemmatize text while removing stop words."""
    tokens = word_tokenize(text)
    filtered = []
    for token in tokens:
        if token in STOP_WORDS:
            continue
        lemma = LEMMATIZER.lemmatize(token)
        if lemma:
            filtered.append(lemma)
    return filtered


def preprocess_text(text: str) -> str:
    """Run the full text preprocessing pipeline and return a normalized string."""
    cleaned = clean_text(text)
    tokens = tokenize_and_lemmatize(cleaned)
    return " ".join(tokens)


def preprocess_series(texts: Iterable[str]) -> list[str]:
    """Apply preprocessing to a sequence of strings."""
    return [preprocess_text(text) for text in texts]
