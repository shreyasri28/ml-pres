from __future__ import annotations

import re
from collections.abc import Iterable

# Keep negation words: removing "not", "no", and "nor" can reverse a message's meaning.
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "could", "did",
    "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "itself", "just", "me", "more", "most", "my", "myself",
    "of", "off", "on", "once", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "same", "she", "should", "so", "some",
    "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "whom", "why", "will", "with", "would",
    "you", "your", "yours", "yourself", "yourselves",
}


def clean_text(text: str) -> str:
    """Normalize a message without requiring downloads or external corpora."""
    if text is None:
        return ""
    value = str(text).lower()
    value = re.sub(r"https?://\S+|www\.\S+", " url ", value)
    value = re.sub(r"[^a-z\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def tokenize_and_lemmatize(text: str) -> list[str]:
    """Tokenize text and remove common stop words.

    Kept under the historical function name for compatibility; no external
    lemmatizer or downloaded language corpus is required.
    """
    return [token for token in re.findall(r"[a-z]+", text.lower())
            if token not in STOP_WORDS]


def preprocess_text(text: str) -> str:
    """Return normalized tokens for model training and inference."""
    return " ".join(tokenize_and_lemmatize(clean_text(text)))


def preprocess_series(texts: Iterable[str]) -> list[str]:
    """Preprocess a sequence of message strings."""
    return [preprocess_text(text) for text in texts]
