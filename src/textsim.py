"""Dependency-free TF-IDF vectors (char n-gram + word) and cosine similarity."""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

from .normalize import normalize_text

Vector = dict[str, float]


def char_ngrams(text: str, n_min: int = 3, n_max: int = 5) -> Counter:
    grams: Counter = Counter()
    for tok in normalize_text(text).split():
        padded = f" {tok} "
        for n in range(n_min, n_max + 1):
            for i in range(len(padded) - n + 1):
                grams[padded[i:i + n]] += 1
    return grams


def word_terms(text: str) -> Counter:
    return Counter(normalize_text(text).split())


def idf_from(docs: Iterable[Counter]) -> dict[str, float]:
    docs = list(docs)
    n = len(docs)
    df: Counter = Counter()
    for d in docs:
        df.update(d.keys())
    return {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}


def tfidf(counts: Counter, idf: dict[str, float] | None) -> Vector:
    vec = {t: (1.0 + math.log(c)) * (idf.get(t, 1.0) if idf else 1.0) for t, c in counts.items() if c > 0}
    norm = math.sqrt(sum(v * v for v in vec.values()))
    return {t: v / norm for t, v in vec.items()} if norm else {}


def cosine(a: Vector, b: Vector) -> float:
    """Cosine of two L2-normalized sparse vectors."""
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(t, 0.0) for t, v in a.items())
