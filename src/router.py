"""Deterministic, inspectable complexity router (SPEC 3.3)."""
from __future__ import annotations

import math
import re
import time

from .normalize import extract_numbers, normalize_text
from .schemas import RouteDecision

WEIGHTS = {
    "multi_hop_markers": 0.30,
    "quantitative_reasoning": 0.30,
    "structural_strictness": 0.25,
    "context_dispersion": 0.15,
}

_MULTI_HOP = [
    "compare", "comparison", "difference between", "reconcile", "across both", "both",
    "combined", "after applying", "based on clauses", "together with", "versus", "vs",
    "قارن", "مقارنه", "الفرق بين", "احسب إجمالي", "بناء على البندين", "كلا", "معا", "مجموع",
    "بعد تطبيق", "if", "plus", "then", "إذا", "ثم", "بالإضافة إلى",
]
_ARITHMETIC = [
    "calculate", "compute", "how much would", "after applying", "net amount", "net payable",
    "deduct", "penalty amount", "credit amount", "total cost", "total payable", "sum of",
    "احسب", "اجمالي المبلغ", "صافي", "خصم", "قيمه الغرامه", "المبلغ المستحق بعد", "مجموع",
]
_TABLE = ["table", "tabular", "جدول"]


def _compile(phrases: list[str]) -> re.Pattern:
    normed = sorted({normalize_text(p) for p in phrases}, key=len, reverse=True)
    return re.compile(r"(?<!\S)(" + "|".join(re.escape(p) for p in normed) + r")(?!\S)")


_RE_MULTI, _RE_ARITH, _RE_TABLE = _compile(_MULTI_HOP), _compile(_ARITHMETIC), _compile(_TABLE)


def dispersion(scores: list[float], top: int = 4) -> float:
    """Normalized entropy of the top chunk relevance distribution in [0, 1]."""
    s = sorted((max(0.0, x) for x in scores), reverse=True)[:top]
    total = sum(s)
    if len(s) < 2 or total <= 0:
        return 0.0
    p = [x / total for x in s if x > 0]
    return -sum(x * math.log(x) for x in p) / math.log(len(s))


def signals(query: str, chunk_scores: list[float], schema: dict | None) -> dict[str, float]:
    q = normalize_text(query)
    multi = 1.0 if _RE_MULTI.search(q) else 0.0
    if _RE_ARITH.search(q):
        quant = 1.0
    else:
        quant = 0.5 if len(extract_numbers(query)) >= 2 else 0.0
    if schema and len(schema) >= 2:
        struct = 1.0
    elif schema or _RE_TABLE.search(q):
        struct = 0.5 if schema else 1.0
    else:
        struct = 0.0
    return {
        "multi_hop_markers": multi,
        "quantitative_reasoning": quant,
        "structural_strictness": struct,
        "context_dispersion": round(dispersion(chunk_scores), 4),
    }


def route(query: str, chunk_scores: list[float], schema: dict | None, tau: float) -> RouteDecision:
    t0 = time.perf_counter()
    sig = signals(query, chunk_scores, schema)
    c = sum(WEIGHTS[k] * v for k, v in sig.items())
    return RouteDecision(
        complexity=round(c, 4), signals=sig, tier=1 if c < tau else 2, tau=tau,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
    )
