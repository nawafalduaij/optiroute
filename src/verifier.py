"""Runtime, ground-truth-free verifier for Tier-1 outputs (SPEC 3.4.1).

Sees only (query, pruned context, schema, tier-1 output). Never imports the ground-truth scorer.
"""
from __future__ import annotations

import bisect
import json
import re
import time
from itertools import product

from .normalize import arabic_ratio, extract_numbers, is_arabic, normalize_arabic
from .schemas import VerifierResult

_REFUSALS = [
    "i don't know", "i do not know", "cannot determine", "can't determine", "unable to determine",
    "not provided", "not mentioned", "not specified", "no information", "does not contain",
    "doesn't contain", "not enough information", "insufficient information", "i'm not sure",
    "i am not sure", "cannot answer", "can't answer", "not available in the context",
    "لا اعرف", "لا اعلم", "غير مذكور", "غير محدد", "لا يمكن تحديد", "لا يمكنني", "لا تتوفر",
    "غير متوفر", "لا يوجد معلومات", "لا توجد معلومات", "لم يذكر", "لم يتم ذكر",
]
# Generalized "the context lacks it" pattern: negation + information verb (any tense/voice).
_REFUSAL_RE = re.compile(
    r"\b(?:does not|doesn't|do not|don't|did not|didn't|is not|isn't|are not|aren't|was not|wasn't|not)\s+"
    r"(?:explicitly\s+|directly\s+|clearly\s+)?(?:provide|mention|specify|state|contain|include|cover|address"
    r"|list|indicate|say)\w*\b.{0,40}?\b(?:information|details?|context|document|text|this|that|how|what|when|any)\b"
    r"|\bno (?:specific |relevant |explicit )?(?:information|mention|details?|data)\b",
    re.IGNORECASE)
_REFUSAL_AR = re.compile(r"(?:لا|لم)\s+(?:يذكر|يحدد|يتضمن|يوضح|يرد|ترد|تذكر|تحدد|تتضمن|يشير|تشير|يتم ذكر)")
_REPEAT = re.compile(r"(.{6,}?)\1{4,}", re.DOTALL)
_CHAR_RUN = re.compile(r"(.)\1{19,}")
_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)
TRIVIAL_INT = 10  # small integers (list ordinals, "2 parties") are always considered grounded
REL_TOL = 0.005


def parse_json_obj(text: str) -> dict | None:
    m = _JSON_OBJ.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def check_degenerate(output: str) -> bool:
    """True if the output is empty, a refusal/uncertainty, or a repetition loop."""
    text = output.strip()
    if len(text) == 0:
        return True
    low = normalize_arabic(text.lower())
    if any(p in low for p in (normalize_arabic(r) for r in _REFUSALS)):
        return True
    if _REFUSAL_RE.search(text) or _REFUSAL_AR.search(normalize_arabic(text)):
        return True
    return bool(_REPEAT.search(text) or _CHAR_RUN.search(text))


def check_schema(output: str, schema: dict[str, str]) -> bool:
    obj = parse_json_obj(output)
    if obj is None:
        return False
    for key, typ in schema.items():
        if key not in obj:
            return False
        val = obj[key]
        if typ == "number":
            if isinstance(val, bool):
                return False
            if not isinstance(val, (int, float)):
                if not (isinstance(val, str) and extract_numbers(val)):
                    return False
        elif typ == "string" and not isinstance(val, str):
            return False
    return True


def derivable_numbers(source_numbers: list[float]) -> list[float]:
    """Numbers in source plus simple arithmetic combinations (+,-,*,/,%) of pairs."""
    base = sorted(set(source_numbers))
    out = set(base)
    small = [n for n in base if n == int(n) and 1 < n <= 12]
    pair_results = set()
    for a, b in product(base, repeat=2):
        vals = [a + b, a - b, a * b, a * b / 100, a * (1 + b / 100), a * (1 - b / 100)]
        if b:
            vals += [a / b, a / b * 100]
        pair_results.update(vals)
    out |= pair_results
    if small:  # e.g. monthly fee * credit% * number of months
        out |= {r * n for r in pair_results for n in small}
    out.add(sum(base))
    return sorted(out)


def _grounded(value: float, allowed: list[float]) -> bool:
    if abs(value) <= TRIVIAL_INT and value == int(value):
        return True
    tol = max(0.01, abs(value) * REL_TOL)
    i = bisect.bisect_left(allowed, value - tol)
    return i < len(allowed) and allowed[i] <= value + tol


def needs_arithmetic(query: str) -> bool:
    """Same bilingual markers the router uses: only computational questions may yield derived numbers."""
    from .router import signals  # local import keeps the verifier free of routing state

    sig = signals(query, [], None)
    return sig["quantitative_reasoning"] == 1.0 or sig["multi_hop_markers"] == 1.0


def check_grounding(output: str, query: str, context: str) -> bool:
    """Lookups must quote numbers verbatim; computations may also use + - x / % combinations."""
    out_nums = extract_numbers(output)
    if not out_nums:
        return True
    source = extract_numbers(context) + extract_numbers(query)
    allowed = derivable_numbers(source) if needs_arithmetic(query) else sorted(set(source))
    return all(_grounded(v, allowed) for v in out_nums)


def check_language(output: str, query: str, schema: dict | None) -> bool:
    if not is_arabic(query):
        return True
    text = output
    if schema:
        obj = parse_json_obj(output) or {}
        text = " ".join(v for v in obj.values() if isinstance(v, str))
    letters = re.findall(r"[A-Za-z؀-ۿ]", text)
    if len(letters) < 3:
        return True  # purely numeric answers carry no language signal
    return arabic_ratio(text) >= 0.3


def verify(query: str, context: str, output: str, schema: dict | None = None) -> VerifierResult:
    t0 = time.perf_counter()
    failed = []
    if check_degenerate(output):
        failed.append("degenerate_or_refusal")
    if schema and not check_schema(output, schema):
        failed.append("json_schema")
    if not check_grounding(output, query, context):
        failed.append("numeric_grounding")
    if not check_language(output, query, schema):
        failed.append("language_match")
    return VerifierResult(passed=not failed, failed_checks=failed,
                          elapsed_ms=(time.perf_counter() - t0) * 1000)


def answers_agree(small: str, large: str) -> bool:
    """Shadow-audit comparison (no ground truth): numbers must match; otherwise content words must overlap."""
    from .normalize import words

    n_small, n_large = extract_numbers(small), extract_numbers(large)
    if n_small or n_large:
        return bool(n_small) and all(any(abs(a - b) <= max(0.01, abs(b) * REL_TOL) for b in n_large)
                                     for a in n_small if not (abs(a) <= TRIVIAL_INT and a == int(a)))
    ws = {w for w in words(small) if len(w) > 3}
    wl = {w for w in words(large) if len(w) > 3}
    return bool(ws) and len(ws & wl) / len(ws) >= 0.5
