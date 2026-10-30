"""Bilingual text normalization and numeric/entity extraction (shared by pruner, cache, verifier, scorer)."""
from __future__ import annotations

import re

_ALEF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ة": "ه", "ى": "ي"})
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_TASHKEEL = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")  # + tatweel
_PROCLITICS = ("وال", "بال", "كال", "فال", "لل", "ال")
_WORD = re.compile(r"[\w؀-ۿ]+", re.UNICODE)
_ARABIC_CHAR = re.compile(r"[؀-ۿ]")
_LATIN_CHAR = re.compile(r"[A-Za-z]")

# Identifier: INV-8841, PO-2231, #INV-8841, SLA-3, etc. (letters, dash, digits)
_ID = re.compile(r"#?\b([A-Za-z]{2,6})[-_]?(\d{2,})\b")
# Number with optional thousands separators, decimals and percent sign.
_NUM = re.compile(r"(?<![A-Za-z0-9.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|(?<![A-Za-z0-9.])-?\d+(?:\.\d+)?%?")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")


_HYPHENS = str.maketrans({c: "-" for c in "\u2010\u2011\u2012\u2013\u2212"})
# "7 200" with a narrow/no-break/thin space as thousands separator (common in typeset model output)
_SPACE_THOUSANDS = re.compile(r"(?<=\d)[\u202f\u00a0\u2009](?=\d{3}(?!\d))")


def to_western_digits(text: str) -> str:
    # Arabic thousands (٬) and decimal (٫) separators -> ASCII; typographic hyphens -> "-"
    text = text.translate(_DIGITS).translate(_HYPHENS).replace("٬", ",").replace("٫", ".").replace("٪", "%")
    return _SPACE_THOUSANDS.sub(",", text)


def normalize_arabic(text: str) -> str:
    """Unify Alef/Teh Marbuta/Alef Maksura variants and strip diacritics + tatweel."""
    return _TASHKEEL.sub("", text).translate(_ALEF)


def strip_proclitic(token: str) -> str:
    if len(token) >= 5 and _ARABIC_CHAR.match(token):
        for p in _PROCLITICS:
            if token.startswith(p) and len(token) - len(p) >= 2:
                return token[len(p):]
    return token


def normalize_text(text: str) -> str:
    """Full bilingual normalization: digits, Arabic orthography, lowercase, clitic stripping."""
    text = normalize_arabic(to_western_digits(text)).lower()
    return " ".join(strip_proclitic(t) for t in _WORD.findall(text))


def words(text: str) -> list[str]:
    return normalize_text(text).split()


def is_arabic(text: str, threshold: float = 0.5) -> bool:
    ar = len(_ARABIC_CHAR.findall(text))
    la = len(_LATIN_CHAR.findall(text))
    return ar > 0 and ar / max(1, ar + la) >= threshold


def arabic_ratio(text: str) -> float:
    ar = len(_ARABIC_CHAR.findall(text))
    la = len(_LATIN_CHAR.findall(text))
    return ar / max(1, ar + la)


def parse_number(token: str) -> float | None:
    try:
        return float(token.replace(",", "").rstrip("%"))
    except ValueError:
        return None


def extract_numbers(text: str) -> list[float]:
    """All numeric values (Eastern Arabic numerals and thousands separators handled)."""
    text = to_western_digits(text)
    text = _ID.sub(" ", text)  # IDs are entities, not quantities
    text = _DATE.sub(" ", text)
    out = []
    for m in _NUM.finditer(text):
        v = parse_number(m.group(0))
        if v is not None:
            out.append(v)
    return out


def extract_entities(text: str) -> frozenset[str]:
    """Canonical set of numbers, percentages, dates and alphanumeric IDs (cache guard key)."""
    text = to_western_digits(text)
    ents: set[str] = set()
    for m in _ID.finditer(text):
        ents.add(f"id:{m.group(1).upper()}-{m.group(2)}")
    for m in _DATE.finditer(text):
        ents.add(f"date:{m.group(0)}")
    stripped = _DATE.sub(" ", _ID.sub(" ", text))
    for m in _NUM.finditer(stripped):
        tok = m.group(0)
        v = parse_number(tok)
        if v is None:
            continue
        ents.add(f"{'pct' if tok.endswith('%') else 'num'}:{v:g}")
    return frozenset(ents)
