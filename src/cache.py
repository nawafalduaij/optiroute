"""Tier-0 semantic cache with strict numeric/entity guard (SPEC 3.2)."""
from __future__ import annotations

import hashlib
import re
import time
from collections import Counter
from dataclasses import dataclass, field

from .normalize import extract_entities, normalize_text, to_western_digits
from .textsim import Vector, char_ngrams, cosine, tfidf, word_terms

W_CHAR, W_WORD = 0.6, 0.4
_DIGITS = re.compile(r"\d+")


def mask_numbers(text: str) -> str:
    """Similarity is computed on wording only; numbers/IDs are compared exactly by the entity guard."""
    return _DIGITS.sub("0", to_western_digits(text))


def context_hash(chunks: list[str]) -> str:
    return hashlib.sha256("\x1e".join(normalize_text(c) for c in chunks).encode()).hexdigest()[:16]


@dataclass
class CacheEntry:
    task_id: str
    query: str
    char_vec: Vector
    word_vec: Vector
    entities: frozenset[str]
    ctx_hash: str
    answer: str


@dataclass
class CacheLookup:
    event: str  # hit | miss | blocked_trap
    entry: CacheEntry | None
    cosine: float
    elapsed_ms: float


@dataclass
class SemanticCache:
    threshold: float = 0.88
    entity_guard: bool = True
    entries: list[CacheEntry] = field(default_factory=list)
    stats: Counter = field(default_factory=Counter)

    def _similarities(self, query: str) -> list[float]:
        masked = mask_numbers(query)
        q_char, q_word = char_ngrams(masked), word_terms(masked)
        # Sublinear TF without IDF: similarity must not drift as the cache grows (stable 0.88 threshold).
        qc, qw = tfidf(q_char, None), tfidf(q_word, None)
        return [W_CHAR * cosine(qc, e.char_vec) + W_WORD * cosine(qw, e.word_vec) for e in self.entries]

    def lookup(self, query: str, ctx_hash: str) -> CacheLookup:
        t0 = time.perf_counter()
        if not self.entries:
            self.stats["miss"] += 1
            return CacheLookup("miss", None, 0.0, (time.perf_counter() - t0) * 1000)
        sims = self._similarities(query)
        ents = extract_entities(query)
        best_hit, best_trap = None, None
        for sim, e in sorted(zip(sims, self.entries), key=lambda x: -x[0]):
            if sim < self.threshold:
                break
            if e.ctx_hash != ctx_hash:
                continue
            if self.entity_guard and e.entities != ents:
                best_trap = best_trap or (sim, e)
                continue
            best_hit = (sim, e)
            break
        elapsed = (time.perf_counter() - t0) * 1000
        if best_hit:
            self.stats["hit"] += 1
            return CacheLookup("hit", best_hit[1], best_hit[0], elapsed)
        if best_trap:
            self.stats["blocked_trap"] += 1
            return CacheLookup("blocked_trap", best_trap[1], best_trap[0], elapsed)
        self.stats["miss"] += 1
        return CacheLookup("miss", None, max(sims), elapsed)

    def insert(self, task_id: str, query: str, ctx_hash: str, answer: str) -> None:
        self.entries.append(CacheEntry(
            task_id=task_id, query=query, char_vec=tfidf(char_ngrams(mask_numbers(query)), None),
            word_vec=tfidf(word_terms(mask_numbers(query)), None),
            entities=extract_entities(query), ctx_hash=ctx_hash, answer=answer,
        ))
