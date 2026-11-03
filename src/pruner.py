"""Hybrid bilingual context pruner (SPEC 3.1)."""
from __future__ import annotations

import re
import time
from collections import Counter

from .normalize import extract_entities, normalize_text
from .schemas import PruneResult
from .textsim import char_ngrams, cosine, idf_from, tfidf, word_terms
from .tokenizer import count_tokens

W_CHAR, W_WORD, W_ENT = 0.50, 0.35, 0.15
MAX_BOILERPLATE_WORDS = 15  # footers and headers are short; a longer paragraph is content whatever words it uses

# Footer/header phrases. They go through the same normalization as the text they are matched against, which
# strips Arabic clitics such as ال, so writing them in normalized form by hand would silently never match.
_FOOTER_PHRASES = ["all rights reserved", "this document is property of", "this document is the property of",
                   "جميع الحقوق محفوظة", "هذه الوثيقة ملك"]
_BOILERPLATE = re.compile("|".join(re.escape(normalize_text(p)) for p in _FOOTER_PHRASES)
                          + r"|page \d+ of \d+|صفحه \d+ من \d+")
# A bare marking set off from the rest of the line ("Acme Co. — Confidential."), matched on raw text because
# normalization removes the separator. A sentence that merely uses the word ("Salaries are confidential.") is content.
_MARKING = re.compile(r"(?:^|\s[—–|-]\s*)(?:confidential|سري للغاي[ةه])\.?\s*$", re.IGNORECASE)


def join_context(chunks: list[str]) -> str:
    return "\n\n".join(chunks)


def _is_footer(raw: str, norm: str) -> bool:
    if len(norm.split()) > MAX_BOILERPLATE_WORDS:
        return False
    return bool(_BOILERPLATE.search(norm) or _MARKING.search(raw.strip()))


def drop_boilerplate(chunks: list[str]) -> tuple[list[int], int]:
    """Indices of non-boilerplate chunks: removes repeated paragraphs and known footer/header text."""
    norm = [normalize_text(c) for c in chunks]
    counts = Counter(norm)
    keep = [i for i, n in enumerate(norm) if n and counts[n] < 2 and not _is_footer(chunks[i], n)]
    return keep, len(chunks) - len(keep)


def score_chunks(query: str, chunks: list[str]) -> list[float]:
    """S(q,p) = 0.50*cos_char + 0.35*cos_word + 0.15*1[shared numbers/IDs]."""
    if not chunks:
        return []
    q_char, q_word = char_ngrams(query), word_terms(query)
    c_char = [char_ngrams(c) for c in chunks]
    c_word = [word_terms(c) for c in chunks]
    idf_c = idf_from(c_char + [q_char])
    idf_w = idf_from(c_word + [q_word])
    qv_c, qv_w = tfidf(q_char, idf_c), tfidf(q_word, idf_w)
    q_ents = extract_entities(query)
    scores = []
    for cc, cw, chunk in zip(c_char, c_word, chunks):
        shared = bool(q_ents & extract_entities(chunk))
        scores.append(
            W_CHAR * cosine(qv_c, tfidf(cc, idf_c))
            + W_WORD * cosine(qv_w, tfidf(cw, idf_w))
            + W_ENT * float(shared)
        )
    return scores


def prune(query: str, chunks: list[str], theta: float = 0.18, min_k: int = 2,
          encoding: str = "o200k_base") -> PruneResult:
    t0 = time.perf_counter()
    candidates, n_boiler = drop_boilerplate(chunks)
    cand_chunks = [chunks[i] for i in candidates]
    scores = score_chunks(query, cand_chunks)
    ranked = sorted(range(len(scores)), key=lambda j: -scores[j])
    chosen = {j for j in ranked[:min_k]} | {j for j, s in enumerate(scores) if s >= theta}
    kept_local = sorted(chosen)  # preserve original document order
    kept_indices = [candidates[j] for j in kept_local]
    kept = [chunks[i] for i in kept_indices]
    tokens_before = count_tokens(join_context(chunks), encoding)
    tokens_after = count_tokens(join_context(kept), encoding)
    return PruneResult(
        kept=kept,
        kept_indices=kept_indices,
        candidates=candidates,
        scores=scores,
        dropped_boilerplate=n_boiler,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
    )


def relaxed(query: str, chunks: list[str], base: PruneResult, k: int = 4,
            encoding: str = "o200k_base") -> PruneResult:
    """Escalation-safety variant: union of the pruned set and the top-k chunks."""
    t0 = time.perf_counter()
    ranked = sorted(range(len(base.scores)), key=lambda j: -base.scores[j])
    chosen = {base.candidates[j] for j in ranked[:k]} | set(base.kept_indices)
    kept_indices = sorted(chosen)
    kept = [chunks[i] for i in kept_indices]
    return PruneResult(
        kept=kept,
        kept_indices=kept_indices,
        candidates=base.candidates,
        scores=base.scores,
        dropped_boilerplate=base.dropped_boilerplate,
        tokens_before=base.tokens_before,
        tokens_after=count_tokens(join_context(kept), encoding),
        elapsed_ms=(time.perf_counter() - t0) * 1000,
    )
