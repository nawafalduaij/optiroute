"""Offline benchmark judge (SPEC 3.4.2). Used ONLY by evaluate.py — never by the runtime cascade."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .normalize import extract_numbers, normalize_text
from .verifier import parse_json_obj  # parsing helper only; no verifier decision is used here

REL_TOL = 0.005


@dataclass
class Score:
    correct: bool
    fact_recall: float
    f1: float


def _num_eq(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.01, abs(b) * REL_TOL)


def _fact_present(fact: Any, output_norm: str, output_nums: list[float]) -> bool:
    alts = fact if isinstance(fact, list) else [fact]
    for alt in alts:
        if isinstance(alt, (int, float)):
            if any(_num_eq(n, float(alt)) for n in output_nums):
                return True
            continue
        alt_nums = extract_numbers(str(alt))
        alt_norm = normalize_text(str(alt))
        if alt_nums:
            # mixed fact like "4,500 SAR": numbers compared by value, words by normalized inclusion
            words = [w for w in alt_norm.split() if not any(ch.isdigit() for ch in w)]
            if all(any(_num_eq(n, v) for n in output_nums) for v in alt_nums) and \
                    all(w in output_norm.split() for w in words):
                return True
        elif alt_norm and alt_norm in output_norm:
            return True
    return False


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_text(pred).split(), normalize_text(gold).split()
    if not p or not g:
        return 0.0
    common = sum(min(p.count(t), g.count(t)) for t in set(p))
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def score(task, output: str) -> Score:
    """Exact normalized match for JSON/numbers, fact inclusion (+F1) for text."""
    out_norm = normalize_text(output)
    out_nums = extract_numbers(output)
    facts = task.required_facts or []
    present = [_fact_present(f, out_norm, out_nums) for f in facts]
    fact_recall = sum(present) / len(present) if present else 1.0

    if task.answer_type == "json":
        obj = parse_json_obj(output)
        ok = obj is not None
        if ok:
            for k, v in task.expected_answer.items():
                got = obj.get(k)
                if isinstance(v, (int, float)):
                    got_nums = extract_numbers(str(got)) if got is not None else []
                    ok &= any(_num_eq(n, float(v)) for n in got_nums)
                else:
                    ok &= got is not None and normalize_text(str(v)) in normalize_text(str(got))
        return Score(correct=bool(ok), fact_recall=fact_recall, f1=1.0 if ok else 0.0)

    if task.answer_type == "number":
        ok = any(_num_eq(n, float(task.expected_answer)) for n in out_nums) and all(present)
        return Score(correct=ok, fact_recall=fact_recall, f1=token_f1(output, str(task.expected_answer)))

    return Score(correct=all(present) and bool(facts), fact_recall=fact_recall,
                 f1=token_f1(output, str(task.expected_answer)))
