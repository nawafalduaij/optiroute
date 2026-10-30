"""OpenAI-compatible LLM client (Ollama / Groq) with disk recorder, replayer and offline simulator.

Every call is keyed by (task_id, variant, tier). `variant` names the context the prompt was built
from: "raw" (unpruned), "pruned" (top-k=2 + theta) or "relaxed" (top-k=4 escalation context).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Protocol

from .config import Settings, TierConfig
from .normalize import is_arabic
from .schemas import LLMResponse, Task
from .tokenizer import count_chat_tokens, count_tokens

SYSTEM_PROMPT = (
    "You are an enterprise document assistant. Answer strictly from the provided context. "
    "Be concise: give the final answer in one or two sentences, including exact figures, IDs and units. "
    "When arithmetic is needed, compute it and state the final number."
)


def build_messages(task: Task, context_chunks: list[str]) -> list[dict[str, str]]:
    context = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(context_chunks))
    instructions = []
    if is_arabic(task.query):
        instructions.append("أجب باللغة العربية.")
    if task.schema:
        keys = ", ".join(f'"{k}" ({t})' for k, t in task.schema.items())
        instructions.append(f"Return ONLY a JSON object with keys: {keys}. No prose.")
    user = f"Context:\n{context}\n\nQuestion: {task.query}"
    if instructions:
        user += "\n\n" + " ".join(instructions)
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def prompt_sha(messages: list[dict[str, str]]) -> str:
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()[:16]


class LLMBackend(Protocol):
    def complete(self, task: Task, variant: str, tier: int, chunks: list[str]) -> LLMResponse: ...


class LiveClient:
    """Calls the configured OpenAI-compatible endpoints; optionally appends every call to a JSONL file."""

    def __init__(self, settings: Settings, record_path: Path | None = None):
        import httpx

        self.settings = settings
        self.http = httpx.Client(timeout=settings.request_timeout_s)
        self.record_path = record_path
        self._seen: dict[tuple[str, int], dict] = {}  # (prompt_sha, tier) -> record, dedupes identical prompts

    def tier_cfg(self, tier: int) -> TierConfig:
        return self.settings.tier1 if tier == 1 else self.settings.tier2

    def _post(self, cfg: TierConfig, messages: list[dict[str, str]], max_tokens: int) -> tuple[str, dict, float]:
        body = {"model": cfg.model, "messages": messages, "temperature": self.settings.cascade.temperature,
                "max_tokens": max_tokens, "seed": 0}
        t0 = time.perf_counter()
        r = self.http.post(f"{cfg.base_url.rstrip('/')}/chat/completions", json=body,
                           headers={"Authorization": f"Bearer {cfg.api_key}"})
        latency = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"] or "", data.get("usage", {}), latency

    def warmup(self) -> dict[str, float]:
        """Load both models before timing so first-call model load is not counted as inference latency."""
        out = {}
        for tier in (1, 2):
            cfg = self.tier_cfg(tier)
            _, _, ms = self._post(cfg, [{"role": "user", "content": "ok"}], 1)
            out[cfg.model] = ms
        return out

    def complete(self, task: Task, variant: str, tier: int, chunks: list[str]) -> LLMResponse:
        cfg = self.tier_cfg(tier)
        messages = build_messages(task, chunks)
        sha = prompt_sha(messages)
        if (sha, tier) in self._seen:
            rec = dict(self._seen[(sha, tier)], task_id=task.task_id, variant=variant, deduped=True)
        else:
            text, usage, latency = self._post(cfg, messages, self.settings.cascade.max_tokens)
            enc = self.settings.tokenizer_encoding
            rec = {
                "task_id": task.task_id, "variant": variant, "tier": tier, "model_id": cfg.model,
                "params_b": cfg.params_b, "prompt_sha": sha, "prompt_tokens": count_chat_tokens(messages, enc),
                "completion_tokens": count_tokens(text, enc), "latency_ms": round(latency, 2),
                "api_usage": usage, "raw_output": text,
            }
            self._seen[(sha, tier)] = rec
        if self.record_path:
            with self.record_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return _to_response(rec, "live")


class ReplayClient:
    """Deterministically serves recorded outputs; fails loudly if a prompt was never recorded."""

    def __init__(self, path: Path, strict_prompts: bool = True):
        self.records: dict[tuple[str, str, int], dict] = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                self.records[(rec["task_id"], rec["variant"], rec["tier"])] = rec
        self.strict_prompts = strict_prompts
        self.prompt_mismatches: list[tuple[str, str, int]] = []
        self.path = Path(path)

    def tier_models(self) -> dict[int, tuple[str, float | None]]:
        """The (model_id, params_b) actually recorded for each tier; a tier must use exactly one model."""
        out: dict[int, set] = {}
        for rec in self.records.values():
            out.setdefault(rec["tier"], set()).add((rec["model_id"], rec.get("params_b")))
        for tier, models in out.items():
            if len(models) != 1:
                raise ValueError(f"{self.path}: tier {tier} mixes models {sorted(models)}")
        return {tier: next(iter(m)) for tier, m in out.items()}

    def complete(self, task: Task, variant: str, tier: int, chunks: list[str]) -> LLMResponse:
        key = (task.task_id, variant, tier)
        if key not in self.records:
            raise KeyError(f"No recorded run for {key}; re-run `evaluate.py --record`.")
        rec = self.records[key]
        if rec.get("prompt_sha") != prompt_sha(build_messages(task, chunks)):
            self.prompt_mismatches.append(key)
            if self.strict_prompts:
                raise ValueError(f"Prompt drift for {key}: pruner/prompt changed since recording.")
        return _to_response(rec, "replay")


class SimClient:
    """Offline synthetic model for unit tests and `--sim` smoke runs. NOT for headline numbers.

    Behaviour: answers correctly iff every evidence string survives in the given context. Tier-1
    additionally fails on multi-hop / low-overlap cases (deterministically by task-id hash), half the
    time with an ungrounded number (verifier-catchable) and half with a grounded wrong one.
    Latency is synthetic: 20ms + 0.5ms/token (Tier-1), 60ms + 2ms/token (Tier-2).
    """

    HARD = {"multi_hop", "low_lexical_overlap"}

    def __init__(self, settings: Settings):
        self.settings = settings

    def complete(self, task: Task, variant: str, tier: int, chunks: list[str]) -> LLMResponse:
        cfg = self.settings.tier1 if tier == 1 else self.settings.tier2
        messages = build_messages(task, chunks)
        ctx = "\n".join(chunks)
        h = int(hashlib.md5(task.task_id.encode()).hexdigest(), 16)
        has_evidence = all(e in ctx for e in task.evidence)
        if not has_evidence:
            text = "غير مذكور في السياق." if task.language == "ar" else "The context does not contain this information."
        elif tier == 1 and task.category in self.HARD and h % 3 != 0:
            text = _sim_wrong(task, grounded=(h % 2 == 0), ctx=ctx)
        else:
            text = _sim_answer(task)
        enc = self.settings.tokenizer_encoding
        pt, ct = count_chat_tokens(messages, enc), count_tokens(text, enc)
        latency = (20 + 0.5 * (pt + ct)) if tier == 1 else (60 + 2.0 * (pt + ct))
        rec = {"task_id": task.task_id, "variant": variant, "tier": tier, "model_id": f"sim:{cfg.model}",
               "prompt_tokens": pt, "completion_tokens": ct, "latency_ms": latency, "raw_output": text}
        return _to_response(rec, "sim")


def _sim_answer(task: Task) -> str:
    if task.answer_type == "json":
        return json.dumps(task.expected_answer, ensure_ascii=False)
    facts = [f[0] if isinstance(f, list) else f for f in task.required_facts]
    body = str(task.expected_answer)
    extra = [f for f in facts if f not in body]
    return " ".join([body] + extra)


def _sim_wrong(task: Task, grounded: bool, ctx: str) -> str:
    from .normalize import extract_numbers

    nums = extract_numbers(ctx)
    if grounded and nums:
        val = max(nums)  # grounded in context but wrong
    else:
        val = 987654.21  # ungrounded hallucination
    if task.answer_type == "json":
        return json.dumps({k: val if t == "number" else "?" for k, t in (task.schema or {}).items()})
    return f"{val:g}" if task.language == "en" else f"المبلغ {val:g}"


def _to_response(rec: dict, source: str) -> LLMResponse:
    return LLMResponse(
        tier=rec["tier"], model_id=rec["model_id"], variant=rec["variant"],
        prompt_tokens=rec["prompt_tokens"], completion_tokens=rec["completion_tokens"],
        latency_ms=rec["latency_ms"], raw_output=rec["raw_output"], source=source,
    )


def make_client(settings: Settings, mode: str, path: Path | None = None) -> LLMBackend:
    if mode == "sim":
        return SimClient(settings)
    if mode == "replay":
        if path is None:
            raise ValueError("replay mode needs a recorded JSONL path")
        return ReplayClient(path)
    if mode in ("live", "record"):
        return LiveClient(settings, record_path=path if mode == "record" else None)
    raise ValueError(f"unknown mode {mode!r}")
