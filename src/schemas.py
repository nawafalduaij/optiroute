"""Typed records shared across the cascade."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Task:
    task_id: str
    language: str  # "en" | "ar"
    category: str
    query: str
    context: list[str]  # raw paragraphs / chunks
    expected_answer: Any
    answer_type: str = "text"  # "number" | "text" | "json"
    required_facts: list[Any] = field(default_factory=list)  # str or list[str] alternatives
    evidence: list[str] = field(default_factory=list)  # substrings the pruned context must keep
    schema: dict[str, str] | None = None  # {"key": "number"|"string"} for JSON output
    cache_of: str | None = None  # task_id this case paraphrases / near-duplicates
    expect_cache_hit: bool | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Task":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PruneResult:
    kept: list[str]
    kept_indices: list[int]
    scores: list[float]  # score per surviving (non-boilerplate) chunk, original order
    dropped_boilerplate: int
    tokens_before: int
    tokens_after: int
    elapsed_ms: float


@dataclass
class RouteDecision:
    complexity: float
    signals: dict[str, float]
    tier: int
    tau: float
    elapsed_ms: float


@dataclass
class VerifierResult:
    passed: bool
    failed_checks: list[str]
    elapsed_ms: float


@dataclass
class LLMResponse:
    tier: int
    model_id: str
    variant: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    raw_output: str
    source: str = "live"  # live | replay | sim

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class CascadeResult:
    task_id: str
    answer: str
    final_tier: int  # 0 cache, 1 small, 2 large
    path: list[str]
    calls: list[LLMResponse]
    tokens_raw_context: int
    tokens_pruned_context: int
    compute_units: float  # kTok * B
    llm_latency_ms: float
    overhead_ms: dict[str, float]
    route: RouteDecision | None = None
    verifier: VerifierResult | None = None
    cache_event: str = "miss"  # hit | miss | blocked_trap | disabled
    tau: float = 0.0
    trace: dict[str, Any] = field(default_factory=dict)  # per-stage detail for the demo UI
    cost_usd: float | None = None  # only when every used tier has a configured price

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.calls)

    @property
    def latency_ms(self) -> float:
        return self.llm_latency_ms + sum(self.overhead_ms.values())
