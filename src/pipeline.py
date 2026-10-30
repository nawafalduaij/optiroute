"""Cascade orchestrator: Pruner -> Cache Guard -> Router -> Tier-1 -> Verifier -> (escalate) Tier-2."""
from __future__ import annotations

import time
from dataclasses import dataclass, replace

from . import pruner as pruner_mod
from .adaptation import AdaptationController
from .cache import SemanticCache, context_hash
from .config import Settings
from .llm_client import LLMBackend
from .normalize import extract_entities
from .router import route
from .schemas import CascadeResult, LLMResponse, RouteDecision, Task
from .tokenizer import count_tokens
from .verifier import answers_agree, verify


@dataclass(frozen=True)
class PipelineFlags:
    use_pruner: bool = True
    use_cache: bool = True
    entity_guard: bool = True
    use_router: bool = True
    use_verifier: bool = True
    use_adaptation: bool = True
    force_tier: int | None = None  # 1 or 2 bypasses the router
    shadow_audit_every: int = 0  # >0: every k-th verified Tier-1 answer is re-asked to Tier-2 as a health probe

    def but(self, **kw) -> "PipelineFlags":
        return replace(self, **kw)


class OptiRoutePipeline:
    def __init__(self, settings: Settings, client: LLMBackend, flags: PipelineFlags = PipelineFlags()):
        self.settings = settings
        self.cfg = settings.cascade
        self.client = client
        self.flags = flags
        self.cache = SemanticCache(threshold=self.cfg.cache_cosine, entity_guard=flags.entity_guard)
        self.adapt = AdaptationController(self.cfg, enabled=flags.use_adaptation)
        self.accepted_tier1 = 0

    def _params(self, tier: int) -> float:
        return self.settings.tier1.params_b if tier == 1 else self.settings.tier2.params_b

    def _call(self, task: Task, variant: str, tier: int, chunks: list[str]) -> LLMResponse:
        return self.client.complete(task, variant, tier, chunks)

    def process(self, task: Task) -> CascadeResult:
        enc = self.settings.tokenizer_encoding
        f = self.flags
        overhead = {"pruner": 0.0, "cache": 0.0, "router": 0.0, "verifier": 0.0}
        raw_tokens = count_tokens(pruner_mod.join_context(task.context), enc)

        # 1. Context pruning (or just relevance scoring for router/cache when pruning is disabled)
        t0 = time.perf_counter()
        pr = pruner_mod.prune(task.query, task.context, self.cfg.prune_theta, self.cfg.prune_min_k, enc)
        if f.use_pruner:
            chunks, variant, scores = pr.kept, "pruned", pr.scores
            overhead["pruner"] = (time.perf_counter() - t0) * 1000
        else:
            chunks, variant, scores = task.context, "raw", pr.scores
            overhead["router"] += (time.perf_counter() - t0) * 1000  # scoring cost charged to router
        pruned_tokens = count_tokens(pruner_mod.join_context(chunks), enc)
        candidates, _ = pruner_mod.drop_boilerplate(task.context)
        trace: dict = {"prune": {
            "enabled": f.use_pruner, "scores": {candidates[j]: round(s, 4) for j, s in enumerate(scores)},
            "kept": pr.kept_indices, "boilerplate": [i for i in range(len(task.context)) if i not in candidates],
            "tokens_before": raw_tokens, "tokens_after": pruned_tokens, "theta": self.cfg.prune_theta}}

        # 2. Tier-0 cache (key: query + hash of top-2 relevant chunks)
        cache_event, ctx_hash = "disabled", ""
        if f.use_cache:
            t0 = time.perf_counter()
            top2 = sorted(sorted(range(len(scores)), key=lambda j: -scores[j])[:2])
            ctx_hash = context_hash([task.context[candidates[j]] for j in top2])
            look = self.cache.lookup(task.query, ctx_hash)
            overhead["cache"] = (time.perf_counter() - t0) * 1000
            cache_event = look.event
            trace["cache"] = {
                "event": look.event, "cosine": round(look.cosine, 4), "threshold": self.cache.threshold,
                "entity_guard": self.cache.entity_guard, "entities_new": sorted(extract_entities(task.query)),
                "matched_task": look.entry.task_id if look.entry else None,
                "matched_query": look.entry.query if look.entry else None,
                "entities_cached": sorted(look.entry.entities) if look.entry else None}
            if look.event == "hit":
                res = CascadeResult(
                    task_id=task.task_id, answer=look.entry.answer, final_tier=0,
                    path=[f"cache_hit:{look.entry.task_id}"], calls=[], tokens_raw_context=raw_tokens,
                    tokens_pruned_context=pruned_tokens, compute_units=0.0, llm_latency_ms=0.0,
                    overhead_ms=overhead, cache_event="hit", tau=self.adapt.tau, trace=trace,
                )
                self.adapt.observe(res.latency_ms, None)
                return res

        # 3. Routing
        tau = self.adapt.tau
        decision: RouteDecision | None = None
        if f.force_tier:
            tier = f.force_tier
        else:
            decision = route(task.query, scores, task.schema, tau)
            overhead["router"] += decision.elapsed_ms
            tier = decision.tier if f.use_router else 1

        # 4. Inference + verification + escalation
        calls: list[LLMResponse] = []
        path: list[str] = []
        ver = None
        tier1_passed = None
        audit_agree = None
        if tier == 1:
            r1 = self._call(task, variant, 1, chunks)
            calls.append(r1)
            path.append(f"tier1:{variant}")
            answer, final_tier = r1.raw_output, 1
            if f.use_verifier:
                ver = verify(task.query, pruner_mod.join_context(chunks), r1.raw_output, task.schema)
                overhead["verifier"] = ver.elapsed_ms
                tier1_passed = ver.passed
                if not ver.passed:
                    if f.use_pruner and "degenerate_or_refusal" in ver.failed_checks:
                        # A refusal on pruned context signals context starvation (pruner miss):
                        # recover with the full raw context instead of the top-4 relaxed set.
                        esc_chunks, esc_variant = task.context, "raw"
                    elif f.use_pruner:
                        rel = pruner_mod.relaxed(task.query, task.context, pr, self.cfg.relaxed_k, enc)
                        esc_chunks, esc_variant = rel.kept, "relaxed"
                    else:
                        esc_chunks, esc_variant = task.context, "raw"
                    r2 = self._call(task, esc_variant, 2, esc_chunks)
                    calls.append(r2)
                    path.append(f"escalate:{'+'.join(ver.failed_checks)}")
                    path.append(f"tier2:{esc_variant}")
                    answer, final_tier = r2.raw_output, 2
                elif f.shadow_audit_every:
                    self.accepted_tier1 += 1
                    if self.accepted_tier1 % f.shadow_audit_every == 0:
                        # Unbiased health probe: confident-but-wrong Tier-1 answers pass the verifier,
                        # so sampled disagreement with Tier-2 is what feeds the adaptation window.
                        r2 = self._call(task, variant, 2, chunks)
                        calls.append(r2)
                        audit_agree = answers_agree(r1.raw_output, r2.raw_output)
                        path.append(f"shadow_audit:{'agree' if audit_agree else 'disagree'}")
                        if not audit_agree:
                            answer, final_tier = r2.raw_output, 2
        else:
            r2 = self._call(task, variant, 2, chunks)
            calls.append(r2)
            path.append(f"tier2:{variant}")
            answer, final_tier = r2.raw_output, 2

        res = CascadeResult(
            task_id=task.task_id, answer=answer, final_tier=final_tier, path=path, calls=calls,
            tokens_raw_context=raw_tokens, tokens_pruned_context=pruned_tokens,
            compute_units=sum(c.total_tokens / 1000 * self._params(c.tier) for c in calls),
            llm_latency_ms=sum(c.latency_ms for c in calls), overhead_ms=overhead, route=decision,
            verifier=ver, cache_event=cache_event, tau=tau, trace=trace,
        )
        # 5. Feedback: adaptation + cache population (only accepted answers are cached)
        self.adapt.observe(res.latency_ms, tier1_passed, audit_agree)
        if f.use_cache:
            self.cache.insert(task.task_id, task.query, ctx_hash, answer)
        return res

    def run(self, tasks: list[Task]) -> list[CascadeResult]:
        return [self.process(t) for t in tasks]
