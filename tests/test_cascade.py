"""Control-flow tests for adaptation, scorer, record/replay and the full cascade (sim mode only)."""
import json

import pytest

from src.adaptation import AdaptationController
from src.config import CascadeConfig
from src.ground_truth_scorer import score
from src.llm_client import ReplayClient, SimClient, build_messages, prompt_sha
from src.pipeline import OptiRoutePipeline, PipelineFlags


# ---------------------------------------------------------------- adaptation
def test_adaptation_tightens_on_failures_and_recovers():
    cfg = CascadeConfig(latency_slo_ms=1000)
    a = AdaptationController(cfg)
    for _ in range(10):
        a.observe(100, False)
    assert a.tau < cfg.tau_base
    low = a.tau
    assert low >= cfg.tau_min
    for _ in range(60):
        a.observe(100, True)
    assert a.tau == cfg.tau_base  # hysteresis returns exactly to baseline, never above
    actions = [h["action"] for h in a.history]
    assert "tighten" in actions and "recover" in actions
    first_recover = actions.index("recover")
    # recovery only after the required streak of healthy steps
    assert all(x != "recover" for x in actions[first_recover - cfg.recover_consecutive + 1:first_recover])


def test_adaptation_audit_exposes_confident_failures():
    """Verifier passes everything, but shadow audits disagree: the estimate must fall and tau tighten."""
    cfg = CascadeConfig(latency_slo_ms=1000)
    a = AdaptationController(cfg)
    for i in range(12):
        a.observe(100, True, audit_agree=(i % 2 == 0) if i % 3 == 0 else None)
    assert a.tau < cfg.tau_base
    b = AdaptationController(cfg)
    for _ in range(12):
        b.observe(100, True)  # no audits: verifier-only signal stays healthy
    assert b.tau == cfg.tau_base


def test_adaptation_tightens_on_latency_slo():
    a = AdaptationController(CascadeConfig(latency_slo_ms=500))
    for _ in range(6):
        a.observe(2000, True)
    assert a.tau < 0.55


def test_adaptation_bounds_and_disabled():
    cfg = CascadeConfig(latency_slo_ms=1000)
    a = AdaptationController(cfg)
    for _ in range(200):
        a.observe(100, False)
    assert a.tau == cfg.tau_min
    b = AdaptationController(cfg, enabled=False)
    for _ in range(50):
        b.observe(5000, False)
    assert b.tau == cfg.tau_base


# ---------------------------------------------------------------- ground-truth scorer
def test_scorer_number_text_json(held_out):
    assert score(held_out["std_en_01"], "The monthly fee is SAR 48,000.").correct
    assert not score(held_out["std_en_01"], "The monthly fee is SAR 4,800.").correct
    assert score(held_out["std_ar_01"], "مدة الإجازة السنوية ٢١ يوماً").correct
    assert score(held_out["std_en_03"], "Within the Kingdom of Saudi Arabia, in Riyadh and Dammam.").correct
    assert not score(held_out["std_en_03"], "In Riyadh.").correct
    assert score(held_out["mh_en_02"], '{"invoice_id": "INV-9120", "amount_payable": 4508.0}').correct
    assert not score(held_out["mh_en_02"], '{"invoice_id": "INV-9120", "amount_payable": 4600}').correct
    assert score(held_out["std_en_05"], "Due to a quantity mismatch with PO-5520.").correct


# ---------------------------------------------------------------- record / replay
def test_replay_roundtrip_and_drift_detection(tmp_path, settings, held_out):
    t = held_out["std_en_01"]
    sim = SimClient(settings)
    r = sim.complete(t, "raw", 1, t.context)
    rec = {"task_id": t.task_id, "variant": "raw", "tier": 1, "model_id": "m",
           "prompt_sha": prompt_sha(build_messages(t, t.context)), "prompt_tokens": r.prompt_tokens,
           "completion_tokens": r.completion_tokens, "latency_ms": 12.5, "raw_output": r.raw_output}
    path = tmp_path / "runs.jsonl"
    path.write_text(json.dumps(rec) + "\n")
    rp = ReplayClient(path)
    out = rp.complete(t, "raw", 1, t.context)
    assert out.raw_output == r.raw_output and out.latency_ms == 12.5 and out.source == "replay"
    with pytest.raises(ValueError):
        rp.complete(t, "raw", 1, t.context[:2])  # different prompt -> drift
    with pytest.raises(KeyError):
        rp.complete(t, "raw", 2, t.context)


# ---------------------------------------------------------------- full cascade (sim)
def _run(settings, held_out, flags):
    pipe = OptiRoutePipeline(settings, SimClient(settings), flags)
    return pipe, {r.task_id: r for r in pipe.run(list(held_out.values()))}


def test_full_cascade_paths(settings, held_out):
    pipe, res = _run(settings, held_out, PipelineFlags())
    assert res["cache_en_hit"].final_tier == 0 and res["cache_en_hit"].compute_units == 0
    assert res["cache_ar_hit"].final_tier == 0
    assert res["cache_en_trap"].cache_event == "blocked_trap" and res["cache_en_trap"].final_tier > 0
    assert res["cache_ar_trap"].cache_event == "blocked_trap"
    assert res["std_en_07"].final_tier == 1 and res["std_en_07"].calls[0].variant == "pruned"
    assert res["mh_en_03"].path == ["tier2:pruned"]
    # refusals escalate with full raw context (context starvation); other failures with relaxed top-4
    for r in res.values():
        if any(p.startswith("escalate") for p in r.path):
            expected = "raw" if "degenerate_or_refusal" in r.verifier.failed_checks else "relaxed"
            assert r.calls[-1].variant == expected and r.final_tier == 2
    # low-overlap cases whose evidence was pruned away are recovered via escalation
    assert all(score(held_out[t], res[t].answer).correct for t in ("llo_en_01", "llo_en_02", "llo_en_03"))
    assert set(pipe.cache.stats) <= {"hit", "miss", "blocked_trap"}


def test_no_guard_causes_false_hits(settings, held_out):
    _, res = _run(settings, held_out, PipelineFlags(entity_guard=False))
    assert res["cache_en_trap"].final_tier == 0
    assert not score(held_out["cache_en_trap"], res["cache_en_trap"].answer).correct


def test_full_beats_tier1_only_and_saves_vs_tier2(settings, held_out):
    _, full = _run(settings, held_out, PipelineFlags())
    _, t1 = _run(settings, held_out, PipelineFlags(use_pruner=False, use_cache=False, use_verifier=False,
                                                   use_adaptation=False, force_tier=1))
    _, t2 = _run(settings, held_out, PipelineFlags(use_pruner=False, use_cache=False, use_verifier=False,
                                                   use_adaptation=False, force_tier=2))
    acc = lambda res: sum(score(held_out[k], r.answer).correct for k, r in res.items())
    assert acc(full) > acc(t1)
    assert sum(r.compute_units for r in full.values()) < sum(r.compute_units for r in t2.values())
    assert sum(r.total_tokens for r in full.values()) < sum(r.total_tokens for r in t2.values())


def test_no_pruner_uses_raw_context(settings, held_out):
    _, res = _run(settings, held_out, PipelineFlags(use_pruner=False))
    for r in res.values():
        assert all(c.variant == "raw" for c in r.calls)


def test_replay_binds_recorded_models_not_env(tmp_path, settings, held_out):
    """Regression: reports must name the models that produced the recording, whatever `.env` says."""
    import evaluate

    t = held_out["std_en_01"]
    rows = [{"task_id": t.task_id, "variant": "raw", "tier": tier, "model_id": model, "params_b": params,
             "prompt_sha": prompt_sha(build_messages(t, t.context)), "prompt_tokens": 100,
             "completion_tokens": 10, "latency_ms": 1.0, "raw_output": "SAR 48,000"}
            for tier, model, params in ((1, "tiny-model", 0.5), (2, "big-model", 70.0))]
    path = tmp_path / "runs.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    s = evaluate.settings_for(settings, ReplayClient(path))
    assert (s.tier1.model, s.tier1.params_b, s.tier2.model, s.tier2.params_b) == ("tiny-model", 0.5, "big-model", 70.0)
    path.write_text("".join(json.dumps({**r, "tier": 1}) + "\n" for r in rows[:1]) +
                    json.dumps({**rows[1], "tier": 1, "task_id": "x"}) + "\n")
    with pytest.raises(ValueError):  # one tier, two models -> refuse to label
        evaluate.settings_for(settings, ReplayClient(path))


def test_live_client_retries_rate_limits(settings, held_out):
    """Cloud tiers return 429 under free-tier limits: retry, and time only the successful attempt."""
    import httpx

    from src.llm_client import LiveClient

    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] <= 2:
            return httpx.Response(429, headers={"retry-after": "0.01"}, json={"error": "rate limited"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "SAR 48,000"}}], "usage": {}})

    client = LiveClient(settings)
    client.http = httpx.Client(transport=httpx.MockTransport(handler))
    t = held_out["std_en_01"]
    r = client.complete(t, "raw", 2, t.context)
    assert r.raw_output == "SAR 48,000" and calls["n"] == 3 and r.latency_ms < 1000


def test_profiles_load_and_keep_secrets_out(monkeypatch):
    from src.config import PROFILES, get_settings, list_profiles

    names = {p["name"] for p in list_profiles()}
    assert {"single-gpu", "hybrid", "edge", "cloud"} <= names
    for path in PROFILES.glob("*.env"):  # profiles reference key variables, never key values
        assert "gsk_" not in path.read_text() and "sk-" not in path.read_text()
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    s = get_settings("hybrid")
    assert (s.tier1.model, s.tier2.model, s.tier2.api_key, s.profile) == \
        ("qwen3.5:9b", "openai/gpt-oss-120b", "test-key", "hybrid")
    assert s.tier1.extra_body == {"reasoning_effort": "none"} and s.tier2.params_b == 5.1
    assert get_settings("single-gpu").tier2.model == "qwen2.5:14b"


def test_cost_only_when_prices_configured(settings, held_out):
    from dataclasses import replace

    t = held_out["std_en_07"]
    unpriced = OptiRoutePipeline(settings, SimClient(settings), PipelineFlags(use_cache=False)).process(t)
    assert unpriced.cost_usd is None  # no prices -> no invented dollar figures
    priced = replace(settings, tier1=replace(settings.tier1, price_in_per_mtok=1.0, price_out_per_mtok=2.0),
                     tier2=replace(settings.tier2, price_in_per_mtok=10.0, price_out_per_mtok=20.0))
    r = OptiRoutePipeline(priced, SimClient(priced), PipelineFlags(use_cache=False)).process(t)
    c = r.calls[0]
    assert r.cost_usd == pytest.approx((c.prompt_tokens * 1.0 + c.completion_tokens * 2.0) / 1e6)


def test_settings_precedence_shell_over_profile(monkeypatch):
    from src.config import get_settings

    monkeypatch.setenv("TIER1_MODEL", "one-off-model")  # explicit shell override wins over the profile
    s = get_settings("hybrid")
    assert s.tier1.model == "one-off-model" and s.tier2.model == "openai/gpt-oss-120b"
