"""OptiRoute evaluation: record real model runs once, replay them deterministically for all ablations.

  python evaluate.py --sim                                   # offline smoke run (synthetic model, NOT headline)
  OPTIROUTE_MODE=live python evaluate.py --record --output data/recorded_runs.jsonl
  python evaluate.py --replay data/recorded_runs.jsonl       # headline tables -> results/summary_tables.md

Auxiliary recordings (Tier-1 swapped, Tier-2 reused from the main recording):
  TIER1_MODEL=qwen2.5:0.5b TIER1_PARAMS_B=0.5 python evaluate.py --record --tiers 1 \\
      --reuse data/recorded_runs.jsonl --output data/recorded_runs_t1-qwen2.5-0.5b.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from src import pruner as pruner_mod
from src.adaptation import p95
from src.config import ROOT, get_settings, list_profiles
from src.ground_truth_scorer import score
from src.llm_client import LiveClient, ReplayClient, SimClient, bind_recorded_models, build_messages, prompt_sha
from src.pipeline import OptiRoutePipeline, PipelineFlags
from src.schemas import CascadeResult, Task
from src.verifier import verify

DATA = ROOT / "data"
OFF = dict(use_pruner=False, use_cache=False, use_verifier=False, use_adaptation=False)

CUMULATIVE = {
    "1_Always_Tier2_Raw": PipelineFlags(**OFF, force_tier=2),
    "2_Always_Tier1_Raw": PipelineFlags(**OFF, force_tier=1),
    "3_Router_Only": PipelineFlags(**OFF),
    "4_Pruner_Plus_Router": PipelineFlags(**{**OFF, "use_pruner": True}),
    "5_OptiRoute_Full": PipelineFlags(),
}
LOO = {
    "Full_OptiRoute": PipelineFlags(),
    "LOO_No_Cache": PipelineFlags(use_cache=False),
    "LOO_No_Cache_Entity_Guard": PipelineFlags(entity_guard=False),
    "LOO_No_Pruner": PipelineFlags(use_pruner=False),
    "LOO_No_Verifier": PipelineFlags(use_verifier=False),
    "LOO_No_Adaptation": PipelineFlags(use_adaptation=False),
}
# Every (variant, tier) prompt any configuration above can issue.
RECORD_PLAN = [("raw", 1), ("raw", 2), ("pruned", 1), ("pruned", 2), ("relaxed", 2)]


def load_tasks(name: str) -> list[Task]:
    return [Task.from_dict(d) for d in json.loads((DATA / name).read_text(encoding="utf-8"))]


# ---------------------------------------------------------------- metrics
def summarize(tasks: list[Task], results: list[CascadeResult]) -> dict:
    by_id = {t.task_id: t for t in tasks}
    scores = [score(by_id[r.task_id], r.answer) for r in results]
    correct = sum(s.correct for s in scores)
    compute = sum(r.compute_units for r in results)
    lat = [r.latency_ms for r in results]
    tiers = Counter(r.final_tier for r in results)
    traps = [r for r in results if by_id[r.task_id].expect_cache_hit is False]
    overhead = {k: statistics.mean(r.overhead_ms[k] for r in results) for k in results[0].overhead_ms}
    per_cat: dict[str, list[bool]] = defaultdict(list)
    for r, s in zip(results, scores):
        per_cat[by_id[r.task_id].category].append(s.correct)
    return {
        "n": len(results),
        "accuracy": correct / len(results),
        "correct": correct,
        "total_tokens": sum(r.total_tokens for r in results),
        "tier2_tokens": sum(c.total_tokens for r in results for c in r.calls if c.tier == 2),
        "cost_usd": (sum(r.cost_usd for r in results) if all(r.cost_usd is not None for r in results) else None),
        "tasks_reaching_tier2": sum(any(c.tier == 2 for c in r.calls) for r in results),
        "compute_units": compute,
        "cps": compute / correct if correct else float("inf"),
        "p50_ms": statistics.median(lat),
        "p95_ms": p95(lat),
        "tier0": tiers[0], "tier1": tiers[1], "tier2": tiers[2],
        "escalations": sum(any(p.startswith("escalate") for p in r.path) for r in results),
        "cache_hits": sum(r.cache_event == "hit" for r in results),
        "blocked_traps": sum(r.cache_event == "blocked_trap" for r in results),
        "false_cache_hits": sum(r.final_tier == 0 for r in traps),
        "n_traps": len(traps),
        "overhead_ms": overhead,
        "per_category": {k: sum(v) / len(v) for k, v in per_cat.items()},
    }


def verifier_pr(tasks: list[Task], client, variant: str = "pruned") -> dict:
    """Verifier precision/recall over every held-out Tier-1 output (ground truth used only to grade)."""
    tp = fp = fn = tn = 0
    for t in tasks:
        chunks = pruner_mod.prune(t.query, t.context).kept if variant == "pruned" else t.context
        out = client.complete(t, variant, 1, chunks).raw_output
        flagged = not verify(t.query, pruner_mod.join_context(chunks), out, t.schema).passed
        wrong = not score(t, out).correct
        tp += flagged and wrong
        fp += flagged and not wrong
        fn += (not flagged) and wrong
        tn += (not flagged) and not wrong
    return {"variant": variant, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "tier1_error_rate": (tp + fn) / len(tasks)}


def run_config(settings, client, flags, tasks) -> list[CascadeResult]:
    return OptiRoutePipeline(settings, client, flags).run(tasks)


# ---------------------------------------------------------------- experiments
def noise_scaling(tasks: list[Task], settings) -> list[dict]:
    """Pruner evidence recall + token savings with 1x/2x/4x distractor chunks from other documents."""
    pool = json.loads((DATA / "distractor_pool.json").read_text(encoding="utf-8"))
    rows = []
    for mult in (0, 1, 2, 4):
        recall, before, after, ms = [], 0, 0, []
        for i, t in enumerate(tasks):
            others = [c for doc, chunks in pool.items() if chunks != t.context for c in chunks]
            n = mult * len(t.context)
            distract = [others[(i * 7 + j * 5) % len(others)] for j in range(n)]
            # deterministic interleave: original chunks keep their relative order
            ctx = list(t.context)
            for j, d in enumerate(distract):
                ctx.insert((j * 3) % (len(ctx) + 1), d)
            p = pruner_mod.prune(t.query, ctx, settings.cascade.prune_theta, settings.cascade.prune_min_k)
            kept = "\n".join(p.kept)
            recall.append(all(e in kept for e in t.evidence))
            before += p.tokens_before
            after += p.tokens_after
            ms.append(p.elapsed_ms)
        rows.append({"noise": f"{mult}x", "evidence_recall": sum(recall) / len(recall),
                     "tokens_before": before, "tokens_after": after, "reduction": 1 - after / before,
                     "prune_ms_mean": statistics.mean(ms)})
    return rows


def wave_stream(tasks: list[Task]) -> list[Task]:
    std = [t for t in tasks if t.category == "standard"]
    hard = [t for t in tasks if t.category in ("multi_hop", "low_lexical_overlap", "arabic_orthographic_variants")]
    # easy (10) -> sustained hard burst (every hard task twice) -> easy recovery (20); cache is off, so repeated
    # tasks are re-inferred
    return std[:10] + hard * 2 + std[10:30]


def wave_experiment(settings, client, tasks) -> dict:
    stream = wave_stream(tasks)
    n_hard = 2 * sum(t.category in ("multi_hop", "low_lexical_overlap", "arabic_orthographic_variants") for t in tasks)
    out = {"phases": {"easy_1": [0, 10], "hard_burst": [10, 10 + n_hard], "easy_2": [10 + n_hard, len(stream)]}}
    for name, flags in (("adaptive", PipelineFlags(use_cache=False)),
                        ("fixed_tau", PipelineFlags(use_cache=False, use_adaptation=False)),
                        ("adaptive_audit", PipelineFlags(use_cache=False, shadow_audit_every=4)),
                        ("fixed_tau_audit", PipelineFlags(use_cache=False, use_adaptation=False,
                                                          shadow_audit_every=4))):
        pipe = OptiRoutePipeline(settings, client, flags)
        res = pipe.run(stream)
        out[name] = {"summary": summarize(stream, res),
                     "trace": [{"step": i, "task_id": r.task_id, "tau_used": r.tau, "final_tier": r.final_tier,
                                "path": r.path, "correct": score(t, r.answer).correct,
                                **pipe.adapt.history[i]} for i, (t, r) in enumerate(zip(stream, res))]}
    return out


# ---------------------------------------------------------------- reporting
def fmt_row(name: str, m: dict, base: dict) -> str:
    red = 1 - m["total_tokens"] / base["total_tokens"] if base["total_tokens"] else 0
    cred = 1 - m["compute_units"] / base["compute_units"] if base["compute_units"] else 0
    return (f"| {name} | {m['accuracy']:.1%} ({m['correct']}/{m['n']}) | {m['total_tokens']:,} | {red:+.1%} | "
            f"{m['compute_units']:.2f} | {cred:+.1%} | {m['cps']:.3f} | {m['p50_ms']:.0f} | {m['p95_ms']:.0f} | "
            f"{m['tier0']}/{m['tier1']}/{m['tier2']} | {m['escalations']} | "
            f"{m['false_cache_hits']}/{m['n_traps']} |")


HEADER = ("| Config | Accuracy | Tokens | Δ Tokens vs T2 | Compute (kTok·B) | Δ Compute | CPS (kTok·B / success) "
          "| P50 ms | P95 ms | Final tier 0/1/2 | Escalations | False cache hits / traps |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|---|")


def write_report(out_dir: Path, mode: str, settings, cum, loo, vpr, noise, wave, source_note: str) -> Path:
    base = cum["1_Always_Tier2_Raw"]
    lines = [f"# OptiRoute — Measured Results ({mode.upper()})", ""]
    if mode == "sim":
        lines += ["> **SIMULATED RUN — synthetic model outputs and synthetic LLM latency. Control-flow smoke "
                  "test only; NOT valid for headline numbers.**", ""]
    lines += [f"- Source: {source_note}",
              f"- Tier-1: `{settings.tier1.model}` ({settings.tier1.params_b:g}B) · Tier-2: `{settings.tier2.model}` "
              f"({settings.tier2.params_b:g}B) · tokenizer: tiktoken `{settings.tokenizer_encoding}`",
              f"- Held-out cases: {base['n']} · Compute = Σ tokens/1000 × active params (B); Tier-0 cache = 0",
              "- Latency = recorded LLM latency + measured CPU overhead (`time.perf_counter`)",
              "- Iteration disclosure (after inspecting held-out failures): (1) the verifier's refusal detector was "
              "generalized from a phrase list to a negation+information-verb pattern; (2) numeric grounding now "
              "requires verbatim numbers for lookup questions and allows derived (+ − × ÷ %) numbers only when the "
              "query asks for computation. Both were re-checked for false positives on all 540 recorded Tier-1 "
              "outputs across 3 models. A later code review (3) stopped the boilerplate filter deleting paragraphs that "
              "merely contain the word \"confidential\" and made its Arabic footer phrases match, and (4) limited the "
              "cache to answers that pass the runtime checks; only the affected prompts were re-recorded. Dataset "
              "answers and model outputs were never edited.", "",
              "## Table 1 — Cumulative Progression", "", HEADER]
    lines += [fmt_row(k, v, base) for k, v in cum.items()]
    if full_cost := cum["5_OptiRoute_Full"].get("cost_usd"):
        lines += ["", f"Cost (USD, from configured prices): always-Tier-2 ${base['cost_usd']:.4f} → "
                      f"OptiRoute ${full_cost:.4f}"]
    lines += ["", "## Table 2 — Leave-One-Out Ablation", "", HEADER]
    lines += [fmt_row(k, v, base) for k, v in loo.items()]

    cats = sorted({c for m in cum.values() for c in m["per_category"]})
    lines += ["", "## Accuracy by Category", "", "| Config | " + " | ".join(cats) + " |",
              "|---|" + "---|" * len(cats)]
    for k, m in {**cum, **{k: v for k, v in loo.items() if k != "Full_OptiRoute"}}.items():
        lines.append(f"| {k} | " + " | ".join(f"{m['per_category'].get(c, 0):.0%}" for c in cats) + " |")

    lines += ["", "## Runtime Verifier Precision / Recall (graded offline by ground-truth scorer)", "",
              "| Tier-1 context | Tier-1 error rate | Escalations (TP+FP) | TP | FP | FN | TN | Precision | Recall |",
              "|---|---|---|---|---|---|---|---|---|"]
    for v in vpr:
        lines.append(f"| {v['variant']} | {v['tier1_error_rate']:.1%} | {v['tp'] + v['fp']} | {v['tp']} | {v['fp']} | "
                     f"{v['fn']} | {v['tn']} | {v['precision']:.1%} | {v['recall']:.1%} |")

    full = cum["5_OptiRoute_Full"]
    lines += ["", "## Measured Component Overhead (mean ms per task, Full cascade)", "",
              "| Pruner | Cache | Router | Verifier | Total CPU | P50 end-to-end |", "|---|---|---|---|---|---|",
              "| " + " | ".join(f"{full['overhead_ms'][k]:.2f}" for k in ("pruner", "cache", "router", "verifier"))
              + f" | {sum(full['overhead_ms'].values()):.2f} | {full['p50_ms']:.0f} |"]

    lines += ["", "## Experiment 1 — Bilingual Context Noise Scaling (pruner only, CPU)", "",
              "| Distractors | Evidence recall | Tokens before | Tokens after | Reduction | Prune ms |",
              "|---|---|---|---|---|---|"]
    for r in noise:
        lines.append(f"| {r['noise']} | {r['evidence_recall']:.1%} | {r['tokens_before']:,} | {r['tokens_after']:,} | "
                     f"{r['reduction']:.1%} | {r['prune_ms_mean']:.2f} |")

    lines += ["", "## Experiment 2 — Wave Workload (easy → hard burst → easy), cache off", "",
              f"Phases (step index): {wave['phases']}", "",
              "| Controller | Accuracy | Compute | CPS | Escalations | Final tier 0/1/2 | τ min | τ final |",
              "|---|---|---|---|---|---|---|---|"]
    for name in ("adaptive", "fixed_tau", "adaptive_audit", "fixed_tau_audit"):
        m, tr = wave[name]["summary"], wave[name]["trace"]
        lines.append(f"| {name} | {m['accuracy']:.1%} | {m['compute_units']:.2f} | {m['cps']:.3f} | {m['escalations']} | "
                     f"{m['tier0']}/{m['tier1']}/{m['tier2']} | {min(x['tau'] for x in tr):.2f} | {tr[-1]['tau']:.2f} |")
    tr = wave["adaptive"]["trace"]
    rates = [x["pass_rate"] for x in tr if x["pass_rate"] is not None]
    cfg = settings.cascade
    lines += ["", "τ(t) trace (adaptive): " + " ".join(f"{x['tau']:.2f}" for x in tr), "",
              f"Trigger check: min windowed Tier-1 verifier pass rate = {min(rates) if rates else float('nan'):.0%} "
              f"(tighten if < {cfg.tighten_pass_rate:.0%}); max windowed P95 = {max(x['p95_ms'] for x in tr):.0f} ms "
              f"(tighten if > SLO {cfg.latency_slo_ms:.0f} ms)."]

    full_l, noguard = loo["Full_OptiRoute"], loo["LOO_No_Cache_Entity_Guard"]
    lines += ["", "## Experiment 3 — Near-Duplicate Adversarial Audit", "",
              "| Config | Cache hits | Blocked traps | False cache hits / traps | Cache-category accuracy |",
              "|---|---|---|---|---|",
              f"| With entity guard | {full_l['cache_hits']} | {full_l['blocked_traps']} | "
              f"{full_l['false_cache_hits']}/{full_l['n_traps']} | {full_l['per_category'].get('cache_near_duplicates', 0):.0%} |",
              f"| Without entity guard | {noguard['cache_hits']} | {noguard['blocked_traps']} | "
              f"{noguard['false_cache_hits']}/{noguard['n_traps']} | "
              f"{noguard['per_category'].get('cache_near_duplicates', 0):.0%} |", ""]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary_tables.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def rel(path: Path) -> str:
    p = path.resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(path)


settings_for = bind_recorded_models  # reports name the models that produced a recording


def stress_experiment(settings, path: Path, tasks: list[Task]) -> dict:
    """Experiment 4: the same wave workload with a much weaker real Tier-1 model (genuine verifier failures)."""
    client = ReplayClient(path)
    s = settings_for(settings, client)
    out = wave_experiment(s, client, tasks)
    out["tier1_model"], out["source"] = s.tier1.model, rel(path)
    return out


def cross_model(settings, paths: list[Path], tasks: list[Task]) -> list[dict]:
    """Experiment 5: identical cascade, different recorded model pairs."""
    rows = []
    for path in paths:
        client = ReplayClient(path)
        s = settings_for(settings, client)
        res = {k: summarize(tasks, run_config(s, client, CUMULATIVE[k], tasks))
               for k in ("1_Always_Tier2_Raw", "2_Always_Tier1_Raw", "5_OptiRoute_Full")}
        profile = next((p["name"] for p in list_profiles() if p["recording"] and (ROOT / p["recording"]).resolve()
                        == path.resolve()), None)
        rows.append({"file": path.name, "profile": profile, "tier1": f"{s.tier1.model} ({s.tier1.params_b:g}B)",
                     "tier2": f"{s.tier2.model} ({s.tier2.params_b:g}B)", "configs": res,
                     "verifier": verifier_pr(tasks, client, "pruned")})
    return rows


def extra_sections(stress: dict | None, xmodel: list[dict], cfg) -> list[str]:
    lines = []
    if stress:
        lines += ["## Experiment 4 — Real Tier-1 Degradation Stress (adaptation under genuine failures)", "",
                  f"Same wave workload as Experiment 2, but Tier-1 is the much weaker real model "
                  f"`{stress['tier1_model']}` (recorded: `{stress['source']}`).", "",
                  "| Controller | Accuracy | Compute | CPS | Tier-1 attempts | Escalations | Final tier 0/1/2 | τ min | τ final |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for name in ("adaptive", "fixed_tau", "adaptive_audit", "fixed_tau_audit"):
            m, tr = stress[name]["summary"], stress[name]["trace"]
            t1_attempts = sum(any(p.startswith("tier1") for p in x["path"]) for x in tr)
            lines.append(f"| {name} | {m['accuracy']:.1%} | {m['compute_units']:.2f} | {m['cps']:.3f} | {t1_attempts} | "
                         f"{m['escalations']} | {m['tier0']}/{m['tier1']}/{m['tier2']} | "
                         f"{min(x['tau'] for x in tr):.2f} | {tr[-1]['tau']:.2f} |")
        for name in ("adaptive", "adaptive_audit"):
            tr = stress[name]["trace"]
            acts = [f"{x['step']}:{x['action']}" for x in tr if x["action"] != "hold"]
            lines += ["", f"τ(t) trace ({name}): " + " ".join(f"{x['tau']:.2f}" for x in tr), "",
                      f"Controller actions ({name}, step:action): {' '.join(acts) or 'none'}"]
        lines += ["", "`_audit` = every 4th verified Tier-1 answer is also sent to Tier-2; disagreement counts as a "
                  "Tier-1 failure for the adaptation window (and the Tier-2 answer is used). Audit calls are "
                  "included in compute.", ""]
    if xmodel:
        lines += ["## Experiment 5 — Cross-Model Generalization (same cascade, different recorded models)", "",
                  "| Profile | Recording | Tier-1 | Tier-2 | Always-T2 acc | Always-T1 acc | Full acc | Full compute Δ vs T2 "
                  "| CPS T2 → Full | Verifier P / R | Tasks sent to Tier-2 | Tier-2 tokens vs always-T2 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in xmodel:
            c = r["configs"]
            t2, t1, full = c["1_Always_Tier2_Raw"], c["2_Always_Tier1_Raw"], c["5_OptiRoute_Full"]
            v = r["verifier"]
            lines.append(f"| {('**' + r['profile'] + '**') if r.get('profile') else '—'} | `{r['file']}` | {r['tier1']} | {r['tier2']} | {t2['accuracy']:.1%} | {t1['accuracy']:.1%} | "
                         f"**{full['accuracy']:.1%}** | {full['compute_units'] / t2['compute_units'] - 1:+.1%} | "
                         f"{t2['cps']:.2f} → {full['cps']:.2f} | {v['precision']:.0%} / {v['recall']:.0%} | "
                         f"{full['tasks_reaching_tier2']}/{full['n']} | "
                         f"{full['tier2_tokens'] / t2['tier2_tokens'] - 1:+.1%} |")
        lines.append("")
    return lines


def evaluate(mode: str, client, settings, out_dir: Path, source_note: str,
             stress_path: Path | None = None, compare: list[Path] | None = None) -> Path:
    tasks = load_tasks("held_out_test.json")
    settings = settings_for(settings, client)
    t0 = time.perf_counter()
    cum_res = {k: run_config(settings, client, f, tasks) for k, f in CUMULATIVE.items()}
    loo_res = {k: run_config(settings, client, f, tasks) for k, f in LOO.items()}
    cum = {k: summarize(tasks, v) for k, v in cum_res.items()}
    loo = {k: summarize(tasks, v) for k, v in loo_res.items()}
    vpr = [verifier_pr(tasks, client, "pruned"), verifier_pr(tasks, client, "raw")]
    noise = noise_scaling(tasks, settings)
    wave = wave_experiment(settings, client, tasks)
    stress = stress_experiment(settings, stress_path, tasks) if stress_path else None
    xmodel = cross_model(settings, [client.path] + list(compare or []), tasks) \
        if compare and isinstance(client, ReplayClient) else []
    elapsed = time.perf_counter() - t0
    report = write_report(out_dir, mode, settings, cum, loo, vpr, noise, wave, source_note)
    extra = extra_sections(stress, xmodel, settings.cascade)
    if extra:
        report.write_text(report.read_text(encoding="utf-8") + "\n" + "\n".join(extra), encoding="utf-8")
    per_task = {k: [{"task_id": r.task_id, "answer": r.answer, "final_tier": r.final_tier, "path": r.path,
                     "correct": score(next(t for t in tasks if t.task_id == r.task_id), r.answer).correct,
                     "tokens": r.total_tokens, "compute_units": r.compute_units, "latency_ms": r.latency_ms,
                     "cache_event": r.cache_event, "tau": r.tau,
                     "complexity": r.route.complexity if r.route else None,
                     "verifier_failed": r.verifier.failed_checks if r.verifier else None}
                    for r in v] for k, v in {**cum_res, **loo_res}.items()}
    (out_dir / "metrics.json").write_text(json.dumps(
        {"mode": mode, "source": source_note, "eval_seconds": elapsed,
         "models": {"tier1": [settings.tier1.model, settings.tier1.params_b],
                    "tier2": [settings.tier2.model, settings.tier2.params_b]},
         "cumulative": cum, "loo": loo, "verifier": vpr, "noise_scaling": noise, "wave": wave,
         "stress": stress, "cross_model": xmodel}, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    (out_dir / "per_task.json").write_text(json.dumps(per_task, indent=1, ensure_ascii=False), encoding="utf-8")
    print(report.read_text(encoding="utf-8"))
    print(f"\nEvaluation wall time: {elapsed:.2f}s -> {report}")
    return report


def prompt_chunks(task: Task, variant: str, settings) -> list[str]:
    """The context chunks the cascade sends for a variant."""
    if variant == "raw":
        return task.context
    p = pruner_mod.prune(task.query, task.context, settings.cascade.prune_theta, settings.cascade.prune_min_k)
    return p.kept if variant == "pruned" else pruner_mod.relaxed(task.query, task.context, p, settings.cascade.relaxed_k).kept


def record(settings, output: Path, fresh: bool, tiers: set[int], reuse: Path | None) -> None:
    """Record live calls for `tiers`; records for other tiers are copied verbatim from `reuse`."""
    tasks = load_tasks("held_out_test.json")
    if fresh and output.exists():
        output.unlink()
    current = {(t.task_id, v): prompt_sha(build_messages(t, prompt_chunks(t, v, settings)))
               for t in tasks for v in {variant for variant, _ in RECORD_PLAN}}
    is_stale = lambda r: current.get((r["task_id"], r["variant"])) != r.get("prompt_sha")
    done = set()
    if output.exists():
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
        fresh_rows = [r for r in rows if not is_stale(r)]
        if len(fresh_rows) != len(rows):  # the prompts changed since recording: those outputs no longer apply
            print(f"Dropping {len(rows) - len(fresh_rows)} stale records from {output}")
            output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in fresh_rows), encoding="utf-8")
        done = {(r["task_id"], r["variant"], r["tier"]) for r in fresh_rows}
    if reuse:
        with output.open("a", encoding="utf-8") as f:
            for line in reuse.read_text(encoding="utf-8").splitlines():
                r = json.loads(line) if line.strip() else None
                if r and r["tier"] not in tiers and not is_stale(r) and (r["task_id"], r["variant"], r["tier"]) not in done:
                    f.write(json.dumps({**r, "reused_from": reuse.name}, ensure_ascii=False) + "\n")
                    done.add((r["task_id"], r["variant"], r["tier"]))
    client = LiveClient(settings, record_path=output)
    print(f"Warmup (model load, excluded from timings): {client.warmup()}")
    todo = [(t, v, tier) for t in tasks for v, tier in RECORD_PLAN
            if tier in tiers and (t.task_id, v, tier) not in done]
    print(f"Recording {len(todo)} calls ({len(done)} already present) -> {output}")
    for i, (t, variant, tier) in enumerate(todo, 1):
        r = client.complete(t, variant, tier, prompt_chunks(t, variant, settings))
        print(f"[{i}/{len(todo)}] {t.task_id} {variant} T{tier} {r.latency_ms:.0f}ms "
              f"{r.prompt_tokens}+{r.completion_tokens}tok :: {r.raw_output[:70]!r}", flush=True)


def assemble(output: Path, tier1_from: Path, tier2_from: Path) -> None:
    """Combine Tier-1 records of one recording with Tier-2 records of another (no model calls).

    Valid because prompts do not depend on the tier: every (task, variant) prompt hash must match.
    """
    def load(path: Path, tier: int) -> list[dict]:
        return [r for r in (json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip())
                if r["tier"] == tier]
    t1, t2 = load(tier1_from, 1), load(tier2_from, 2)
    shas = {(r["task_id"], r["variant"]): r["prompt_sha"] for r in t1 + t2}
    for r in t1 + t2:
        if shas[(r["task_id"], r["variant"])] != r["prompt_sha"]:
            raise ValueError(f"prompt mismatch for {r['task_id']}/{r['variant']}; cannot assemble")
    with output.open("w", encoding="utf-8") as f:
        for r, src in [(r, tier1_from) for r in t1] + [(r, tier2_from) for r in t2]:
            f.write(json.dumps({**r, "reused_from": src.name}, ensure_ascii=False) + "\n")
    print(f"assembled {output.name}: {len(t1)} Tier-1 ({tier1_from.name}) + {len(t2)} Tier-2 ({tier2_from.name})")


STRESS_DEFAULT = DATA / "recorded_runs_t1-qwen2.5-0.5b.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sim", action="store_true", help="offline synthetic model (control-flow smoke test)")
    g.add_argument("--record", action="store_true", help="call live models and append to --output")
    g.add_argument("--replay", type=Path, metavar="JSONL", help="replay recorded runs (headline numbers)")
    g.add_argument("--assemble", nargs=3, type=Path, metavar=("OUT", "TIER1_FROM", "TIER2_FROM"),
                   help="build a recording from Tier-1 of one file and Tier-2 of another (no model calls)")
    ap.add_argument("--output", type=Path, default=DATA / "recorded_runs.jsonl")
    ap.add_argument("--fresh", action="store_true", help="with --record: delete existing recordings first")
    ap.add_argument("--tiers", default="1,2", help="with --record: tiers to call live, e.g. '1'")
    ap.add_argument("--reuse", type=Path, help="with --record: copy the other tiers' records from this file")
    ap.add_argument("--stress", type=Path, help=f"weak-Tier-1 recording for Experiment 4 "
                                                f"(default: {STRESS_DEFAULT.name} if present)")
    ap.add_argument("--compare", type=Path, nargs="*", help="extra recordings for Experiment 5 "
                                                            "(default: data/recorded_runs_t[12]-*.jsonl)")
    ap.add_argument("--results", type=Path, default=None, help="results directory")
    args = ap.parse_args()
    settings = get_settings()

    if args.assemble:
        assemble(*args.assemble)
        return 0
    if args.sim:
        evaluate("sim", SimClient(settings), settings, args.results or ROOT / "results" / "sim",
                 "SimClient (synthetic)")
        return 0
    if args.record:
        record(settings, args.output, args.fresh, {int(t) for t in args.tiers.split(",")}, args.reuse)
        if args.output != DATA / "recorded_runs.jsonl":
            return 0  # auxiliary recordings are consumed by the main replay via --stress / --compare
        path = args.output
    else:
        path = args.replay
    stress = args.stress or (STRESS_DEFAULT if STRESS_DEFAULT.exists() else None)
    compare = args.compare if args.compare is not None else sorted(DATA.glob("recorded_runs_t[12]-*.jsonl"))
    evaluate("replay", ReplayClient(path), settings, args.results or ROOT / "results",
             f"replay of `{rel(path)}`",
             stress_path=stress, compare=compare)
    return 0


if __name__ == "__main__":
    sys.exit(main())
