"""OptiRoute demo server: interactive cascade runner + recorded benchmark viewer.

  .venv/bin/uvicorn app:app --port 8000

Dataset tasks replay recorded real model outputs (`data/recorded_runs.jsonl`); anything not recorded
(custom input, or `live=true`) is sent to the configured Ollama/Groq endpoints.
"""
from __future__ import annotations

import hashlib
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import ROOT, get_settings
from src.ground_truth_scorer import score
from src.llm_client import LiveClient, ReplayClient
from src.normalize import is_arabic
from src.pipeline import OptiRoutePipeline, PipelineFlags
from src.schemas import LLMResponse, Task

RECORDED = ROOT / "data" / "recorded_runs.jsonl"
RESULTS = ROOT / "results"
PRESETS = [
    ("Easy · English", "std_en_07"),
    ("Easy · Arabic", "std_ar_01"),
    ("Cache seed · #INV-8841", "std_en_04"),
    ("Paraphrase hit · #INV-8841", "cache_en_hit"),
    ("Near-duplicate trap · #INV-8842", "cache_en_trap"),
    ("Low-overlap → escalation", "llo_en_01"),
    ("Multi-hop policy math (JSON)", "mh_en_02"),
    ("Arabic spelling variants", "aro_ar_03"),
]

settings = get_settings()
TASKS = {d["task_id"]: Task.from_dict(d) for d in
         json.loads((ROOT / "data" / "held_out_test.json").read_text(encoding="utf-8"))}


class DemoClient:
    """Replay recorded outputs when available; otherwise (or when forced) call the live endpoints."""

    def __init__(self):
        self.replay = ReplayClient(RECORDED) if RECORDED.exists() else None
        self._live: LiveClient | None = None
        self.force_live = False

    @property
    def live(self) -> LiveClient:
        if self._live is None:
            self._live = LiveClient(settings)
        return self._live

    def complete(self, task: Task, variant: str, tier: int, chunks: list[str]) -> LLMResponse:
        if self.replay and not self.force_live:
            try:
                return self.replay.complete(task, variant, tier, chunks)
            except (KeyError, ValueError):
                pass
        try:
            return self.live.complete(task, variant, tier, chunks)
        except Exception as e:  # surfaced to the UI as a 503
            raise HTTPException(503, f"Live model unavailable for tier {tier} ({e.__class__.__name__}). "
                                     "Start Ollama or pick a preset case (replayed from recordings).")


client = DemoClient()
state = {"pipe": OptiRoutePipeline(settings, client, PipelineFlags())}

app = FastAPI(title="OptiRoute")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


class RunRequest(BaseModel):
    task_id: str | None = None
    query: str | None = None
    context: str | None = None
    json_keys: str | None = None  # "key:number, key2:string"
    live: bool = False


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/presets")
def presets():
    return {"presets": [{"label": label, **_task_view(TASKS[tid])} for label, tid in PRESETS],
            "all": [_task_view(t) for t in TASKS.values()],
            "models": {"tier1": [settings.tier1.model, settings.tier1.params_b],
                       "tier2": [settings.tier2.model, settings.tier2.params_b]},
            "recorded": RECORDED.exists()}


@app.post("/api/reset")
def reset():
    state["pipe"] = OptiRoutePipeline(settings, client, PipelineFlags())
    return {"ok": True, "tau": state["pipe"].adapt.tau}


@app.post("/api/run")
def run(req: RunRequest):
    if req.task_id:
        if req.task_id not in TASKS:
            raise HTTPException(404, "unknown task")
        task = TASKS[req.task_id]
    else:
        if not req.query or not req.context:
            raise HTTPException(400, "query and context are required")
        chunks = [c.strip() for c in req.context.replace("\r", "").split("\n\n") if c.strip()]
        schema = None
        if req.json_keys:
            schema = {k.split(":")[0].strip(): (k.split(":")[1].strip() if ":" in k else "string")
                      for k in req.json_keys.split(",") if k.strip()}
        tid = "custom-" + hashlib.sha1((req.query + req.context).encode()).hexdigest()[:8]
        task = Task(task_id=tid, language="ar" if is_arabic(req.query) else "en", category="custom",
                    query=req.query, context=chunks, expected_answer=None, schema=schema)
    pipe: OptiRoutePipeline = state["pipe"]
    client.force_live = req.live
    try:
        res = pipe.process(task)
    finally:
        client.force_live = False
    # Always-Tier-2-on-raw-context baseline for the same task (recorded), for the savings readout
    baseline = None
    if client.replay and (task.task_id, "raw", 2) in client.replay.records:
        rec = client.replay.records[(task.task_id, "raw", 2)]
        baseline = {"tokens": rec["prompt_tokens"] + rec["completion_tokens"],
                    "compute": (rec["prompt_tokens"] + rec["completion_tokens"]) / 1000 * settings.tier2.params_b,
                    "latency_ms": rec["latency_ms"]}
    graded = None
    if task.expected_answer is not None:
        s = score(task, res.answer)
        graded = {"correct": s.correct, "expected": task.expected_answer}
    return {
        "task": _task_view(task),
        "answer": res.answer, "final_tier": res.final_tier, "path": res.path, "cache_event": res.cache_event,
        "tau": res.tau, "tau_after": pipe.adapt.tau, "trace": res.trace,
        "route": None if not res.route else {"complexity": res.route.complexity, "signals": res.route.signals,
                                             "tier": res.route.tier, "tau": res.route.tau},
        "verifier": None if not res.verifier else {"passed": res.verifier.passed,
                                                   "failed": res.verifier.failed_checks,
                                                   "ms": res.verifier.elapsed_ms},
        "calls": [{"tier": c.tier, "model": c.model_id, "variant": c.variant, "prompt_tokens": c.prompt_tokens,
                   "completion_tokens": c.completion_tokens, "latency_ms": c.latency_ms, "output": c.raw_output,
                   "source": c.source} for c in res.calls],
        "tokens": res.total_tokens, "compute": res.compute_units, "latency_ms": res.latency_ms,
        "overhead_ms": res.overhead_ms, "baseline": baseline, "graded": graded,
        "cache_size": len(pipe.cache.entries),
    }


@app.get("/api/results")
def results():
    path = RESULTS / "metrics.json"
    if not path.exists():
        raise HTTPException(404, "Run `evaluate.py --replay data/recorded_runs.jsonl` first.")
    m = json.loads(path.read_text(encoding="utf-8"))
    out = {k: m.get(k) for k in ("mode", "source", "cumulative", "loo", "verifier", "noise_scaling", "wave",
                                 "stress", "cross_model")}
    return out


def _task_view(t: Task) -> dict:
    return {"task_id": t.task_id, "language": t.language, "category": t.category, "query": t.query,
            "context": t.context, "schema": t.schema}
