"""OptiRoute demo server: interactive cascade runner + recorded benchmark viewer.

  .venv/bin/uvicorn app:app --port 8000

The active deployment profile (profiles/*.env, switchable in the UI) decides the models. Preset tasks
replay that profile's recorded real outputs; anything not recorded (custom input, or `live=true`) is sent
to the profile's live endpoints (local Ollama and/or a cloud API whose key is read from `.env`).
"""
from __future__ import annotations

import hashlib
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import ROOT, Settings, get_settings, list_profiles
from src.ground_truth_scorer import score
from src.llm_client import LiveClient, ReplayClient, bind_recorded_models
from src.normalize import is_arabic
from src.pipeline import OptiRoutePipeline, PipelineFlags
from src.schemas import LLMResponse, Task

DEFAULT_PROFILE = "single-gpu"
RESULTS = ROOT / "results"
PRESETS = [
    ("Easy · English", "std_en_07"),
    ("Easy · Arabic", "std_ar_01"),
    ("Cache seed · #INV-8841", "std_en_04"),
    ("Paraphrase hit · #INV-8841", "cache_en_hit"),
    ("Near-duplicate trap · #INV-8842", "cache_en_trap"),
    ("Low-overlap → escalation", "llo_en_01"),
    ("Multi-hop policy math (JSON)", "mh_en_02"),
    ("Multi-hop Arabic · late penalty", "mh_ar_01"),
    ("Arabic spelling variants", "aro_ar_03"),
]

TASKS = {d["task_id"]: Task.from_dict(d) for d in
         json.loads((ROOT / "data" / "held_out_test.json").read_text(encoding="utf-8"))}


class DemoClient:
    """Replay recorded outputs when available; otherwise (or when forced) call the live endpoints."""

    def __init__(self, settings: Settings, recording):
        self.settings = settings
        self.replay = ReplayClient(recording) if recording and recording.exists() else None
        self._live: LiveClient | None = None
        self.force_live = False

    @property
    def live(self) -> LiveClient:
        if self._live is None:
            self._live = LiveClient(self.settings)
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
                                     "Start Ollama / set the cloud key in .env, or pick a preset case "
                                     "(replayed from recordings).")


state: dict = {}


def activate(name: str) -> None:
    """Switch deployment profile: settings, recording, labels and a fresh cascade session."""
    meta = next((p for p in list_profiles() if p["name"] == name), None)
    if meta is None:
        raise HTTPException(404, f"unknown profile {name!r}")
    cfg = get_settings(name)
    client = DemoClient(cfg, ROOT / meta["recording"] if meta["recording"] else None)
    cfg = bind_recorded_models(cfg, client.replay)  # label with the models that produced the recording
    client.settings = cfg
    state.update(profile=meta, settings=cfg, client=client, pipe=OptiRoutePipeline(cfg, client, PipelineFlags()))


activate(DEFAULT_PROFILE)

app = FastAPI(title="OptiRoute")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


class ProfileRequest(BaseModel):
    name: str


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
            "all": [_task_view(t) for t in TASKS.values()], **_profile_view()}


@app.get("/api/profiles")
def profiles():
    return {"profiles": [p for p in list_profiles() if p["measured"]], "active": state["profile"]["name"]}


@app.post("/api/profile")
def set_profile(req: ProfileRequest):
    activate(req.name)
    return {"ok": True, "tau": state["pipe"].adapt.tau, **_profile_view()}


@app.post("/api/reset")
def reset():
    state["pipe"] = OptiRoutePipeline(state["settings"], state["client"], PipelineFlags())
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
    client, settings = state["client"], state["settings"]
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


def _profile_view() -> dict:
    s, p = state["settings"], state["profile"]
    local = lambda cfg: cfg.base_url.startswith(("http://localhost", "http://127.0.0.1"))
    return {"profile": {"name": p["name"], "title": p["title"], "description": p["description"]},
            "models": {"tier1": [s.tier1.model, s.tier1.params_b, "local" if local(s.tier1) else "cloud"],
                       "tier2": [s.tier2.model, s.tier2.params_b, "local" if local(s.tier2) else "cloud"]},
            "recorded": state["client"].replay is not None}


def _task_view(t: Task) -> dict:
    return {"task_id": t.task_id, "language": t.language, "category": t.category, "query": t.query,
            "context": t.context, "schema": t.schema}
