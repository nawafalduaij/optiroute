# OptiRoute: Deliver AI with Less Compute

[![tests](https://github.com/nawafalduaij/optiroute/actions/workflows/tests.yml/badge.svg)](https://github.com/nawafalduaij/optiroute/actions/workflows/tests.yml)

A bilingual (Arabic/English) **3-tier compute cascade** built for the HUMAIN Hackathon, Challenge 2. Cheap, deterministic CPU guards decide whether a request is answered from a **Tier-0 cache** (0 tokens), a **Tier-1 small model** on a pruned context, or a **Tier-2 large model**. It is hardware-agnostic: each tier is any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio, Groq, OpenAI…), and ready-made **deployment profiles** cover everything from a laptop to cloud-only.

```
query + context ─► Pruner ─► Cache Guard ─► Router ─► Tier-1 small ─► Verifier ─► answer
                             (Tier-0, 0 tok)   │ C ≥ τ               │ fail
                                               └───────► Tier-2 large ◄┘  (refusal → full context)
```

**Why it matters:** most enterprise requests are simple lookups, yet the default setup sends every one to the largest model with the whole document attached. OptiRoute cuts that compute by 79% and keeps the data on-premise, and it never trades correctness for cost silently. The entity guard blocks cached answers for a different invoice ID, and the verifier escalates answers it can't trust.

**Submission materials:** [presentation (PDF)](deliverables/OptiRoute_Presentation.pdf) · [presentation (PPTX)](deliverables/OptiRoute_Presentation.pptx) · [UI screenshots](deliverables/screenshots/)

**Tech stack:** Python 3.12+ · FastAPI + Uvicorn (demo server) · vanilla JS + Chart.js (UI) · tiktoken (token counting) · httpx (OpenAI-compatible client for Ollama, vLLM, Groq…) · pytest + GitHub Actions (tests and reproducibility check) · python-pptx (deck generated from measured results). Models measured: qwen2.5 0.5B/3B/7B/14B, qwen3.5 9B, llama3.2 3B, ALLaM-2 7B, gpt-oss-120b.

## Headline results: `single-gpu` profile (qwen2.5 3B → 14B, 96 held-out cases)

Measured on one consumer GPU (AMD RX 7900 XTX). Compute is counted in hardware-independent units (tokens × active parameters), so these savings apply on any hardware; latency is specific to this machine.

Source: [`results/summary_tables.md`](results/summary_tables.md), a replay of [`data/recorded_runs.jsonl`](data/recorded_runs.jsonl) (480 calls, temperature 0).

| Config | Accuracy | Tokens | Compute (kTok·B) | CPS (kTok·B / success) | P50 ms |
|---|---|---|---|---|---|
| Always Tier-2 (14B), raw context | 91.7% | 36,785 | 515.0 | 5.85 | 462 |
| Always Tier-1 (3B), raw context | 85.4% | 35,956 | 107.9 | 1.32 | 138 |
| **OptiRoute Full** | **84.4%** | **18,177** | **106.9 (−79.2%)** | **1.32 (4.4× cheaper)** | **145** |
| OptiRoute, accuracy-first (pruner off) | 89.6% | 34,960 | 172.7 (−66.5%) | 2.01 (2.9× cheaper) | 147 |

- The two OptiRoute rows are two measured operating points: maximum savings, or near-large-model accuracy at two thirds less compute. The gap between them is the pruner on low-overlap questions (50% vs 100%), see Limitations.
- Entity guard: **0/6** false cache hits on near-duplicate traps (3/6 without it).
- Runtime verifier: **78% precision, 33% recall** on Tier-1 outputs (graded offline; it never sees ground truth). Removing it costs 3 correct answers.
- Pruner: −72% context tokens at 0× noise and −94% at 4× noise, with 89–90% evidence recall.
- CPU overhead for all four guards: **under 2 ms** per request.

## Deployment profiles (same code, all recorded real runs)

Pick the profile that matches your hardware and data rules: `OPTIROUTE_PROFILE=<name>` in `.env`, or the selector in the demo UI. Profiles live in [`profiles/`](profiles/); they hold no secrets (cloud keys are read by name from your git-ignored `.env`).

| Profile | Tier-1 → Tier-2 | Who it's for | Tier-1 alone | Always Tier-2 | OptiRoute | Tasks sent to Tier-2 | Tier-2 tokens |
|---|---|---|---|---|---|---|---|
| `single-gpu` | qwen2.5 3B → qwen2.5 14B (both local) | data must stay on-premise | 85.4% | 91.7% | 84.4% | 18/96 | −87% |
| `hybrid` | qwen3.5 9B local → gpt-oss-120b cloud | best accuracy, most data stays local | 94.8% | 99.0% | **95.8%** | 19/96 | −85% |
| `edge` | qwen2.5 0.5B local (CPU-friendly) → gpt-oss-120b cloud | no GPU | 57.3% | 99.0% | 80.2% | 25/96 | −83% |
| `cloud` | ALLaM-2 7B → gpt-oss-120b (both Groq) | no local hardware | 90.6% | 99.0% | 90.6% | 20/96 | −85% |

- **"Tier-2 tokens"** is the cut in large-model traffic versus sending every request to Tier-2. For a cloud Tier-2, that is the API bill and the data leaving your premises.
- **hybrid:** within 3.2 points of always-cloud while 77 of 96 requests never leave the machine.
- **cloud:** the cascade matches ALLaM alone at 85% less large-model traffic, but does not improve on it: refusal rules written from Qwen's phrasing miss some of ALLaM's, so those refusals aren't escalated. Per-model calibration on the dev set is on the roadmap.
- **edge:** the cascade adds 22.9 points over the 0.5B alone.
- More recorded pairs (llama3.2 3B, qwen2.5 0.5B/7B with a local 14B, qwen2.5 3B with a cloud Tier-2) are in [`results/summary_tables.md`](results/summary_tables.md) (Experiment 5).
- Optional cost in USD: set `TIER*_PRICE_IN_PER_MTOK` / `TIER*_PRICE_OUT_PER_MTOK` from your provider's price list. No prices are built in.
- **Adaptation on real runs:** rule checks alone can't see confident errors (during the hard burst the 0.5B passed verification 81% of the time, but only 38% of those answers were right), so τ stays flat. With a **shadow audit**, every 4th verified answer is re-asked to Tier-2 and disagreement lowers the health estimate. τ then goes 0.55 → 0.39 in the hard burst and recovers to 0.55 afterwards, at +34% compute on that stream, without improving accuracy there.

**Limitations** (deck slide 9): the pruner is lexical and drops the right paragraph on zero-overlap paraphrases (low-overlap accuracy 50% with it, 100% without); recovery depends on the small model admitting it doesn't know, and the refusal rules are phrase-based. The verifier misses grounded-but-wrong arithmetic (multi-hop is weak even for the local 14B, which sometimes answers Arabic arithmetic in Chinese). Adaptation needs a costly audit signal. n = 96 synthetic cases (±1 task ≈ 1 point).

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python evaluate.py --replay data/recorded_runs.jsonl   # all tables + profile comparison in seconds, no GPU
.venv/bin/uvicorn app:app --port 8000                            # demo UI → http://localhost:8000
OPTIROUTE_MODE=sim .venv/bin/pytest -v                           # unit tests (synthetic model)
.venv/bin/python evaluate.py --sim                               # offline smoke run → results/sim/ (not headline)
```

Re-record with your own models: copy `.env.example` to `.env`, choose a profile (or set `TIER1_*`/`TIER2_*` directly), add any cloud key, then:
```bash
ollama serve &                                    # for local tiers; ollama pull <models of your profile>
OPTIROUTE_PROFILE=hybrid .venv/bin/python evaluate.py --record --fresh --output data/my_run.jsonl
.venv/bin/python evaluate.py --replay data/recorded_runs.jsonl --compare data/my_run.jsonl
```
Two tiers that both run locally need enough memory for both models at once (`OLLAMA_MAX_LOADED_MODELS=2`); a cloud tier needs none.

## Layout

| Path | Purpose |
|---|---|
| `src/normalize.py` | Arabic normalization (Alef/ة/ى, tashkeel, clitics), Eastern numerals, number/ID extraction |
| `src/pruner.py` | boilerplate removal + hybrid char-n-gram/word-TF-IDF/entity chunk scoring |
| `src/cache.py` | Tier-0 cache: digit-masked cosine ≥ 0.88 + exact entity guard + context hash |
| `src/router.py` | 4-signal complexity score C vs τ |
| `src/verifier.py` | refusal, JSON schema, numeric grounding (+ − × ÷ %), language checks; no ground-truth access |
| `src/adaptation.py` | bidirectional τ controller with hysteresis |
| `src/pipeline.py` | cascade orchestrator with per-component ablation flags |
| `src/llm_client.py` | OpenAI-compatible client + recorder / replayer / simulator |
| `src/ground_truth_scorer.py` | offline benchmark judge (used only by `evaluate.py`) |
| `scripts/generate_dataset.py` | 96 held-out + 10 dev bilingual cases on 11 synthetic enterprise documents |
| `evaluate.py` | cumulative + LOO ablations, verifier P/R, noise scaling, wave workload, near-duplicate audit |
| `app.py`, `static/index.html` | interactive demo (presets replay recordings; custom input runs live) |
| `scripts/generate_pptx.py` | builds `deliverables/OptiRoute_Presentation.pptx` (10 slides) from `results/metrics.json` |
| `scripts/capture_screenshots.py` | headless screenshots of the running UI (used by the deck) |
| `profiles/*.env` | deployment profiles: `single-gpu`, `hybrid`, `edge`, `cloud` |
| `data/recorded_runs*.jsonl` | main recording (single-gpu) + profile and model-swap recordings (`--assemble` combines tiers without new calls) |

**Data:** all documents are synthetic and the companies are fictional, as permitted by the organizers. **Iteration disclosure:** after inspecting held-out failures, two verifier rules were generalized: (1) refusal detection now uses a negation + information-verb pattern instead of a phrase list; (2) lookups must quote numbers verbatim, and derived arithmetic is allowed only for computational questions. Both were re-checked for false positives on all 540 recorded Tier-1 outputs. The held-out set was then expanded from 36 to 96 cases; the 60 new cases (4 new documents) were written and committed before any model was run on them, and no system component was changed afterwards. After the first run, two new reference answers were widened because they rejected correct answers (listed in `scripts/generate_dataset.py`). Model outputs were never edited.

Re-record a Tier-1 swap (Tier-2 reused):
```bash
TIER1_MODEL=llama3.2:3b TIER1_PARAMS_B=3.2 .venv/bin/python evaluate.py --record --tiers 1 \
    --reuse data/recorded_runs.jsonl --output data/recorded_runs_t1-llama3.2-3b.jsonl
```
