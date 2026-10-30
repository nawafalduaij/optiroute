# OptiRoute: Deliver AI with Less Compute

A bilingual (Arabic/English) **3-tier compute cascade** built for the HUMAIN Hackathon, Challenge 2. Cheap, deterministic CPU guards decide whether a request is answered from a **Tier-0 cache** (0 tokens), a **Tier-1 small model** on a pruned context, or a **Tier-2 large model**. It is hardware-agnostic: each tier is any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio, Groq, OpenAI…), and ready-made **deployment profiles** cover everything from a laptop to cloud-only.

```
query + context ─► Pruner ─► Cache Guard ─► Router ─► Tier-1 small ─► Verifier ─► answer
                             (Tier-0, 0 tok)   │ C ≥ τ               │ fail
                                               └───────► Tier-2 large ◄┘  (refusal → full context)
```

## Headline results: `single-gpu` profile (qwen2.5 3B → 14B, 36 held-out cases)

Measured on one consumer GPU (AMD RX 7900 XTX). Compute is counted in hardware-independent units (tokens × active parameters), so these savings apply on any hardware; latency is specific to this machine.

Source: [`results/summary_tables.md`](results/summary_tables.md), a replay of [`data/recorded_runs.jsonl`](data/recorded_runs.jsonl) (180 calls, temperature 0).

| Config | Accuracy | Tokens | Compute (kTok·B) | CPS (kTok·B / success) | P50 ms |
|---|---|---|---|---|---|
| Always Tier-2 (14B), raw context | 88.9% | 13,712 | 192.0 | 6.00 | 472 |
| Always Tier-1 (3B), raw context | 86.1% | 13,363 | 40.1 | 1.29 | 129 |
| **OptiRoute Full** | **86.1%** | **6,847** | **39.2 (−79.6%)** | **1.26 (4.7× cheaper)** | **152** |

- Entity guard: **0/2** false cache hits on near-duplicate traps (2/2 without it).
- Runtime verifier: **100% precision, 57% recall** on Tier-1 outputs (graded offline; it never sees ground truth).
- Pruner: −70% context tokens at 0× noise and −94% at 4× noise, with 89–92% evidence recall.
- CPU overhead for all four guards: **under 2 ms** per request.

## Deployment profiles (same code, all recorded real runs)

Pick the profile that matches your hardware and data rules: `OPTIROUTE_PROFILE=<name>` in `.env`, or the selector in the demo UI. Profiles live in [`profiles/`](profiles/); they hold no secrets (cloud keys are read by name from your git-ignored `.env`).

| Profile | Tier-1 → Tier-2 | Who it's for | Tier-1 alone | Always Tier-2 | OptiRoute | Tasks sent to Tier-2 | Tier-2 tokens |
|---|---|---|---|---|---|---|---|
| `single-gpu` | qwen2.5 3B → qwen2.5 14B (both local) | data must stay on-premise | 86.1% | 88.9% | 86.1% | 6/36 | −88% |
| `hybrid` | qwen3.5 9B local → gpt-oss-120b cloud | best accuracy, most data stays local | 91.7% | 97.2% | **100%** | 7/36 | −84% |
| `edge` | qwen2.5 0.5B local (CPU-friendly) → gpt-oss-120b cloud | no GPU | 55.6% | 97.2% | 83.3% | 8/36 | −83% |
| `cloud` | ALLaM-2 7B → gpt-oss-120b (both Groq) | no local hardware | 94.4% | 97.2% | 86.1% | 6/36 | −87% |

- **"Tier-2 tokens"** is the cut in large-model traffic versus sending every request to Tier-2. For a cloud Tier-2, that is the API bill and the data leaving your premises.
- **hybrid:** the 100% matches the cloud model rather than beating it. Always-cloud's single miss is a correct paraphrase that the strict scorer rejects.
- **cloud:** here the cascade scores *below* ALLaM alone. The refusal rules were written from Qwen's phrasing and miss ALLaM's ("does not give information…"), so starved-context refusals aren't escalated. We did not tune them on these test answers; per-model calibration on the dev set is on the roadmap.
- More recorded pairs (llama3.2 3B, qwen2.5 0.5B/7B with a local 14B) are in [`results/summary_tables.md`](results/summary_tables.md) (Experiment 5). With a weak 0.5B, the cascade adds 19.4 points over that model alone.
- Optional cost in USD: set `TIER*_PRICE_IN_PER_MTOK` / `TIER*_PRICE_OUT_PER_MTOK` from your provider's price list. No prices are built in.
- **Adaptation on real runs:** rule checks alone can't see confident errors (during the burst the 0.5B passed verification 96% of the time at 50% true accuracy), so τ stays flat. With a **shadow audit**, every 4th verified answer is re-asked to Tier-2 and disagreement lowers the health estimate. τ then goes 0.55 → 0.47 in the hard burst and recovers to 0.55 afterwards, at +42% compute on that stream.

**Limitations** (deck slide 9): the verifier misses grounded-but-wrong arithmetic (multi-hop is weak even for the local 14B); a lexical pruner cannot see zero-overlap paraphrases, and the refusal rules are model-specific; adaptation needs a costly audit signal and rarely changes routing because complexity scores cluster; n = 36.

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
| `scripts/generate_dataset.py` | 36 held-out + 10 dev bilingual cases (synthetic enterprise documents) |
| `evaluate.py` | cumulative + LOO ablations, verifier P/R, noise scaling, wave workload, near-duplicate audit |
| `app.py`, `static/index.html` | interactive demo (presets replay recordings; custom input runs live) |
| `scripts/generate_pptx.py` | builds `deliverables/OptiRoute_Presentation.pptx` (10 slides) from `results/metrics.json` |
| `scripts/capture_screenshots.py` | headless screenshots of the running UI (used by the deck) |
| `profiles/*.env` | deployment profiles: `single-gpu`, `hybrid`, `edge`, `cloud` |
| `data/recorded_runs*.jsonl` | main recording (single-gpu) + profile and model-swap recordings (`--assemble` combines tiers without new calls) |

**Data:** all documents are synthetic and the companies are fictional, as permitted by the organizers. **Iteration disclosure:** after inspecting held-out failures, two verifier rules were generalized: (1) refusal detection now uses a negation + information-verb pattern instead of a phrase list; (2) lookups must quote numbers verbatim, and derived arithmetic is allowed only for computational questions. Both were re-checked for false positives on all 540 recorded Tier-1 outputs. Dataset answers and model outputs were never edited.

Re-record a Tier-1 swap (Tier-2 reused):
```bash
TIER1_MODEL=llama3.2:3b TIER1_PARAMS_B=3.2 .venv/bin/python evaluate.py --record --tiers 1 \
    --reuse data/recorded_runs.jsonl --output data/recorded_runs_t1-llama3.2-3b.jsonl
```
