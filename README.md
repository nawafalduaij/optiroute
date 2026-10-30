# OptiRoute: Deliver AI with Less Compute

A bilingual (Arabic/English) **3-tier compute cascade** built for the HUMAIN Hackathon, Challenge 2. Cheap, deterministic CPU guards decide whether a request is answered from a **Tier-0 cache** (0 tokens), a **Tier-1 small model** (`qwen2.5:3b`) on a pruned context, or the **Tier-2 large model** (`qwen2.5:14b`).

```
query + context ─► Pruner ─► Cache Guard ─► Router ─► Tier-1 3B ─► Verifier ─► answer
                             (Tier-0, 0 tok)   │ C ≥ τ            │ fail
                                               └───────► Tier-2 14B ◄┘  (refusal → full context)
```

## Headline results (36 held-out cases, recorded real runs, AMD RX 7900 XTX)

Source: [`results/summary_tables.md`](results/summary_tables.md), a replay of [`data/recorded_runs.jsonl`](data/recorded_runs.jsonl) (180 calls, temperature 0).

| Config | Accuracy | Tokens | Compute (kTok·B) | CPS (kTok·B / success) | P50 ms |
|---|---|---|---|---|---|
| Always Tier-2 (14B), raw context | 88.9% | 13,712 | 192.0 | 6.00 | 472 |
| Always Tier-1 (3B), raw context | 86.1% | 13,363 | 40.1 | 1.29 | 129 |
| **OptiRoute Full** | **86.1%** | **6,847** | **39.2 (−79.6%)** | **1.26 (4.7× cheaper)** | **152** |

- Entity guard: **0/2** false cache hits on near-duplicate traps (2/2 without it).
- Runtime verifier: **100% precision, 57% recall** on Tier-1 outputs (graded offline; it never sees ground truth).
- Pruner: −70% context tokens at 0× noise and −94% at 4× noise, with 89–92% evidence recall.
- CPU overhead for all four guards: about **1.6 ms** per request.

### Robustness (also replayed from real recordings)

| Tier-1 model (Tier-2 = qwen2.5:14b) | Tier-1 alone | OptiRoute | Compute vs always-14B | Verifier P / R |
|---|---|---|---|---|
| qwen2.5:3b | 86.1% | 86.1% | −79.6% | 100% / 57% |
| llama3.2:3b | 88.9% | 86.1% | −79.3% | 75% / 50% |
| qwen2.5:0.5b | 55.6% | **75.0%** | −84.1% | 100% / 47% |

- **Same code, any OpenAI-compatible model:** only the recording changes. With a weak 0.5B, the cascade adds 19.4 points over using the 0.5B alone.
- **Adaptation on real runs:** rule checks alone can't see confident errors (during the burst the 0.5B passed verification 96% of the time at 50% true accuracy), so τ stays flat. With a **shadow audit**, every 4th verified answer is re-asked to Tier-2 and disagreement lowers the health estimate. τ then goes 0.55 → 0.47 in the hard burst and recovers to 0.55 afterwards, at +42% compute on that stream.

**Limitations** (deck slide 9): the verifier misses grounded-but-wrong arithmetic (multi-hop is weak even for the 14B); a lexical pruner cannot see zero-overlap paraphrases; adaptation needs a costly audit signal and rarely changes routing because complexity scores cluster; n = 36.

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python evaluate.py --replay data/recorded_runs.jsonl   # all tables in ~1.4 s, no GPU needed
.venv/bin/uvicorn app:app --port 8000                            # demo UI → http://localhost:8000
OPTIROUTE_MODE=sim .venv/bin/pytest -v                           # unit tests (synthetic model)
.venv/bin/python evaluate.py --sim                               # offline smoke run → results/sim/ (not headline)
```

Re-record on your own hardware. Copy `.env.example` to `.env` (local Ollama, or Groq for Tier-2), then:
```bash
OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_NUM_PARALLEL=1 OLLAMA_KEEP_ALIVE=-1 ollama serve &
ollama pull qwen2.5:3b && ollama pull qwen2.5:14b
.venv/bin/python evaluate.py --record --fresh
```

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
| `data/recorded_runs*.jsonl` | main recording (qwen2.5 3B/14B) + Tier-1 swaps (llama3.2 3B, qwen2.5 0.5B) |

**Data:** all documents are synthetic and the companies are fictional, as permitted by the organizers. **Iteration disclosure:** after inspecting held-out failures, two verifier rules were generalized: (1) refusal detection now uses a negation + information-verb pattern instead of a phrase list; (2) lookups must quote numbers verbatim, and derived arithmetic is allowed only for computational questions. Both were re-checked for false positives on all 540 recorded Tier-1 outputs. Dataset answers and model outputs were never edited.

Re-record a Tier-1 swap (Tier-2 reused):
```bash
TIER1_MODEL=llama3.2:3b TIER1_PARAMS_B=3.2 .venv/bin/python evaluate.py --record --tiers 1 \
    --reuse data/recorded_runs.jsonl --output data/recorded_runs_t1-llama3.2-3b.jsonl
```
