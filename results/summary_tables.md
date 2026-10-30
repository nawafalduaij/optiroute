# OptiRoute — Measured Results (REPLAY)

- Source: replay of `data/recorded_runs.jsonl`
- Tier-1: `qwen2.5:3b` (3B) · Tier-2: `qwen2.5:14b` (14B) · tokenizer: tiktoken `o200k_base`
- Held-out cases: 96 · Compute = Σ tokens/1000 × active params (B); Tier-0 cache = 0
- Latency = recorded LLM latency + measured CPU overhead (`time.perf_counter`)
- Iteration disclosure (after inspecting held-out failures): (1) the verifier's refusal detector was generalized from a phrase list to a negation+information-verb pattern; (2) numeric grounding now requires verbatim numbers for lookup questions and allows derived (+ − × ÷ %) numbers only when the query asks for computation. Both were re-checked for false positives on all 540 recorded Tier-1 outputs across 3 models. Dataset answers and model outputs were never edited.

## Table 1 — Cumulative Progression

| Config | Accuracy | Tokens | Δ Tokens vs T2 | Compute (kTok·B) | Δ Compute | CPS (kTok·B / success) | P50 ms | P95 ms | Final tier 0/1/2 | Escalations | False cache hits / traps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1_Always_Tier2_Raw | 91.7% (88/96) | 36,785 | +0.0% | 514.99 | +0.0% | 5.852 | 462 | 1125 | 0/0/96 | 0 | 0/6 |
| 2_Always_Tier1_Raw | 85.4% (82/96) | 35,956 | +2.3% | 107.87 | +79.1% | 1.315 | 138 | 241 | 0/96/0 | 0 | 0/6 |
| 3_Router_Only | 89.6% (86/96) | 36,494 | +0.8% | 168.05 | +67.4% | 1.954 | 148 | 1159 | 0/84/12 | 0 | 0/6 |
| 4_Pruner_Plus_Router | 81.2% (78/96) | 17,452 | +52.6% | 84.04 | +83.7% | 1.077 | 141 | 896 | 0/84/12 | 0 | 0/6 |
| 5_OptiRoute_Full | 84.4% (81/96) | 18,177 | +50.6% | 106.87 | +79.2% | 1.319 | 145 | 980 | 6/72/18 | 6 | 0/6 |

## Table 2 — Leave-One-Out Ablation

| Config | Accuracy | Tokens | Δ Tokens vs T2 | Compute (kTok·B) | Δ Compute | CPS (kTok·B / success) | P50 ms | P95 ms | Final tier 0/1/2 | Escalations | False cache hits / traps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Full_OptiRoute | 84.4% (81/96) | 18,177 | +50.6% | 106.87 | +79.2% | 1.319 | 145 | 980 | 6/72/18 | 6 | 0/6 |
| LOO_No_Cache | 84.4% (81/96) | 19,330 | +47.5% | 110.33 | +78.6% | 1.362 | 149 | 979 | 0/78/18 | 6 | 0/6 |
| LOO_No_Cache_Entity_Guard | 83.3% (80/96) | 17,479 | +52.5% | 99.27 | +80.7% | 1.241 | 141 | 896 | 9/71/16 | 6 | 3/6 |
| LOO_No_Pruner | 89.6% (86/96) | 34,960 | +5.0% | 172.73 | +66.5% | 2.008 | 147 | 1218 | 6/76/14 | 2 | 0/6 |
| LOO_No_Verifier | 81.2% (78/96) | 16,299 | +55.7% | 80.58 | +84.4% | 1.033 | 139 | 896 | 6/78/12 | 0 | 0/6 |
| LOO_No_Adaptation | 84.4% (81/96) | 18,177 | +50.6% | 106.87 | +79.2% | 1.319 | 145 | 980 | 6/72/18 | 6 | 0/6 |

## Accuracy by Category

| Config | arabic_orthographic_variants | cache_near_duplicates | low_lexical_overlap | multi_hop | standard |
|---|---|---|---|---|---|
| 1_Always_Tier2_Raw | 90% | 83% | 100% | 58% | 100% |
| 2_Always_Tier1_Raw | 100% | 67% | 100% | 25% | 98% |
| 3_Router_Only | 100% | 83% | 100% | 42% | 98% |
| 4_Pruner_Plus_Router | 90% | 83% | 25% | 50% | 100% |
| 5_OptiRoute_Full | 90% | 83% | 50% | 50% | 100% |
| LOO_No_Cache | 90% | 83% | 50% | 50% | 100% |
| LOO_No_Cache_Entity_Guard | 90% | 75% | 50% | 50% | 100% |
| LOO_No_Pruner | 100% | 83% | 100% | 42% | 98% |
| LOO_No_Verifier | 90% | 83% | 25% | 50% | 100% |
| LOO_No_Adaptation | 90% | 83% | 50% | 50% | 100% |

## Runtime Verifier Precision / Recall (graded offline by ground-truth scorer)

| Tier-1 context | Tier-1 error rate | Escalations (TP+FP) | TP | FP | FN | TN | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| pruned | 21.9% | 9 | 7 | 2 | 14 | 73 | 77.8% | 33.3% |
| raw | 14.6% | 3 | 1 | 2 | 13 | 80 | 33.3% | 7.1% |

## Measured Component Overhead (mean ms per task, Full cascade)

| Pruner | Cache | Router | Verifier | Total CPU | Mean LLM latency |
|---|---|---|---|---|---|
| 1.34 | 0.32 | 0.02 | 0.11 | 1.79 | 145 (P50 end-to-end) |

## Experiment 1 — Bilingual Context Noise Scaling (pruner only, CPU)

| Distractors | Evidence recall | Tokens before | Tokens after | Reduction | Prune ms |
|---|---|---|---|---|---|
| 0x | 89.6% | 24,072 | 6,766 | 71.9% | 1.27 |
| 1x | 89.6% | 47,435 | 7,003 | 85.2% | 2.52 |
| 2x | 88.5% | 70,926 | 7,175 | 89.9% | 3.64 |
| 4x | 88.5% | 117,530 | 7,450 | 93.7% | 5.30 |

## Experiment 2 — Wave Workload (easy → hard burst → easy), cache off

Phases (step index): {'easy_1': [0, 10], 'hard_burst': [10, 78], 'easy_2': [78, 98]}

| Controller | Accuracy | Compute | CPS | Escalations | Final tier 0/1/2 | τ min | τ final |
|---|---|---|---|---|---|---|---|
| adaptive | 73.5% | 139.39 | 1.936 | 10 | 0/72/26 | 0.55 | 0.55 |
| fixed_tau | 73.5% | 139.39 | 1.936 | 10 | 0/72/26 | 0.55 | 0.55 |
| adaptive_audit | 72.4% | 186.17 | 2.622 | 9 | 0/66/32 | 0.39 | 0.55 |
| fixed_tau_audit | 73.5% | 184.04 | 2.556 | 10 | 0/68/30 | 0.55 | 0.55 |

τ(t) trace (adaptive): 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55

Trigger check: min windowed Tier-1 verifier pass rate = 75% (tighten if < 70%); max windowed P95 = 1221 ms (tighten if > SLO 8000 ms).

## Experiment 3 — Near-Duplicate Adversarial Audit

| Config | Cache hits | Blocked traps | False cache hits / traps | Cache-category accuracy |
|---|---|---|---|---|
| With entity guard | 6 | 3 | 0/6 | 83% |
| Without entity guard | 9 | 0 | 3/6 | 75% |

## Experiment 4 — Real Tier-1 Degradation Stress (adaptation under genuine failures)

Same wave workload as Experiment 2, but Tier-1 is the much weaker real model `qwen2.5:0.5b` (recorded: `data/recorded_runs_t1-qwen2.5-0.5b.jsonl`).

| Controller | Accuracy | Compute | CPS | Tier-1 attempts | Escalations | Final tier 0/1/2 | τ min | τ final |
|---|---|---|---|---|---|---|---|---|
| adaptive | 61.2% | 104.74 | 1.746 | 82 | 13 | 0/69/29 | 0.55 | 0.55 |
| fixed_tau | 61.2% | 104.74 | 1.746 | 82 | 13 | 0/69/29 | 0.55 | 0.55 |
| adaptive_audit | 63.3% | 156.77 | 2.529 | 76 | 12 | 0/61/37 | 0.31 | 0.41 |
| fixed_tau_audit | 62.2% | 148.10 | 2.428 | 82 | 13 | 0/64/34 | 0.55 | 0.55 |

τ(t) trace (adaptive): 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55

Controller actions (adaptive, step:action): none

τ(t) trace (adaptive_audit): 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.47 0.47 0.47 0.47 0.47 0.47 0.47 0.47 0.47 0.47 0.47 0.47 0.47 0.47 0.47 0.47 0.47 0.47 0.47 0.39 0.39 0.39 0.39 0.39 0.39 0.39 0.39 0.39 0.39 0.39 0.39 0.39 0.39 0.39 0.39 0.39 0.39 0.39 0.39 0.39 0.39 0.39 0.39 0.39 0.39 0.39 0.39 0.39 0.39 0.31 0.31 0.31 0.31 0.31 0.31 0.31 0.31 0.31 0.36 0.41 0.41 0.41 0.41 0.41 0.41 0.41 0.41 0.41 0.41 0.41 0.41 0.41 0.41

Controller actions (adaptive_audit, step:action): 25:tighten 44:tighten 74:tighten 83:recover 84:recover

`_audit` = every 4th verified Tier-1 answer is also sent to Tier-2; disagreement counts as a Tier-1 failure for the adaptation window (and the Tier-2 answer is used). Audit calls are included in compute.

## Experiment 5 — Cross-Model Generalization (same cascade, different recorded models)

| Profile | Recording | Tier-1 | Tier-2 | Always-T2 acc | Always-T1 acc | Full acc | Full compute Δ vs T2 | CPS T2 → Full | Verifier P / R | Tasks sent to Tier-2 | Tier-2 tokens vs always-T2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **single-gpu** | `recorded_runs.jsonl` | qwen2.5:3b (3B) | qwen2.5:14b (14B) | 91.7% | 85.4% | **84.4%** | -79.2% | 5.85 → 1.32 | 78% / 33% | 18/96 | -87.1% |
| — | `recorded_runs_t1-llama3.2-3b.jsonl` | llama3.2:3b (3.2B) | qwen2.5:14b (14B) | 91.7% | 85.4% | **84.4%** | -77.9% | 5.85 → 1.41 | 64% / 50% | 19/96 | -86.8% |
| — | `recorded_runs_t1-qwen2.5-0.5b.jsonl` | qwen2.5:0.5b (0.5B) | qwen2.5:14b (14B) | 91.7% | 57.3% | **74.0%** | -81.8% | 5.85 → 1.32 | 90% / 41% | 25/96 | -83.1% |
| — | `recorded_runs_t1-qwen2.5-7b.jsonl` | qwen2.5:7b (7B) | qwen2.5:14b (14B) | 91.7% | 88.5% | **87.5%** | -69.2% | 5.85 → 1.89 | 89% / 42% | 17/96 | -87.4% |
| **cloud** | `recorded_runs_t2-gpt-oss-120b_t1-allam-2-7b.jsonl` | allam-2-7b (7B) | openai/gpt-oss-120b (5.1B) | 99.0% | 90.6% | **90.6%** | -30.4% | 2.11 → 1.60 | 50% / 42% | 20/96 | -85.1% |
| **edge** | `recorded_runs_t2-gpt-oss-120b_t1-qwen2.5-0.5b.jsonl` | qwen2.5:0.5b (0.5B) | openai/gpt-oss-120b (5.1B) | 99.0% | 57.3% | **80.2%** | -79.2% | 2.11 → 0.54 | 90% / 41% | 25/96 | -82.5% |
| — | `recorded_runs_t2-gpt-oss-120b_t1-qwen2.5-3b.jsonl` | qwen2.5:3b (3B) | openai/gpt-oss-120b (5.1B) | 99.0% | 85.4% | **91.7%** | -66.3% | 2.11 → 0.77 | 78% / 33% | 18/96 | -86.4% |
| **hybrid** | `recorded_runs_t2-gpt-oss-120b_t1-qwen3.5-9b.jsonl` | qwen3.5:9b (9B) | openai/gpt-oss-120b (5.1B) | 99.0% | 94.8% | **95.8%** | -23.0% | 2.11 → 1.68 | 85% / 69% | 19/96 | -84.6% |
