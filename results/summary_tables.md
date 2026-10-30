# OptiRoute — Measured Results (REPLAY)

- Source: replay of `data/recorded_runs.jsonl`
- Tier-1: `qwen2.5:3b` (3B) · Tier-2: `qwen2.5:14b` (14B) · tokenizer: tiktoken `o200k_base`
- Held-out cases: 36 · Compute = Σ tokens/1000 × active params (B); Tier-0 cache = 0
- Latency = recorded LLM latency + measured CPU overhead (`time.perf_counter`)
- Iteration disclosure (after inspecting held-out failures): (1) the verifier's refusal detector was generalized from a phrase list to a negation+information-verb pattern; (2) numeric grounding now requires verbatim numbers for lookup questions and allows derived (+ − × ÷ %) numbers only when the query asks for computation. Both were re-checked for false positives on all 540 recorded Tier-1 outputs across 3 models. Dataset answers and model outputs were never edited.

## Table 1 — Cumulative Progression

| Config | Accuracy | Tokens | Δ Tokens vs T2 | Compute (kTok·B) | Δ Compute | CPS (kTok·B / success) | P50 ms | P95 ms | Final tier 0/1/2 | Escalations | False cache hits / traps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1_Always_Tier2_Raw | 88.9% (32/36) | 13,712 | +0.0% | 191.97 | +0.0% | 5.999 | 472 | 1028 | 0/0/36 | 0 | 0/2 |
| 2_Always_Tier1_Raw | 86.1% (31/36) | 13,363 | +2.5% | 40.09 | +79.1% | 1.293 | 129 | 199 | 0/36/0 | 0 | 0/2 |
| 3_Router_Only | 86.1% (31/36) | 13,580 | +1.0% | 60.63 | +68.4% | 1.956 | 146 | 819 | 0/32/4 | 0 | 0/2 |
| 4_Pruner_Plus_Router | 80.6% (29/36) | 6,519 | +52.5% | 30.08 | +84.3% | 1.037 | 148 | 648 | 0/32/4 | 0 | 0/2 |
| 5_OptiRoute_Full | 86.1% (31/36) | 6,847 | +50.1% | 39.16 | +79.6% | 1.263 | 152 | 648 | 2/28/6 | 2 | 0/2 |

## Table 2 — Leave-One-Out Ablation

| Config | Accuracy | Tokens | Δ Tokens vs T2 | Compute (kTok·B) | Δ Compute | CPS (kTok·B / success) | P50 ms | P95 ms | Final tier 0/1/2 | Escalations | False cache hits / traps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Full_OptiRoute | 86.1% (31/36) | 6,847 | +50.1% | 39.16 | +79.6% | 1.263 | 152 | 648 | 2/28/6 | 2 | 0/2 |
| LOO_No_Cache | 86.1% (31/36) | 7,255 | +47.1% | 40.39 | +79.0% | 1.303 | 152 | 648 | 0/30/6 | 2 | 0/2 |
| LOO_No_Cache_Entity_Guard | 83.3% (30/36) | 6,401 | +53.3% | 35.10 | +81.7% | 1.170 | 148 | 586 | 4/27/5 | 2 | 2/2 |
| LOO_No_Pruner | 86.1% (31/36) | 13,235 | +3.5% | 64.51 | +66.4% | 2.081 | 142 | 1218 | 2/29/5 | 1 | 0/2 |
| LOO_No_Verifier | 80.6% (29/36) | 6,111 | +55.4% | 28.86 | +85.0% | 0.995 | 142 | 648 | 2/30/4 | 0 | 0/2 |
| LOO_No_Adaptation | 86.1% (31/36) | 6,847 | +50.1% | 39.16 | +79.6% | 1.263 | 152 | 648 | 2/28/6 | 2 | 0/2 |

## Accuracy by Category

| Config | arabic_orthographic_variants | cache_near_duplicates | low_lexical_overlap | multi_hop | standard |
|---|---|---|---|---|---|
| 1_Always_Tier2_Raw | 75% | 75% | 100% | 50% | 100% |
| 2_Always_Tier1_Raw | 100% | 75% | 100% | 25% | 95% |
| 3_Router_Only | 100% | 75% | 100% | 25% | 95% |
| 4_Pruner_Plus_Router | 100% | 75% | 25% | 25% | 100% |
| 5_OptiRoute_Full | 100% | 75% | 75% | 25% | 100% |
| LOO_No_Cache | 100% | 75% | 75% | 25% | 100% |
| LOO_No_Cache_Entity_Guard | 100% | 50% | 75% | 25% | 100% |
| LOO_No_Pruner | 100% | 75% | 100% | 25% | 95% |
| LOO_No_Verifier | 100% | 75% | 25% | 25% | 100% |
| LOO_No_Adaptation | 100% | 75% | 75% | 25% | 100% |

## Runtime Verifier Precision / Recall (graded offline by ground-truth scorer)

| Tier-1 context | Tier-1 error rate | Escalations (TP+FP) | TP | FP | FN | TN | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| pruned | 19.4% | 4 | 4 | 0 | 3 | 29 | 100.0% | 57.1% |
| raw | 13.9% | 2 | 1 | 1 | 4 | 30 | 50.0% | 20.0% |

## Measured Component Overhead (mean ms per task, Full cascade)

| Pruner | Cache | Router | Verifier | Total CPU | Mean LLM latency |
|---|---|---|---|---|---|
| 1.34 | 0.18 | 0.02 | 0.11 | 1.65 | 152 (P50 end-to-end) |

## Experiment 1 — Bilingual Context Noise Scaling (pruner only, CPU)

| Distractors | Evidence recall | Tokens before | Tokens after | Reduction | Prune ms |
|---|---|---|---|---|---|
| 0x | 91.7% | 8,999 | 2,693 | 70.1% | 1.17 |
| 1x | 91.7% | 17,836 | 2,823 | 84.2% | 2.34 |
| 2x | 91.7% | 26,522 | 2,718 | 89.8% | 2.47 |
| 4x | 88.9% | 43,821 | 2,642 | 94.0% | 3.07 |

## Experiment 2 — Wave Workload (easy → hard burst → easy), cache off

Phases (step index): {'easy_1': [0, 10], 'hard_burst': [10, 34], 'easy_2': [34, 54]}

| Controller | Accuracy | Compute | CPS | Escalations | Final tier 0/1/2 | τ min | τ final |
|---|---|---|---|---|---|---|---|
| adaptive | 85.2% | 65.09 | 1.415 | 4 | 0/44/10 | 0.55 | 0.55 |
| fixed_tau | 85.2% | 65.09 | 1.415 | 4 | 0/44/10 | 0.55 | 0.55 |
| adaptive_audit | 85.2% | 92.67 | 2.014 | 4 | 0/43/11 | 0.47 | 0.55 |
| fixed_tau_audit | 85.2% | 92.67 | 2.014 | 4 | 0/43/11 | 0.55 | 0.55 |

τ(t) trace (adaptive): 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55

Trigger check: min windowed Tier-1 verifier pass rate = 80% (tighten if < 70%); max windowed P95 = 1221 ms (tighten if > SLO 8000 ms).

## Experiment 3 — Near-Duplicate Adversarial Audit

| Config | Cache hits | Blocked traps | False cache hits / traps | Cache-category accuracy |
|---|---|---|---|---|
| With entity guard | 2 | 2 | 0/2 | 75% |
| Without entity guard | 4 | 0 | 2/2 | 50% |

## Experiment 4 — Real Tier-1 Degradation Stress (adaptation under genuine failures)

Same wave workload as Experiment 2, but Tier-1 is the much weaker real model `qwen2.5:0.5b` (recorded: `data/recorded_runs_t1-qwen2.5-0.5b.jsonl`).

| Controller | Accuracy | Compute | CPS | Tier-1 attempts | Escalations | Final tier 0/1/2 | τ min | τ final |
|---|---|---|---|---|---|---|---|---|
| adaptive | 74.1% | 41.86 | 1.047 | 48 | 5 | 0/43/11 | 0.55 | 0.55 |
| fixed_tau | 74.1% | 41.86 | 1.047 | 48 | 5 | 0/43/11 | 0.55 | 0.55 |
| adaptive_audit | 75.9% | 66.23 | 1.615 | 48 | 5 | 0/42/12 | 0.47 | 0.55 |
| fixed_tau_audit | 75.9% | 66.23 | 1.615 | 48 | 5 | 0/42/12 | 0.55 | 0.55 |

τ(t) trace (adaptive): 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55

Controller actions (adaptive, step:action): none

τ(t) trace (adaptive_audit): 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.47 0.47 0.47 0.47 0.47 0.47 0.47 0.47 0.47 0.52 0.55

Controller actions (adaptive_audit, step:action): 43:tighten 52:recover 53:recover

`_audit` = every 4th verified Tier-1 answer is also sent to Tier-2; disagreement counts as a Tier-1 failure for the adaptation window (and the Tier-2 answer is used). Audit calls are included in compute.

## Experiment 5 — Cross-Model Generalization (same cascade, different recorded models)

| Recording | Tier-1 | Tier-2 | Always-T2 acc | Always-T1 acc | Full acc | Full compute Δ vs T2 | CPS T2 → Full | Verifier P / R |
|---|---|---|---|---|---|---|---|---|
| `recorded_runs.jsonl` | qwen2.5:3b (3B) | qwen2.5:14b (14B) | 88.9% | 86.1% | **86.1%** | -79.6% | 6.00 → 1.26 | 100% / 57% |
| `recorded_runs_t1-llama3.2-3b.jsonl` | llama3.2:3b (3.2B) | qwen2.5:14b (14B) | 88.9% | 88.9% | **86.1%** | -79.3% | 6.00 → 1.28 | 75% / 50% |
| `recorded_runs_t1-qwen2.5-0.5b.jsonl` | qwen2.5:0.5b (0.5B) | qwen2.5:14b (14B) | 88.9% | 55.6% | **75.0%** | -84.1% | 6.00 → 1.13 | 100% / 47% |
