# Project Plan: Mobile-Efficient 4× Super-Resolution

## Project State

**All planned experiments and report artifacts are complete.** Ready for submission.

| Component | Status |
|---|---|
| Bicubic / FSRCNN / FSRCNN-Small / SwinIR baselines | Done |
| MobileSRNet-Base (40ch, 6 blocks) | Done — fair-budget 20k |
| **MobileSRNet-Plus (64ch, 8 blocks)** | Done — 20k + benchmarks (+0.172 dB vs Base) |
| Capacity Stage B (Plus @ 2k) | Done — passed |
| Fair-budget 8-run retraining (10k + 20k) | Done |
| KD isolation + λ-sweep | Done — **null** |
| KD gates + VGG Stage B | Done — **null** |
| Plus profile + benchmarks | Done |
| Latency audit (180×180 + 320×180, fair-budget ckpts) | Done |
| FLOPs vs latency scatter | Done — `latency_audit/flops_vs_latency.png` |
| Qualitative panels (4 crops, Base + Plus) | Done — `results/qualitative/` |
| Training analysis figures | Done |
| Final report draft | Done — `results/final_report.md` |

---

## Locked Assumptions

| Assumption | Statement |
|---|---|
| Training compute | Expensive training is allowed; only the deployed student must be efficient. |
| SwinIR teacher | Offline at training time only. Deployment: `LR → MobileSRNet → SR`. |
| Headline model | **MobileSRNet-Plus** (quality); **MobileSRNet-Base** (min FLOPs). |
| Mobile-proxy | Params, FLOPs, latency FP32/FP16 — not phone hardware. |

---

## Research Questions (answered)

| RQ | Answer |
|---|---|
| **RQ1 Architecture** | Yes. Plus +0.51 dB vs FSRCNN @ ~3.4× lower FLOPs; Base +0.34 dB @ ~7.6× lower FLOPs. |
| **RQ2 Distillation** | No. Pixel + VGG KD null. Capacity scaling works. |
| **RQ3 Deployment** | FLOPs incomplete predictor; Plus FP16 0.88 ms vs FSRCNN 1.35 ms @ 180×180. |

---

## Task List

| Task | Status |
|---|---|
| 1. Latency re-audit (fair-budget ckpts, 180² + 320×180) | ✅ Done |
| 2. Input-size language (720p = LR 320×180) | ✅ Done |
| 3. Qualitative panels | ✅ Done |
| 4. KD evidence (one paragraph in report) | ✅ Done |
| 5. Metrics / FLOPs audit | ✅ Done |
| 6. FLOPs vs latency plot | ✅ Done |
| 7. Capacity curve (Base + Plus) | ✅ Done |
| 8. CPU / ONNX (optional) | Skipped |

---

## Key Artifacts

| Artifact | Path |
|---|---|
| Final report | `results/final_report.md` |
| Training reference | `results/training_analysis/training_analysis.md` |
| Benchmark summary | `results/exp_runs/fair_budget_runs.json` |
| Qualitative panels | `results/qualitative/*.png` |
| RQ3 scatter | `results/latency_audit/flops_vs_latency.png` |
| Latency JSON | `results/latency_audit/latency_audit.json`, `latency_audit_320x180.json` |
| Capacity curves | `results/training_analysis/02b_capacity_base_vs_plus_full.png` |
| Pareto plot | `results/exp_runs/fair_budget_curves.png` |

---

## Optional (post-submission)

- Extend Plus/Base to 30k updates (curves still rising at epoch 607).
- CPU / ONNX backend benchmark.
- MobileSRNet-Tiny ablation (skipped — not needed for narrative).

---

## Final Narrative

> MobileSRNet beats FSRCNN at lower FLOPs. Plus adds +0.17 dB over Base; KD does not help; audited latency shows FLOPs alone do not predict speed. See `results/final_report.md`.
