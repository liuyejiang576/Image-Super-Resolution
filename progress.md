## KD λ-sweep complete (Jul 6 ~12:30)

### Verdict: robust null — no further KD training

Output-level Charbonnier KD from frozen SwinIR **does not improve** MobileSRNet under a fair training budget, across λ ∈ {0.0, 0.2, 0.5, 1.0, 2.0} at 10k updates and at the clean 20k λ=0 vs 0.2 isolation. **Demote KD to secondary / negative result per Gate 3.** Do not run a 20k λ redo.

### Evidence chain (clean, unconfounded)

| Stage | Comparison | Δ avg PSNR | Notes |
|---|---|---:|---|
| Baseline-era | `mobile_srnet` vs `mobile_srnet_kd` | +0.06 dB | **Confounded** (MSE vs Charb+KD, batch 24 vs 16, amp, clamp) |
| Fair-budget 20k | `kd0_20k` vs `kd02_20k` | **−0.012 dB** | Only λ differs; same Charb, batch 16, amp |
| λ-sweep 10k | λ=0.0 … 2.0 (5 points) | **0.011 dB spread** | See table below |

**5-point λ-sweep at 10k** (`results/exp_runs/lambda_sweep_summary.json`):

| λ | avg PSNR | avg LPIPS | DIV2K val PSNR |
|--:|---------:|----------:|---------------:|
| 0.0 | 26.769 | 0.1834 | 27.482 |
| 0.2 | 26.770 | 0.1840 | 27.492 |
| 0.5 | 26.760 | 0.1847 | 27.485 |
| 1.0 | 26.759 | 0.1843 | 27.478 |
| 2.0 | 26.759 | 0.1837 | 27.491 |

### Why null despite teacher being better (`kd_diagnostic.json`)

- **D1:** teacher 31.17 dB vs student 29.45 dB on DIV2K-valid patches → **+1.7 dB** teacher advantage; KD signal exists.
- **D2:** cosine(∇loss_gt, ∇loss_kd) = **0.877** at λ=0; at λ=0.2 weighted KD grad is 0.56× GT norm and still **88% aligned** → redundant with GT loss. Higher λ does not break this (sweep flat).

### What to report

- **RQ1 (headline):** MobileSRNet vs FSRCNN under fair-budget 20k — still strong (+0.34 dB avg PSNR, better LPIPS, ~7.6× fewer FLOPs).
- **RQ2 (honest null):** Retract baseline-era per-image KD claims (+0.04 dB / 86% improved). Present clean isolation + λ-sweep + gradient diagnostics. KD is a negative result, not a selling point.
- **RQ3:** Audited latency unchanged.

### Artifacts

| Artifact | Path |
|---|---|
| Fair-budget 8-run metrics | `results/exp_runs/fair_budget_runs.json` |
| λ-sweep summary | `results/exp_runs/lambda_sweep_summary.json` |
| KD diagnostics | `results/exp_runs/kd_diagnostic.json` |
| Updated report draft | `results/final_report.md` |
| Report writing guide | `report_plan.md` (updated for null KD) |

### Experiment status

- 8/8 fair-budget runs: **done**
- λ-sweep kd05/kd10/kd20 10k: **done** (200/200 epochs, ~13h overnight Jul 5–6)
- Analysis: **done** (`scripts/analyze_lambda_sweep.py`)
- `results/final_report.md`: **updated** with fair-budget main table + RQ2 null section
- Baseline-era `results/kd_analysis/` per-image deltas: **superseded** (confounded); kept for audit trail only

### Remaining work (writing / polish, no training)

1. Finalize prose in `results/final_report.md` → submission draft.
2. Tasks 1–3, 5–6 in `report_plan.md` (latency audit, qualitative panels, FLOPs plot) if not already complete.
3. Optional: per-image Δ on fair-budget `kd0_20k` vs `kd02_20k` checkpoints (expect flat; not required).
