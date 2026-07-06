# Training Analysis Summary

Reference doc for training-curve decisions, Stage B gates, and final model selection.

## Key comparisons (matched steps)

| Comparison | Step | A (dB) | B (dB) | Δ (dB) |
|---|---:|---:|---:|---:|
| Plus vs Base @ 2k | 2000 | 27.164 | 27.099 | +0.065 |
| Plus vs Base @ 20k | 20000 | 27.913 | 27.781 | +0.132 |
| VGG λ=0.01 vs λ=0 @ 2k | 2000 | 27.057 | 27.060 | -0.003 |
| Pixel KD λ=0.2 vs λ=0 @ 20k | 20000 | 27.741 | 27.749 | -0.008 |

## Benchmark results (fair-budget 20k, test sets)

Avg PSNR = unweighted mean over Set5 / Set14 / BSD100 / Urban100 (same as `final_report.md`).

| Model | Avg PSNR ↑ | Avg LPIPS ↓ | Params | FLOPs (G) | FP32 (ms) | FP16 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| FSRCNN | 26.820 | 0.1738 | 24,683 | 7.408 | 1.61 | 1.82 |
| MobileSRNet-Base | 27.159 | 0.1642 | 30,208 | 0.976 | 1.56 | 1.35 |
| **MobileSRNet-Plus** | **27.331** | **0.1565** | 66,864 | 2.163 | 2.38 | 0.80 |
| SwinIR (teacher) | 29.126 | 0.1080 | 11,900,199 | 417.797 | 651.56 | — |

Δ(Plus − Base): **+0.172 dB** avg PSNR, **−0.008** avg LPIPS.  
Δ(Plus − FSRCNN): **+0.511 dB** avg PSNR at **~3.4×** lower FLOPs.

Per-dataset PSNR (Plus vs Base):

| Dataset | Base | Plus | Δ |
|---|---:|---:|---:|
| Set5 | 30.24 | 30.49 | +0.25 |
| Set14 | 27.37 | 27.53 | +0.17 |
| BSD100 | 26.79 | 26.88 | +0.10 |
| Urban100 | 24.24 | 24.42 | +0.17 |

**Verdict:** Plus promoted to headline model for RQ1. Benchmarks confirm Stage B val-PSNR gain.

## Insights

- Plus beats Base by +0.137 dB at 20k val PSNR (27.936 vs 27.799) and **+0.172 dB** on test benchmarks.
- Stage B capacity probe (+0.065 dB @ 2k val) correctly predicted 20k benchmark benefit — gate workflow validated.
- Plus gain **accelerates** late (0.005 dB/1k steps vs Base 0.003) — not saturated at 20k.
- Base, Plus, and FSRCNN all hit **best checkpoint at final epoch** — 20k is a floor, not a ceiling.
- Pixel KD curves track together (λ=0 slope 0.0048, λ=0.2 slope 0.0049 dB/1k steps) — null is structural.
- VGG Stage B null: Δ=−0.003 dB @ 2k — non-redundant gradient did not translate to val gain.
- Charbonnier KD runs reach ~27.75 val PSNR; MSE Base/Plus reach ~27.80–27.94 — loss choice matters slightly on val, not on headline benchmarks.

## Recommendations

- **Done:** benchmark eval, profile, qualitative panels, FLOPs–latency plot, latency re-audit.
- **Next:** submission — `final_report.md` is complete.
- **Optional:** extend Plus to 30k if more GPU time — curves still rising at epoch 607.
- **Do not:** more SwinIR KD (pixel + VGG both null across all budgets and gates).

## Figures

- `training_analysis/01_fair_budget_20k_architectures.png`
- `training_analysis/02_capacity_base_vs_plus_early.png`
- `training_analysis/02b_capacity_base_vs_plus_full.png`
- `training_analysis/03_kd_lambda_sweep_curves.png`
- `training_analysis/03b_kd_lambda_sweep_final_psnr.png`
- `training_analysis/04_kd_loss_decomposition.png`
- `training_analysis/05_kd_isolation_20k.png`
- `training_analysis/06_vgg3_stage_b_2k.png`
- `training_analysis/07_train_loss_vs_val_psnr.png`
- `training_analysis/08_learning_rate_by_phase.png`

## Observations (reference)

**1. Capacity scaling is the real win — KD is not**
- Plus leads Base at every matched step: +0.065 dB @ 2k → +0.132 dB @ 20k val → **+0.172 dB** on benchmarks.
- Pixel KD and VGG KD flat across λ, budget, and gradient gates — do not revisit.

**2. Stage B gate workflow saved ~13h twice**
- Pixel KD: Gate 2 fail (cosine 0.87) → skipped full run (confirmed null by prior λ-sweep).
- VGG relu3: Gates pass → Stage B null @ 2k → skipped 20k VGG.
- Plus capacity: Stage B pass (+0.065 dB @ 2k) → 20k Plus confirmed (+0.172 dB benchmarks).

**3. Nothing fully converged at 20k**
- Best checkpoint at final epoch for Base, Plus, FSRCNN — optional 30k extension is real but diminishing.

**4. DIV2K-val ranking ≠ benchmark ranking**
- FSRCNN val PSNR looks competitive on DIV2K-valid but loses on Urban100/Set5 benchmarks.
- Use test-set eval for model selection, not val alone (see `01_fair_budget_20k_architectures.png`).

**5. Plus efficiency trade-off is favorable**
- 2.2× FLOPs vs Base, still **3.4× below FSRCNN**; FP16 latency 0.80 ms vs FSRCNN 1.82 ms @ LR 180×180.
- Params 67k — still mobile-proxy scale.

## Artifacts

| Artifact | Path |
|---|---|
| Plus benchmarks | `results/exp_runs/mobile_srnet_plus_20k/benchmark_metrics.json` |
| Plus profile | `results/exp_runs/mobile_srnet_plus_20k/profile.json` |
| Fair-budget summary (incl. Plus) | `results/exp_runs/fair_budget_runs.json` |
| KD method gates | `results/exp_runs/kd_method_gates.json` |
| Training curves JSON | `results/training_analysis/training_analysis_summary.json` |
