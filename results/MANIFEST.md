# Results manifest

Single source of truth for which **runs/artifacts** are headline, active, or inactive.

**Doc triangle (do not merge roles):**

| Doc | Role |
|---|---|
| [`../../progress/README.md`](../../progress/README.md) | **主循环**（抽象脚本）+ 写权限；状态在 `progress/checklist.md` 等 |
| [`../IMPLEMENTATION.md`](../IMPLEMENTATION.md) | How training / resume / fair-budget / parallel work |
| **This file** | Which checkpoints & JSON counts as evidence |

Checkpoints (`.pt`) and training logs stay local and gitignored. Small JSON/PNG/MD evidence files listed below may be tracked in git per `.gitignore` exceptions.

---

## Train / deploy control manifests (JSON)

These are **launcher registries**, not the human planning file. Do not invent a fourth “current plan” here.

| Manifest | Controls | Models | Notes |
|---|---|---|---|
| `exp_runs/fair_budget_manifest.json` | `run_exp_parallel.py` / `run_exp_pipeline.py` | FSRCNN, Base, KD0, KD02 | **Plus not included** |
| `exp_runs/plus_20k_manifest.json` | `plus_20k.py` → `run_plus_20k.py` | Plus 20k only | |
| `exp_runs/arch_30k_manifest.json` | `arch_30k.py` → `run_arch_30k.py` | FSRCNN/Base/Plus 30k continuation | Bootstraps from `*_20k` |
| `exp_runs/ecbsr_20k_manifest.json` | `ecbsr_20k.py` → `run_ecbsr_20k.py` | ECBSR (M10C16) | B3 done；对外显示 **ECBSR** |
| `exp_runs/sepres_v2_20k_manifest.json` | `sepres_v2_20k.py` → `run_sepres_v2_20k.py` | SepResV2 a/b/c | **default resume = v2_b only** |
| `exp_runs/dual_plain_2k_manifest.json` | `dual_plain_2k.py` → `run_dual_plain_2k_probes.py` | DualStream + Plain C20N5 | B4 round-2 2k；default = both |
| `exp_runs/b5_train_20k_manifest.json` | `b5_train_20k.py` → `run_b5_train_20k.py` | B5a P0 + seeds + KD Stage-B 2k | MSE 3-wide; KD after MSE |
| `exp_runs/b5a_20k_manifest.json` | (legacy) | P0 plain3x3 | superseded by unified |
| `exp_runs/b5_confirm_20k_manifest.json` | (legacy) | PECSR/ECBSR × seeds | superseded by unified |
| `exp_runs/pecsr_kd_20k_manifest.json` | (legacy pointer) | PECSR Stage-B 2k KD ids | prefer unified manifest |
| `deploy/models.json` | export + `bench_mobile.py` | freeze_ref **`sepres_v2_c16n10` fused** + ECBSR + FSRCNN/Lite/Plus | **20k**；`deploy_fuse` for v2/ECBSR |

Inactive probe manifests: `results/_inactive/exp_runs/stage_b/`.

---

## Headline models

| Role | Run dir | Checkpoint (local) | Avg PSNR (20k) | Avg PSNR (30k) |
|---|---|---|---:|---:|
| Baseline | `exp_runs/fsrcnn_fix_clean_20k` | `_headline/checkpoints/fsrcnn_20k.pt` | 26.820 | 26.907 |
| Student (efficient) | `exp_runs/mobile_srnet_20k` | `_headline/checkpoints/mobile_srnet_base_20k.pt` | 27.159 | 27.201 |
| **Headline (best quality)** | `exp_runs/mobile_srnet_plus_20k` | `_headline/checkpoints/mobile_srnet_plus_20k.pt` | **27.331** | **27.384** |

30k continuation dirs: `*_30k/` (resume from 20k `latest.pt`, low-LR fine-tuning).

**Deploy pairing (important):**

- Quality main table: **20k** `best.pt` / `fair_budget_runs.json`.
- Phone JSON on disk: **20k** via `deploy/models.json` (re-exported + bench 2026-07-10).
- **A0 done:** quality–latency cells share 20k `best.pt` identity. Canonical phone file: `deploy/artifacts/results/mobile_benchmark_latest.json` (also `mobile_benchmark_20260710_191531.json`).

Symlinks in `results/_headline/checkpoints/` point at the three 20k `best.pt` files above.

---

## Active evidence

### `exp_runs/` (headline + RQ2 null + 30k)

| Path | Purpose |
|---|---|
| `fair_budget_manifest.json` | Launcher list for FSRCNN/Base/KD 20k |
| `fair_budget_runs.json` | Main comparison table source（20k + KD + A1a + ECBSR + **`sepres_v2_c16n10_20k` freeze_ref**） |
| `fair_budget_curves.png` | Fair-budget training curves |
| `plus_20k_manifest.json`, `plus_20k_reference.json` | Plus 20k control plane |
| `arch_30k_runs.json` | 20k vs 30k benchmark stability |
| `arch_30k_manifest.json`, `arch_30k_reference.json` | 30k continuation audit trail |
| `latency_audit.json` | Copy of canonical RTX audit (from report) |
| `kd_method_gates.json` | Pre-training KD gate results |
| `lambda_sweep_summary.json` | Pixel Charbonnier λ-sweep (null) |
| `kd_diagnostic.json` | Gradient cosine vs GT |
| `fsrcnn_fix_clean_20k/`, `mobile_srnet_20k/`, `mobile_srnet_plus_20k/` | 20k metrics + logs |
| `fsrcnn_fix_clean_20k_bs24/` | **A1a done** — FSRCNN bs=24 / 20k；avg PSNR 26.81 ≈ bs=8 |
| `ecbsr_m10c16_20k/` | **B3 done** — fused ~31K / 0.99G；avg PSNR 27.51；phone med 32.1 ms |
| `sepres_v2_20k_manifest.json` | B4 launcher registry (default train = c16n10) |
| `b4_v2_fuse_smoke.json` | B4 model/fuse/budget audit (all candidates PASS) |
| `b4_v2_prescreen.json` | **B4 Gate-0** export+graph_smoke PASS（随机权重；非正式 latency） |
| `b4_measurement_envelope.json` | **B4 Gate-1** Lite-sep vs ECBSR paired sessions → E_med/E_p90 |
| `b4_v2_posttrain_smoke.json` | Gate-2 **pipeline smoke**（latest.pt export；非正式） |
| `b4_v2_compare.json` | **B4 Gate-2 official** — freeze v2_b；val 27.96/28.02 tie；med **24.82/26.34**；p90 **29.16/30.10** tie；bench 27.37 |
| `b4_v2_desktop_latency.json` | Desktop CUDA fused v2 vs ECBSR（180²；fp16 med 1.46 / 1.61） |
| `sepres_v2_c16n10_20k/` | **freeze_ref** — val best 27.96；bench avg **27.37** + LPIPS **0.156** |
| `dual_plain_2k_manifest.json` | B4 round-2 launcher（Dual→Plain 2k） |
| `b4_dual_plain_fuse_smoke.json` | Dual→Plain fuse + budget **PASS**（27,348 / 7 conv） |
| `b4_dual_plain_prescreen.json` | Dual/Plain Gate-0 export **PASS**（`--skip-bench`；NCNN ~110KB） |
| `b4_dual_plain_posttrain_smoke.json` | E12：`latest.pt` fuse/export smoke |
| `dual_stream_c20n5_2k/` | B4 round-2 Dual 2k **done** — best val **26.18** |
| `plain_c20n5_2k/` | B4 round-2 Plain 2k **done** — best val **26.65** |
| `b4_dual_plain_2k_gate.json` | **INTERNAL** D18 FAIL — Dual 26.18 / Plain 26.65；**not for report** |
| `b5_train_20k_manifest.json` | B5 unified（MSE 3-wide + KD Stage-B 2k） |
| `b5a_plain_fuse_smoke.json` | B5a P0 Gate-0 fuse/budget **PASS**（iso-MAC vs PECSR） |
| `ablation_b5a_p0.json` | B5a P0 plain3x3 vs PECSR/ECBSR（val+bench） |
| `b5_multi_seed.json` | **B5 multi-seed** PECSR/ECBSR seeds 42/123/2026 — mean±std avg PSNR **27.42±0.05** / **27.57±0.06** |
| `b5_kd_stageb.json` | **B5 KD screen** ~2k；pixel λ0.2 **+0.51** dB / VGG λ0.01 **−0.13** vs matched λ=0 |
| `pecsr_pixel_kd0_2k/` / `pecsr_pixel_kd02_2k/` | PECSR pixel KD λ=0 / 0.2 + bench |
| `pecsr_vgg3_kd0_2k/` / `pecsr_vgg3_kd01_2k/` | PECSR VGG relu3 KD λ=0 / 0.01 + bench |
| `sepres_v2_c16n10_20k_s123/` / `_s2026/` | PECSR multi-seed runs + `benchmark_metrics.json` |
| `ecbsr_m10c16_20k_s123/` / `_s2026/` | ECBSR multi-seed runs + `benchmark_metrics.json` |
| `b5a_20k_manifest.json` | legacy（redirected） |
| `b5_confirm_20k_manifest.json` | legacy（redirected） |
| `pecsr_kd_20k_manifest.json` | legacy pointer → Stage-B 2k ids |
| `b2_profile_latest.json`, `b2_profile_note.md` | B2 desktop module timing + conclusions |
| `fsrcnn_fix_clean_30k/`, `mobile_srnet_30k/`, `mobile_srnet_plus_30k/` | 30k metrics + logs |
| `mobile_srnet_kd0_20k/`, `mobile_srnet_kd02_20k/` | KD null @ λ=0, 0.2 |

### `results/` figure dirs

| Path | Purpose |
|---|---|
| `training_analysis/*.png` | Learning curves (mirrored from report) |
| `summary/*.png` | Bar charts (PSNR, FLOPs, latency, …) |
| `latency/*.png` | RTX + mobile latency figures |
| `exp_runs/fair_budget_curves.png` | Legacy fair-budget plot |

### `deploy/artifacts/`

| Path | Purpose |
|---|---|
| `../deploy/models.json` | Checkpoint registry；**freeze_ref=`sepres_v2_c16n10`** + ECBSR fused |
| `results/mobile_benchmark_latest.json` | On-device NCNN Vulkan FP16 (**20k**, 2026-07-10) |
| `results/ecbsr_mobile_bench_latest.json` | **B3 official** fused ECBSR phone (best.pt, 300 iter; med 32.1 ms) |
| `results/sepres_v2_paired_phone_latest.json` | **B4 Gate-2** 3-session paired：v2 med mean **24.82** / p90 **29.16**；ECBSR med **26.34** / p90 **30.10**（勿与 B3 单次 32.1 混比） |
| `results/fused_sep_compare_latest.json` | B1: sep vs fused phone latency (20k; fused slower) |
| `figures/bars_mobile_median.png` | Mobile latency bar chart |
| `figures/flops_vs_mobile_latency.png` | FLOPs vs on-device latency |

### Report bundle

| Path | Purpose |
|---|---|
| [`../../report/`](../../report/) | Submission-facing report (LaTeX PDF, figures, metrics JSON) |

Canonical GPU latency: `../../report/assets/metrics/latency_audit.json` (regenerate: `cd ../../report/plot && python audit_latency.py`).

---

## Inactive

| Location | Contents |
|---|---|
| `_inactive/baseline_era/` | Pre fair-budget runs |
| `_inactive/exp_runs/` | 10k runs, 2k probes, λ-sweep 10k, paused/observe JSON, `fair_budget_runs_10k.json` |
| `_inactive/exp_runs/stage_b/` | Stage B Plus / VGG3 manifests (probes done) |
| `_inactive/latency_audit/` | Legacy Jul 6 audit JSON |
| `deploy/_inactive/artifacts/results/` | Superseded mobile benchmark JSON snapshots |
| `configs/_inactive/` | Superseded experiment configs |
| **`configs/README.md`** | **exp vs templates vs inactive map** |
| `scripts/_inactive/stage_b/` | Stage B pause/resume/watch + launchers |
| `scripts/_inactive/kd_probes/` | λ-sweep / KD parallel probe tooling |
| `scripts/_inactive/kd_diag/` | gate / diagnose / per-image KD |
| `scripts/_inactive/lab_plot_dup/` | Lab plot/audit copies（canonical=`report/plot/`） |
| `../../_inactive/planning_20260710/` | Superseded planning (next_steps, sug_*, …) |

> `_inactive` = 现行 workflow **不应再跑**。过夜脚本 `keep_awake.sh` + `win_*.ps1` 在 `scripts/` 根下，属 **ACTIVE ops**。

Implementation rules: [`../IMPLEMENTATION.md`](../IMPLEMENTATION.md) · Planning: [`../../progress/`](../../progress/)

---

## Regenerate figures

```bash
cd ../../report/plot
python build_arch_30k_runs.py
python build_model_summary.py
python plot_model_summary.py
python plot_flops_vs_latency.py
python plot_arch_30k.py
python plot_mobile_bench.py
python plot_training_analysis.py
python audit_latency.py          # GPU
python sync_lab_report.py        # mirror → Image-Super-Resolution/results/
```

```bash
cd ../../report/plot
python plot_fair_budget.py
```

## Eval headline model (quality table = 20k)

```bash
python scripts/eval_sr.py \
  --checkpoint results/exp_runs/mobile_srnet_plus_20k/checkpoints/best.pt \
  --save-json results/exp_runs/mobile_srnet_plus_20k/benchmark_metrics.json \
  --compute-lpips
```

30k eval (appendix / continuation only):

```bash
python scripts/eval_sr.py \
  --checkpoint results/exp_runs/mobile_srnet_plus_30k/checkpoints/best.pt \
  --save-json results/exp_runs/mobile_srnet_plus_30k/benchmark_metrics.json \
  --compute-lpips
```
