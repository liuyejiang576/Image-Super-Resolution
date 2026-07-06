# Results manifest

Single source of truth for which artifacts are **headline** (report-facing), **active** (supporting evidence), or **inactive** (superseded but kept locally).

Checkpoints (`.pt`) and training logs stay local and gitignored. Small JSON/PNG/MD evidence files listed below may be tracked in git per `.gitignore` exceptions.

---

## Headline models (20k fair budget)

| Role | Run dir | Checkpoint (local) | Avg PSNR |
|---|---|---|---|
| Baseline | `exp_runs/fsrcnn_fix_clean_20k` | `_headline/checkpoints/fsrcnn_20k.pt` → `exp_runs/.../best.pt` | 26.820 |
| Student (efficient) | `exp_runs/mobile_srnet_20k` | `_headline/checkpoints/mobile_srnet_base_20k.pt` | 27.159 |
| **Headline (best quality)** | `exp_runs/mobile_srnet_plus_20k` | `_headline/checkpoints/mobile_srnet_plus_20k.pt` | **27.331** |

Symlinks in `results/_headline/checkpoints/` point at the three `best.pt` files above.

---

## Active evidence

### `exp_runs/` (headline + RQ2 null)

| Path | Purpose |
|---|---|
| `fair_budget_runs.json` | Main comparison table source |
| `fair_budget_curves.png` | Fair-budget training curves |
| `kd_method_gates.json` | Pre-training KD gate results |
| `lambda_sweep_summary.json` | Pixel Charbonnier λ-sweep (null) |
| `kd_diagnostic.json` | Gradient cosine vs GT |
| `stage_b_plus_manifest.json`, `plus_20k_manifest.json` | Capacity scaling audit trail |
| `fsrcnn_fix_clean_20k/` | FSRCNN 20k metrics + profile |
| `mobile_srnet_20k/` | Base 20k metrics + profile |
| `mobile_srnet_plus_20k/` | Plus 20k metrics + profile |
| `mobile_srnet_kd0_20k/`, `mobile_srnet_kd02_20k/` | KD null @ λ=0, 0.2 (Charbonnier) |

### Report bundles

| Path | Purpose |
|---|---|
| `final_report.md` | Submission-facing report |
| `qualitative/` | Crop panels (Base / Plus / FSRCNN) |
| `latency_audit/` | FP16 latency + FLOPs vs latency plot |
| `training_analysis/` | Learning curves + summary |

---

## Inactive configs (`configs/_inactive/`)

Superseded experiment configs (10k budget, 2k probes). Active headline configs remain in `configs/exp/`.

---

## Inactive results (`results/_inactive/`)

Moved here on 2026-07-06. Safe to delete if disk is tight; not cited in final report.

| Location | Contents |
|---|---|
| `_inactive/baseline_era/` | Pre fair-budget runs: `mobile_srnet`, `mobile_srnet_kd`, `fsrcnn*`, `kd_analysis`, early ablations |
| `_inactive/exp_runs/` | 10k runs, Plus/VGG3 2k probes, λ-sweep 10k extras, `kd_parallel_probe`, training logs |
| `_inactive/exp_runs/loose_logs/` | Launcher / watch / parallel logs from `exp_runs/` root |

---

## Regenerate figures

```bash
python scripts/make_qualitative_panels.py
python scripts/plot_flops_vs_latency.py
python scripts/plot_training_analysis.py
python scripts/audit_latency.py
```

## Eval headline model

```bash
python scripts/eval_sr.py \
  --checkpoint results/exp_runs/mobile_srnet_plus_20k/checkpoints/best.pt \
  --save-json results/exp_runs/mobile_srnet_plus_20k/benchmark_metrics.json \
  --compute-lpips
```
