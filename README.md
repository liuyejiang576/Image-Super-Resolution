# Image Super-Resolution (Mobile-Efficient)

Mobile-efficient 4× super-resolution course project: **MobileSRNet-Plus** beats FSRCNN on quality at lower FLOPs; capacity scaling (Base → Plus) is the main win; KD was tested and rejected.

## Docs

| File | Role |
|---|---|
| `results/final_report.md` | Submission report (headline results) |
| `results/MANIFEST.md` | Which runs/artifacts are active vs archived |
| `report_plan.md` | Experiment checklist (complete) |
| `proposal.md`, `plan.md` | Original project docs |
| `docs/ablation_tracking.md` | Ablation notes |

Master file registry (active/inactive across the whole `CV_project/` tree, **outside this git repo**): `../codebase.md`

## Headline results (fair-budget 20k)

| Model | Avg PSNR | FLOPs (G) | Checkpoint |
|---|---:|---:|---|
| FSRCNN | 26.820 | 7.408 | `results/_headline/checkpoints/fsrcnn_20k.pt` |
| MobileSRNet-Base | 27.159 | 0.976 | `results/_headline/checkpoints/mobile_srnet_base_20k.pt` |
| **MobileSRNet-Plus** | **27.331** | 2.163 | `results/_headline/checkpoints/mobile_srnet_plus_20k.pt` |

Checkpoints are local only (gitignored). Metrics JSON is tracked under `results/exp_runs/*/benchmark_metrics.json`.

## Quick start

```bash
pip install -r requirements.txt
python scripts/check_env.py
python scripts/sanity_check_data.py --config configs/data.yaml --num-samples 4 --output-dir results/sanity
```

## Reproduce headline eval / figures

```bash
# Eval Plus on benchmarks
python scripts/eval_sr.py \
  --checkpoint results/exp_runs/mobile_srnet_plus_20k/checkpoints/best.pt \
  --save-json results/exp_runs/mobile_srnet_plus_20k/benchmark_metrics.json \
  --compute-lpips

# Figures
python scripts/make_qualitative_panels.py
python scripts/plot_flops_vs_latency.py
python scripts/plot_training_analysis.py
python scripts/audit_latency.py
```

## Training (reference)

Fair-budget configs live in `configs/exp/`:

- `fsrcnn_fix_clean_20k.yaml`
- `mobile_srnet_20k.yaml`
- `mobile_srnet_plus_20k.yaml`

Plus 20k launcher: `python scripts/plus_20k.py watch|pause|resume`

General pipeline: `python scripts/run_exp_pipeline.py --config configs/exp/<name>.yaml`

## Data

Datasets are gitignored. Local layout:

- `data/div2k/`
- `data/benchmarks/`

Or symlink from `../data/` at the `CV_project` level.

## Repo layout

```
configs/          Training configs (base + configs/exp/)
scripts/          Training, eval, plotting, experiment runners
scripts/_archive/ One-off sweep/launcher scripts (inactive)
src/              Models (mobile_srnet, fsrcnn) and utilities
results/          Local artifacts; see MANIFEST.md
results/_archive/ Superseded runs (local only)
results/_headline/ Symlinks to best checkpoints (local only)
```
