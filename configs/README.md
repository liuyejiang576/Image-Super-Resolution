# configs/ — what to open

Locked training recipes live in **`exp/*.yaml`**. Root `train_*.yaml` files are script templates only.

## Report cast (start here)

| File | Model |
|---|---|
| `exp/sepres_v2_c16n10_20k.yaml` | **PECSR** (default) |
| `exp/ecbsr_m10c16_20k.yaml` | ECBSR baseline |
| `exp/fsrcnn_fix_clean_20k_bs24.yaml` | FSRCNN (equal batch) |
| `exp/abl_plain3x3_c16n10_20k.yaml` | Plain-body ablation |
| `exp/sepres_v2_c16n10_20k_s*.yaml` / `exp/ecbsr_m10c16_20k_s*.yaml` | Multi-seed |
| `exp/pecsr_*_kd*_2k.yaml` | Short KD screens (after MSE) |

Launch via the matching `scripts/*_20k.py resume` helpers when available.

## Other `exp/` recipes

Historical MobileSRNet Lite/Plus / KD and 30k continuations remain under `exp/` for reproducibility; they are not the PECSR headline path.
