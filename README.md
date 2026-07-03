# Image Super-Resolution (Mobile-Efficient)

This repository contains the project scaffold for efficient 4x image super-resolution experiments with a contribution-focused workflow.

## Included docs

- `proposal.md`
- `plan.md`
- `suggestion.md` (if present)

## Quick start

1. Create/activate environment (example: `cv_env`).
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Verify environment:
   - `python scripts/check_env.py`
4. Run data sanity check:
   - `python scripts/sanity_check_data.py --config configs/data.yaml --num-samples 4 --output-dir results/sanity`
5. Run bicubic baseline metrics:
   - `python scripts/eval_bicubic.py --benchmark-root data/benchmarks --scale 4 --crop-border 4`

## FSRCNN baseline training

- Default training config: `configs/train_fsrcnn.yaml`
- Start training:
  - `python scripts/train_fsrcnn.py --config configs/train_fsrcnn.yaml`
- For a quick smoke run, set `train.max_train_steps_per_epoch` to a small number in the config.

## Data note

Datasets are intentionally ignored by git. Put local data under:

- `data/div2k/`
- `data/benchmarks/`
