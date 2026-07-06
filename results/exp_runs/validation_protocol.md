# Validation protocol for experiment runs

## Checkpoint selection

- During training, validation uses a **fixed subset of 100 images** from DIV2K-valid (`validation.max_images: 100`).
- Best checkpoint is selected by highest DIV2K-valid PSNR on this subset.
- For final reporting, the selected checkpoint is evaluated on all four standard benchmarks (Set5, Set14, BSD100, Urban100) with full image sets.

## Rationale

Earlier runs used only 20 validation images, which can mis-rank close checkpoints. The 100-image subset reduces variance while keeping epoch-time validation tractable on RTX 4060.

## Fair comparison rule

All fair-budget retraining runs use:

- Same update budget (10k or 20k optimizer steps)
- Milestones at 60% and 85% of total epochs
- Batch size chosen from GPU probe recommendations (OOM-free, best throughput under ~7.2 GB peak)

## KD isolation rule

KD deconfounding runs use `train_mobile_srnet_kd.py` for both conditions:

- `lambda_kd = 0.0` — Charbonnier-only (no teacher term)
- `lambda_kd = 0.2` — Charbonnier + teacher output matching

All other hyperparameters are identical.
