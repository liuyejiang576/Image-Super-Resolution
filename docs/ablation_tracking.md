# Ablation Tracking

## Experiment A: Stable Baseline (FSRCNN fix clean)

- Config: `configs/train_fsrcnn_fix_clean.yaml`
- Status: completed
- Best validation PSNR (DIV2K valid subset): `27.6685`
- Benchmark metrics: `results/fsrcnn_fix_clean/benchmark_metrics.json`
  - Set5: `29.1739 / 0.8233`
  - Set14: `26.5974 / 0.7277`
  - BSD100: `26.4074 / 0.6949`
  - Urban100: `23.6471 / 0.6810`

Model profile:

- Params: `24,683`
- Model size (fp32): `0.094 MB`
- Latency (bs=1, lr=180): `1.1013 ms`

## Experiment B: Small Model (FSRCNN-small)

- Config: `configs/train_fsrcnn_small.yaml`
- Status: running
- Goal: compare quality-efficiency trade-off vs baseline.

Model profile:

- Params: `12,019`
- Model size (fp32): `0.046 MB`
- Latency (bs=1, lr=180): `0.8120 ms`

Pending after training:

1. Run benchmark eval for `results/fsrcnn_small/checkpoints/best.pt`
2. Compare with bicubic and baseline using `scripts/compare_metrics.py`
3. Decide if small model is acceptable Pareto alternative
