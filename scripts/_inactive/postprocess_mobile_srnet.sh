#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

TRAIN_PATTERN="scripts/train_mobile_srnet.py --config configs/train_mobile_srnet.yaml"
CHECK_INTERVAL_SEC=30

echo "[postprocess] Waiting for MobileSRNet training..."
while pgrep -f "${TRAIN_PATTERN}" >/dev/null; do
  sleep "${CHECK_INTERVAL_SEC}"
done

echo "[postprocess] Running MobileSRNet evaluations..."
conda run -n cv_env python scripts/eval_sr.py \
  --checkpoint results/mobile_srnet/checkpoints/best.pt \
  --benchmark-root data/benchmarks \
  --scale 4 \
  --crop-border 4 \
  --device cuda \
  --compute-lpips \
  --save-json results/mobile_srnet/benchmark_metrics.json

conda run -n cv_env python scripts/profile_model.py \
  --model-type checkpoint \
  --checkpoint results/mobile_srnet/checkpoints/best.pt \
  --device cuda \
  --save-json results/mobile_srnet/profile.json

conda run -n cv_env python scripts/compare_metrics.py \
  --base results/bicubic_metrics.json \
  --target results/mobile_srnet/benchmark_metrics.json \
  > results/mobile_srnet/compare_vs_bicubic.txt

conda run -n cv_env python scripts/compare_metrics.py \
  --base results/fsrcnn_fix_clean/benchmark_metrics.json \
  --target results/mobile_srnet/benchmark_metrics.json \
  > results/mobile_srnet/compare_vs_fsrcnn.txt

conda run -n cv_env python scripts/summarize_train_log.py \
  --log-path results/mobile_srnet/train_log.jsonl \
  > results/mobile_srnet/train_summary.txt

echo "[postprocess] MobileSRNet done."
