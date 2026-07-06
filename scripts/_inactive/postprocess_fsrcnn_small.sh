#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

TRAIN_PATTERN="scripts/train_fsrcnn.py --config configs/_inactive/train_fsrcnn_small.yaml"
CHECK_INTERVAL_SEC=30
MAX_WAIT_MINUTES="${MAX_WAIT_MINUTES:-720}"
MAX_WAIT_SEC=$((MAX_WAIT_MINUTES * 60))

echo "[postprocess] Waiting for small FSRCNN training to finish..."
waited=0
while pgrep -f "${TRAIN_PATTERN}" >/dev/null; do
  sleep "${CHECK_INTERVAL_SEC}"
  waited=$((waited + CHECK_INTERVAL_SEC))
  if (( waited >= MAX_WAIT_SEC )); then
    echo "[postprocess] Timeout waiting for training process."
    exit 1
  fi
done

echo "[postprocess] Training finished, running evaluations..."

conda run -n cv_env python scripts/eval_fsrcnn.py \
  --checkpoint results/fsrcnn_small/checkpoints/best.pt \
  --benchmark-root data/benchmarks \
  --scale 4 \
  --crop-border 4 \
  --device cuda \
  --save-json results/fsrcnn_small/benchmark_metrics.json

conda run -n cv_env python scripts/compare_metrics.py \
  --base results/bicubic_metrics.json \
  --target results/fsrcnn_small/benchmark_metrics.json \
  > results/fsrcnn_small/compare_vs_bicubic.txt

conda run -n cv_env python scripts/compare_metrics.py \
  --base results/fsrcnn_fix_clean/benchmark_metrics.json \
  --target results/fsrcnn_small/benchmark_metrics.json \
  > results/fsrcnn_small/compare_vs_fix_clean.txt

conda run -n cv_env python scripts/summarize_train_log.py \
  --log-path results/fsrcnn_small/train_log.jsonl \
  > results/fsrcnn_small/train_summary.txt

echo "[postprocess] Done. Outputs in results/fsrcnn_small/."
