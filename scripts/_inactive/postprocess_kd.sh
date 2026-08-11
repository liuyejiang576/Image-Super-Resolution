#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

TRAIN_PATTERN="scripts/train_mobile_srnet_kd.py"
echo "[postprocess-kd] Waiting for KD training..."
while pgrep -f "${TRAIN_PATTERN}" >/dev/null; do sleep 30; done

conda run -n cv_env python scripts/eval_sr.py \
  --checkpoint results/mobile_srnet_kd/checkpoints/best.pt \
  --compute-lpips --save-json results/mobile_srnet_kd/benchmark_metrics.json

conda run -n cv_env python scripts/profile_model.py \
  --model-type checkpoint \
  --checkpoint results/mobile_srnet_kd/checkpoints/best.pt \
  --save-json results/mobile_srnet_kd/profile.json

conda run -n cv_env python scripts/quantize_benchmark.py \
  --checkpoint results/mobile_srnet_kd/checkpoints/best.pt \
  --save-json results/mobile_srnet_kd/quantization.json

conda run -n cv_env python scripts/_inactive/compare_metrics.py \
  --base results/mobile_srnet/benchmark_metrics.json \
  --target results/mobile_srnet_kd/benchmark_metrics.json \
  > results/mobile_srnet_kd/compare_vs_mobile.txt

conda run -n cv_env python scripts/build_final_report.py
echo "[postprocess-kd] Done."
