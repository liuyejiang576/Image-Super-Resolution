#!/usr/bin/env bash
# Autonomous pipeline: MobileSRNet postprocess -> KD train -> KD eval -> quantize -> report
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "[pipeline] Stage 1: wait for MobileSRNet training + postprocess"
bash scripts/postprocess_mobile_srnet.sh

echo "[pipeline] Stage 2: KD training (100 epochs)"
conda run -n cv_env python scripts/train_mobile_srnet_kd.py \
  --config configs/train_mobile_srnet_kd.yaml \
  --device cuda

echo "[pipeline] Stage 3: KD evaluation + profile"
conda run -n cv_env python scripts/eval_sr.py \
  --checkpoint results/mobile_srnet_kd/checkpoints/best.pt \
  --benchmark-root data/benchmarks \
  --scale 4 --crop-border 4 --device cuda --compute-lpips \
  --save-json results/mobile_srnet_kd/benchmark_metrics.json

conda run -n cv_env python scripts/profile_model.py \
  --model-type checkpoint \
  --checkpoint results/mobile_srnet_kd/checkpoints/best.pt \
  --device cuda \
  --save-json results/mobile_srnet_kd/profile.json

conda run -n cv_env python scripts/_inactive/compare_metrics.py \
  --base results/mobile_srnet/benchmark_metrics.json \
  --target results/mobile_srnet_kd/benchmark_metrics.json \
  > results/mobile_srnet_kd/compare_vs_mobile.txt

echo "[pipeline] Stage 4: quantization benchmark"
conda run -n cv_env python scripts/quantize_benchmark.py \
  --checkpoint results/mobile_srnet_kd/checkpoints/best.pt \
  --device cuda \
  --save-json results/mobile_srnet_kd/quantization.json

echo "[pipeline] Stage 5: LPIPS refresh for FSRCNN baseline"
conda run -n cv_env python scripts/eval_sr.py \
  --checkpoint results/fsrcnn_fix_clean/checkpoints/best.pt \
  --benchmark-root data/benchmarks \
  --scale 4 --crop-border 4 --device cuda --compute-lpips \
  --save-json results/fsrcnn_fix_clean/benchmark_metrics_lpips.json

echo "[pipeline] Stage 6: final report"
conda run -n cv_env python scripts/build_final_report.py \
  --report-path results/final_report.md \
  --pareto-path results/pareto_frontier.png

echo "[pipeline] All stages complete."
