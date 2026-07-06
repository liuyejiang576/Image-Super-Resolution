#!/usr/bin/env bash
# Resume paused KD 20k runs (both in parallel). Run after: bash scripts/resume_paused_kd.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=/home/hyb/miniforge3/envs/cv_env/bin/python
LOG_DIR=results/exp_runs/logs
mkdir -p "$LOG_DIR"

nohup "$PYTHON" scripts/train_mobile_srnet_kd.py \
  --config configs/exp/mobile_srnet_kd0_20k.yaml --lambda-kd 0.0 \
  --resume-from results/exp_runs/mobile_srnet_kd0_20k/checkpoints/latest.pt \
  >> "$LOG_DIR/train_mobile_srnet_kd0_20k.log" 2>&1 &

nohup "$PYTHON" scripts/train_mobile_srnet_kd.py \
  --config configs/exp/mobile_srnet_kd02_20k.yaml --lambda-kd 0.2 \
  --resume-from results/exp_runs/mobile_srnet_kd02_20k/checkpoints/latest.pt \
  >> "$LOG_DIR/train_mobile_srnet_kd02_20k.log" 2>&1 &

sleep 5
"$PYTHON" scripts/exp_status.py
echo ""
echo "To auto-eval when both finish: bash scripts/watch_and_finalize.sh"
