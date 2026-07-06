#!/usr/bin/env bash
# Poll the three lambda-sweep 10k runs; when all reach epoch 200, run the analysis.
set -u
cd /home/hyb/CV_project/Image-Super-Resolution
PY=/home/hyb/miniforge3/envs/cv_env/bin/python
RUNS=(mobile_srnet_kd05_10k mobile_srnet_kd10_10k mobile_srnet_kd20_10k)
TARGET_EPOCHS=200
LOG=results/exp_runs/lambda_sweep_watch.log

echo "[$(date +%H:%M:%S)] watcher started, target=$TARGET_EPOCHS epochs" | tee -a "$LOG"

while true; do
  all_done=1
  status=""
  for r in "${RUNS[@]}"; do
    f="results/exp_runs/$r/train_log.jsonl"
    if [ ! -f "$f" ]; then
      n=0
    else
      n=$(wc -l < "$f" | tr -d ' ')
    fi
    status="$status $r=$n"
    if [ "$n" -lt "$TARGET_EPOCHS" ]; then
      all_done=0
    fi
  done
  echo "[$(date +%H:%M:%S)]$status" | tee -a "$LOG"

  if [ "$all_done" -eq 1 ]; then
    echo "[$(date +%H:%M:%S)] all runs reached $TARGET_EPOCHS epochs, running analysis" | tee -a "$LOG"
    "$PY" scripts/analyze_lambda_sweep.py 2>&1 | tee -a "$LOG"
    echo "[$(date +%H:%M:%S)] analysis complete" | tee -a "$LOG"
    exit 0
  fi

  # also bail if the training processes are all gone (crash or completion)
  alive=$(pgrep -c -f "train_mobile_srnet_kd.py --config configs/exp/mobile_srnet_kd(05|10|20)_10k" 2>/dev/null || echo 0)
  if [ "$alive" -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] no training procs alive; running analysis on whatever finished" | tee -a "$LOG"
    "$PY" scripts/analyze_lambda_sweep.py 2>&1 | tee -a "$LOG"
    echo "[$(date +%H:%M:%S)] analysis complete (after proc exit)" | tee -a "$LOG"
    exit 0
  fi

  sleep 300
done
