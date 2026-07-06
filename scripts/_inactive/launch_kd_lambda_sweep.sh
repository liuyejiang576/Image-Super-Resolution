#!/usr/bin/env bash
# Launch 3 KD lambda-sweep runs (10k updates each) in parallel.
# Extends the existing clean {lambda=0.0, 0.2} 10k pair to a 5-point sweep.
set -u
cd /home/hyb/CV_project/Image-Super-Resolution
PY=/home/hyb/miniforge3/envs/cv_env/bin/python
LOG_DIR=results/exp_runs/logs
mkdir -p "$LOG_DIR"

declare -a RUNS=(
  "kd05:configs/_inactive/exp/mobile_srnet_kd05_10k.yaml:0.5"
  "kd10:configs/_inactive/exp/mobile_srnet_kd10_10k.yaml:1.0"
  "kd20:configs/_inactive/exp/mobile_srnet_kd20_10k.yaml:2.0"
)

PIDS=()
for entry in "${RUNS[@]}"; do
  name="${entry%%:*}"
  rest="${entry#*:}"
  cfg="${rest%%:*}"
  lam="${rest##*:}"
  outdir="results/exp_runs/mobile_srnet_${name}_10k"
  mkdir -p "$outdir/checkpoints"
  echo "[$(date +%H:%M:%S)] starting $name lambda=$lam cfg=$cfg"
  "$PY" scripts/train_mobile_srnet_kd.py \
    --config "$cfg" --lambda-kd "$lam" \
    > "$LOG_DIR/${name}_10k.log" 2>&1 &
  PIDS+=($!)
  echo "  pid=$!"
  # stagger starts slightly to avoid simultaneous first-epoch validation spike
  sleep 5
done

echo "[$(date +%H:%M:%S)] all launched, waiting for completion..."
FAIL=0
for p in "${PIDS[@]}"; do
  if ! wait "$p"; then
    echo "  pid $p FAILED"
    FAIL=1
  fi
done
echo "[$(date +%H:%M:%S)] done (fail=$FAIL)"
exit $FAIL
