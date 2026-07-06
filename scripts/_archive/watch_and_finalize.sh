#!/usr/bin/env bash
# Poll until all fair-budget training runs finish, then evaluate and refresh report.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=/home/hyb/miniforge3/envs/cv_env/bin/python
POLL_SEC="${POLL_SEC:-300}"

pending_count() {
  $PYTHON scripts/exp_status.py --pending-count --no-write
}

while [[ "$(pending_count)" -gt 0 ]]; do
  echo "[$(date -Iseconds)] $(pending_count) run(s) incomplete:"
  $PYTHON scripts/exp_status.py --no-write
  sleep "$POLL_SEC"
done

echo "[$(date -Iseconds)] All training complete. Running evaluation..."
$PYTHON scripts/eval_exp_runs.py
$PYTHON scripts/build_enhanced_report.py
$PYTHON scripts/exp_status.py
echo "[$(date -Iseconds)] Done."
