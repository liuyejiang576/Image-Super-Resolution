#!/usr/bin/env bash
# Control the KD lambda-sweep (3 parallel 10k-update runs): pause / resume / status.
#
#   bash scripts/kd_sweep.sh pause  [--dry-run]   stop training + watcher, free GPU, keep checkpoints
#   bash scripts/kd_sweep.sh resume [--dry-run]   relaunch all not-yet-done runs from latest.pt + restart watcher
#   bash scripts/kd_sweep.sh status               concise table of runs / watcher / launcher / GPU
#   bash scripts/kd_sweep.sh watch [--watch SEC]  rich live progress (delegates to watch_kd_progress.py)
#
# Why pause also kills the watcher: watch_lambda_sweep.sh runs analyze_lambda_sweep.py as soon as
# no training procs are alive, which would analyze a partial sweep. Pause stops it; resume restarts it.
#
# Resume safety: training saves checkpoints/latest.pt after every completed epoch, so a SIGTERM
# mid-epoch only discards the in-progress epoch; --resume-from latest.pt restarts it cleanly.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/hyb/miniforge3/envs/cv_env/bin/python
LOG_DIR=results/exp_runs/logs
PAUSE_STATE=results/exp_runs/kd_sweep_paused.json
mkdir -p "$LOG_DIR"

# name:lambda  (single source of truth for the 3-point lambda-sweep)
RUNS=("kd05:0.5" "kd10:1.0" "kd20:2.0")

run_id()  { echo "mobile_srnet_${1}_10k"; }
cfg_of()  { echo "configs/_inactive/exp/mobile_srnet_${1}_10k.yaml"; }
ckpt_of() { echo "results/exp_runs/$(run_id "$1")/checkpoints/latest.pt"; }
log_of()  { echo "$LOG_DIR/train_$(run_id "$1").log"; }

last_epoch() {  # $1=name -> last epoch logged (0 if none)
  local f="results/exp_runs/$(run_id "$1")/train_log.jsonl"
  [[ -f "$f" ]] || { echo 0; return; }
  local n
  n=$(tail -n1 "$f" | grep -oE '"epoch":[[:space:]]*[0-9]+' | grep -oE '[0-9]+' | head -n1)
  echo "${n:-0}"
}

target_epochs() {  # $1=name -> train.epochs from config (the only `epochs:` key in these configs)
  grep -E '^[[:space:]]*epochs:[[:space:]]*[0-9]+' "$(cfg_of "$1")" | head -n1 | grep -oE '[0-9]+' | head -n1
}

proc_alive_pids() {  # $1=config path -> space-separated pids (main + dataloader workers)
  pgrep -f "train_mobile_srnet_kd.py --config $1" 2>/dev/null | tr '\n' ' ' | sed 's/ $//'
}
watcher_pids()  { pgrep -f "watch_lambda_sweep.sh"        2>/dev/null | tr '\n' ' ' | sed 's/ $//'; }
launcher_pids() { pgrep -f "launch_kd_lambda_sweep.sh"    2>/dev/null | tr '\n' ' ' | sed 's/ $//'; }

fmt3() {  # $1=numeric string or "-" -> 3-decimal float, or "-" if not a number
  awk -v x="$1" 'BEGIN{ if (x=="" || x=="-") print "-"; else if (x+0==x) printf "%.3f", x; else print "-" }'
}

# ----- status -----------------------------------------------------------------
cmd_status() {
  echo "KD lambda-sweep status  —  $(date '+%Y-%m-%d %H:%M:%S')"
  echo
  printf "%-11s %-8s %9s %9s %9s %9s  %s\n" run state epoch val_psnr best step pids
  printf -- "----------------------------------------------------------------------\n"
  for e in "${RUNS[@]}"; do
    local name="${e%%:*}"
    local cfg; cfg=$(cfg_of "$name")
    local f="results/exp_runs/$(run_id "$name")/train_log.jsonl"
    local target; target=$(target_epochs "$name")
    local pids; pids=$(proc_alive_pids "$cfg")
    local state="pending" epoch=0 psnr="-" best="-" step="-"
    if [[ -f "$f" ]]; then
      local last; last=$(tail -n1 "$f")
      epoch=$(echo "$last" | grep -oE '"epoch":[[:space:]]*[0-9]+'        | grep -oE '[0-9]+'  | head -n1); epoch=${epoch:-0}
      step=$( echo "$last" | grep -oE '"global_step":[[:space:]]*[0-9]+'  | grep -oE '[0-9]+'  | head -n1); step=${step:-"-"}
      psnr=$( echo "$last" | grep -oE '"val_psnr":[[:space:]]*[0-9.]+'    | grep -oE '[0-9.]+$')
      best=$( grep -oE '"val_psnr":[[:space:]]*[0-9.]+' "$f" | grep -oE '[0-9.]+$' | sort -rn | head -n1)
      psnr=$(fmt3 "$psnr"); best=$(fmt3 "$best")
    fi
    if [[ -n "$pids" ]]; then state="running"
    elif [[ "$epoch" -ge "$target" ]] 2>/dev/null; then state="done"
    elif [[ -f "$f" ]]; then state="paused"; fi
    printf "%-11s %-8s %9s %9s %9s %9s  %s\n" "$name" "$state" "${epoch}/${target}" "$psnr" "$best" "$step" "${pids:-(none)}"
  done
  echo
  local wp; wp=$(watcher_pids)
  local lp; lp=$(launcher_pids)
  echo "watcher:   ${wp:-not running}"
  echo "launcher:  ${lp:-not running}"
  [[ -f "$PAUSE_STATE" ]] && echo "pause-state: $PAUSE_STATE"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU:       $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo '(nvidia-smi failed)')"
  fi
  echo
  echo "Rich progress: bash scripts/kd_sweep.sh watch --watch 60"
}

# ----- pause ------------------------------------------------------------------
cmd_pause() {
  local dry=0
  [[ "${2:-}" == "--dry-run" ]] && dry=1
  echo "Pausing KD lambda-sweep  —  $(date '+%Y-%m-%d %H:%M:%S')"

  local state_json
  state_json=$("$PY" - <<'PY'
import json, datetime, yaml
from pathlib import Path
RUNS = [("kd05", 0.5), ("kd10", 1.0), ("kd20", 2.0)]
def run_id(n): return f"mobile_srnet_{n}_10k"
def last_epoch(n):
    f = Path(f"results/exp_runs/{run_id(n)}/train_log.jsonl")
    if not f.exists(): return 0
    lines = [l for l in f.read_text().splitlines() if l.strip()]
    return int(json.loads(lines[-1])["epoch"]) if lines else 0
def target(n):
    return int(yaml.safe_load(Path(f"configs/_inactive/exp/mobile_srnet_{n}_10k.yaml").read_text())["train"]["epochs"])
out = {
    "paused_at": datetime.datetime.now().astimezone().isoformat(),
    "runs": [
        {"run_id": run_id(n), "name": n, "lambda_kd": lam, "epoch": last_epoch(n),
         "target_epochs": target(n),
         "resume_from": f"results/exp_runs/{run_id(n)}/checkpoints/latest.pt"}
        for n, lam in RUNS
    ],
}
print(json.dumps(out, indent=2))
PY
)
  echo "Pre-pause state captured:"
  echo "$state_json"
  if [[ $dry -eq 1 ]]; then
    echo "(dry-run: not killing anything, not writing state file)"
    return 0
  fi
  echo "$state_json" > "$PAUSE_STATE"

  # 1) stop the watcher FIRST so it does not run premature analysis on a partial sweep
  local wp; wp=$(watcher_pids)
  if [[ -n "$wp" ]]; then echo "Stopping watcher (pids: $wp)..."; kill $wp 2>/dev/null || true;
  else echo "Watcher not running."; fi

  # 2) SIGTERM each training run (main proc + dataloader workers, matched by config path)
  for e in "${RUNS[@]}"; do
    local name="${e%%:*}"; local cfg; cfg=$(cfg_of "$name")
    local pids; pids=$(proc_alive_pids "$cfg")
    if [[ -n "$pids" ]]; then echo "SIGTERM $name (pids: $pids)..."; kill -TERM $pids 2>/dev/null || true;
    else echo "$name: no training proc (already stopped?)"; fi
  done

  # 3) wait up to 30s for graceful exit, then SIGKILL stragglers
  local waited=0
  while [[ $waited -lt 30 ]]; do
    local remain=0
    for e in "${RUNS[@]}"; do
      local name="${e%%:*}"; local cfg; cfg=$(cfg_of "$name")
      [[ -n "$(proc_alive_pids "$cfg")" ]] && remain=1
    done
    [[ $remain -eq 0 ]] && break
    sleep 2; waited=$((waited + 2))
  done
  for e in "${RUNS[@]}"; do
    local name="${e%%:*}"; local cfg; cfg=$(cfg_of "$name")
    local pids; pids=$(proc_alive_pids "$cfg")
    if [[ -n "$pids" ]]; then echo "SIGKILL $name (pids: $pids)..."; kill -KILL $pids 2>/dev/null || true; fi
  done

  # 4) stop the launcher (it is just a `wait` loop; harmless to kill)
  local lp; lp=$(launcher_pids)
  if [[ -n "$lp" ]]; then echo "Stopping launcher (pids: $lp)..."; kill $lp 2>/dev/null || true; fi

  # 5) wait for GPU memory to drain
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "Waiting for GPU to drain..."
    local gw=0
    while [[ $gw -lt 20 ]]; do
      local used
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ')
      [[ -z "$used" ]] && break
      [[ "$used" -lt 200 ]] && break
      sleep 2; gw=$((gw + 2))
    done
    echo "GPU now: $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null)"
  fi

  echo
  echo "Paused. Checkpoints kept at results/exp_runs/mobile_srnet_*_10k/checkpoints/latest.pt"
  echo "Any in-progress epoch was discarded; resume will restart it from the last completed epoch."
  echo "Resume with: bash scripts/kd_sweep.sh resume"
}

# ----- resume -----------------------------------------------------------------
cmd_resume() {
  local dry=0
  [[ "${2:-}" == "--dry-run" ]] && dry=1
  echo "Resuming KD lambda-sweep  —  $(date '+%Y-%m-%d %H:%M:%S')"

  for e in "${RUNS[@]}"; do
    local name="${e%%:*}" lam="${e##*:}"
    local cfg; cfg=$(cfg_of "$name")
    local ckpt; ckpt=$(ckpt_of "$name")
    local target; target=$(target_epochs "$name")
    local ep; ep=$(last_epoch "$name")

    if [[ -n "$(proc_alive_pids "$cfg")" ]]; then
      echo "  $name: already running — skip"; continue; fi
    if [[ "$ep" -ge "$target" ]] 2>/dev/null; then
      echo "  $name: already at target ($ep/$target) — skip"; continue; fi
    if [[ ! -f "$ckpt" ]]; then
      echo "  $name: NO checkpoint at $ckpt — cannot resume, skip"; continue; fi

    echo "  $name: resume from epoch $ep/$target (λ=$lam) ckpt=$ckpt"
    if [[ $dry -eq 1 ]]; then continue; fi
    nohup "$PY" scripts/train_mobile_srnet_kd.py \
      --config "$cfg" --lambda-kd "$lam" \
      --resume-from "$ckpt" \
      >> "$(log_of "$name")" 2>&1 &
    sleep 5  # stagger starts to avoid simultaneous first-epoch validation spike
  done

  if [[ $dry -eq 1 ]]; then echo "(dry-run: not launching, not restarting watcher)"; return 0; fi

  # restart the watcher (only if not already running)
  local wp; wp=$(watcher_pids)
  if [[ -z "$wp" ]]; then
    echo "Restarting watcher..."
    nohup bash scripts/watch_lambda_sweep.sh > /dev/null 2>&1 &
    echo "  watcher pid=$!"
  else
    echo "Watcher already running (pids: $wp)."
  fi

  echo
  echo "Resume launched. Monitor: bash scripts/kd_sweep.sh status  (or)  bash scripts/kd_sweep.sh watch --watch 60"
}

# ----- dispatch ---------------------------------------------------------------
case "${1:-}" in
  pause)  cmd_pause  "$@" ;;
  resume) cmd_resume "$@" ;;
  status) cmd_status ;;
  watch)
    shift
    exec "$PY" scripts/watch_kd_progress.py "$@"
    ;;
  *)
    cat <<'USAGE'
Usage: bash scripts/kd_sweep.sh {pause|resume|status|watch} [opts]

  pause  [--dry-run]   Stop training + watcher, free GPU, keep checkpoints.
  resume [--dry-run]   Relaunch not-yet-done runs from latest.pt + restart watcher.
  status               Concise table of runs / watcher / launcher / GPU.
  watch  [--watch SEC] Rich live progress (per-run epoch, spent, ETA, progress %).
USAGE
    exit 1 ;;
esac
