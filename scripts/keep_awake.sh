#!/usr/bin/env bash
# Keep Windows awake overnight so WSL training is not suspended.
#
#   bash scripts/keep_awake.sh lock     # apply powercfg never-sleep settings (no admin)
#   bash scripts/keep_awake.sh start    # start keep-awake loop + lock power
#   bash scripts/keep_awake.sh stop     # stop keep-awake loop
#   bash scripts/keep_awake.sh status   # show keep-awake + training + GPU
set -euo pipefail
cd "$(dirname "$0")/.."

PS=/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
PID_FILE=results/exp_runs/keep_awake.pid
LOG_FILE=results/exp_runs/keep_awake.log
WIN_PS1=$(wslpath -w "$PWD/scripts/win_keep_awake.ps1")
WIN_LOCK=$(wslpath -w "$PWD/scripts/win_lock_power.ps1")
mkdir -p results/exp_runs

PC=/mnt/c/Windows/System32/powercfg.exe

# Reliable powercfg lock from WSL (works without admin on this machine).
bash_power_lock() {
  local guid
  guid=$("$PC" /getactivescheme 2>&1 | grep -oiE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' | head -n1)
  [[ -n "$guid" ]] || { echo "ERROR: could not parse active power scheme GUID"; return 1; }
  echo "Active scheme: $guid"
  "$PC" /change standby-timeout-ac 0
  "$PC" /change standby-timeout-dc 0
  "$PC" /change hibernate-timeout-ac 0
  "$PC" /change hibernate-timeout-dc 0
  "$PC" /change monitor-timeout-ac 0
  "$PC" /change monitor-timeout-dc 0
  "$PC" /SETACVALUEINDEX "$guid" SUB_BUTTONS LIDACTION 0
  "$PC" /SETDCVALUEINDEX "$guid" SUB_BUTTONS LIDACTION 0
  "$PC" /SETACVALUEINDEX "$guid" SUB_SLEEP HYBRIDSLEEP 0
  "$PC" /SETDCVALUEINDEX "$guid" SUB_SLEEP HYBRIDSLEEP 0
  "$PC" /SETACTIVE "$guid"
  echo "powercfg lock applied (bash path)"
}

is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid=$(tr -d ' \n' < "$PID_FILE")
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

cmd_lock() {
  echo "Applying Windows power locks..."
  {
    echo "[$(date -Iseconds)] power lock"
    bash_power_lock
    echo "--- PS extras (lid/power button, WU active hours) ---"
    "$PS" -NoProfile -ExecutionPolicy Bypass -File "$WIN_LOCK" 2>&1 || true
  } | tee -a "$LOG_FILE"
}

cmd_start() {
  if is_running; then
    echo "keep-awake already running (pid=$(cat "$PID_FILE"))"
    return 0
  fi
  cmd_lock
  echo "Starting keep-awake loop..."
  nohup "$PS" -NoProfile -ExecutionPolicy Bypass \
    -File "$WIN_PS1" -LogFile "$(wslpath -w "$LOG_FILE")" \
    >> "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  sleep 1
  if is_running; then
    echo "keep-awake started (wsl pid=$(cat "$PID_FILE"), log=$LOG_FILE)"
  else
    echo "ERROR: keep-awake failed to start — check $LOG_FILE"
    rm -f "$PID_FILE"
    return 1
  fi
}

cmd_stop() {
  if ! is_running; then
    echo "keep-awake not running"
    rm -f "$PID_FILE"
    return 0
  fi
  local pid
  pid=$(tr -d ' \n' < "$PID_FILE")
  echo "Stopping keep-awake (pid=$pid)..."
  kill "$pid" 2>/dev/null || true
  sleep 1
  kill -9 "$pid" 2>/dev/null || true
  rm -f "$PID_FILE"
  echo "Stopped. Windows power lock released on PS exit."
}

cmd_status() {
  echo "=== keep-awake ==="
  if is_running; then
    echo "  RUNNING  wsl_pid=$(cat "$PID_FILE")  log=$LOG_FILE"
    tail -n 2 "$LOG_FILE" 2>/dev/null | sed 's/^/  /'
  else
    echo "  NOT running"
  fi
  echo
  echo "=== KD sweep ==="
  bash scripts/kd_sweep.sh status 2>/dev/null | head -14 || echo "  (kd_sweep status unavailable)"
}

case "${1:-}" in
  lock)   cmd_lock ;;
  start)  cmd_start ;;
  stop)   cmd_stop ;;
  status) cmd_status ;;
  *)
    cat <<'USAGE'
Usage: bash scripts/keep_awake.sh {lock|start|stop|status}

  lock    Apply powercfg never-sleep (no admin). Safe to run repeatedly.
  start   lock + start keep-awake PowerShell loop (recommended before sleep).
  stop    Stop keep-awake after training finishes.
  status  keep-awake + KD sweep summary.
USAGE
    exit 1 ;;
esac
