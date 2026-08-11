#!/usr/bin/env python3
"""Shared post-train scheduler (IMPLEMENTATION §12 / principles E12).

Every fair-budget family that needs deploy declares ``posttrain`` in its
manifest. Control launchers call this after train success; ``resume`` may also
arm a detached waiter so an already-running job still gets the pipeline.

  # Arm waiter for a run (safe if already scheduled)
  python scripts/posttrain_scheduler.py arm --run-id sepres_v2_c16n10_20k \\
      --manifest results/exp_runs/sepres_v2_20k_manifest.json

  # Run immediately (train must already be done)
  python scripts/posttrain_scheduler.py run --run-id sepres_v2_c16n10_20k \\
      --manifest results/exp_runs/sepres_v2_20k_manifest.json

  # Status / cancel
  python scripts/posttrain_scheduler.py status --run-id sepres_v2_c16n10_20k
  python scripts/posttrain_scheduler.py cancel --run-id sepres_v2_c16n10_20k
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

PY = sys.executable
if Path("/home/hyb/miniforge3/envs/cv_env/bin/python").exists():
    PY = "/home/hyb/miniforge3/envs/cv_env/bin/python"

STATE_DIR = ROOT / "results/exp_runs/posttrain"
LOG_DIR = ROOT / "results/exp_runs/logs"


def state_path(run_id: str) -> Path:
    return STATE_DIR / f"{run_id}.json"


def load_state(run_id: str) -> dict | None:
    p = state_path(run_id)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def save_state(run_id: str, payload: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "updated_at": datetime.now().astimezone().isoformat()}
    state_path(run_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_manifest(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def entry_for(manifest: list[dict], run_id: str) -> dict:
    for e in manifest:
        if e["run_id"] == run_id:
            return e
    raise SystemExit(f"run_id not in manifest: {run_id}")


def posttrain_spec(entry: dict) -> dict | None:
    pt = entry.get("posttrain")
    if not pt or not pt.get("enabled", True):
        return None
    return pt


def build_cmd(entry: dict) -> list[str]:
    pt = posttrain_spec(entry)
    if not pt:
        raise SystemExit(f"no posttrain declared for {entry['run_id']}")
    script = pt["script"]
    args = list(pt.get("args") or [])
    # Ensure run-id is present when the family script expects it
    if "--run-id" not in args and entry["run_id"] not in args:
        args = ["--run-id", entry["run_id"], *args]
    return [PY, script, *args]


def train_done(entry: dict) -> bool:
    run_id = entry["run_id"]
    log = ROOT / "results/exp_runs" / run_id / "train_log.jsonl"
    best = ROOT / "results/exp_runs" / run_id / "checkpoints/best.pt"
    if not log.exists() or not best.exists():
        return False
    rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        return False
    return int(rows[-1]["epoch"]) >= int(entry["epochs"])


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ProcessLookupError, ValueError):
        return False


def cancel(run_id: str) -> None:
    st = load_state(run_id) or {}
    pid = st.get("waiter_pid") or st.get("runner_pid")
    if pid_alive(pid):
        try:
            os.kill(int(pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
    st.update({"state": "cancelled", "waiter_pid": None, "runner_pid": None})
    save_state(run_id, st)
    print(f"cancelled {run_id}")


def run_posttrain(entry: dict, *, force: bool = False) -> int:
    """Execute posttrain now. Returns process exit code."""
    run_id = entry["run_id"]
    pt = posttrain_spec(entry)
    if not pt:
        print(f"[posttrain] {run_id}: not declared — skip")
        return 0
    if not force and not train_done(entry):
        print(f"[posttrain] {run_id}: train not done — refuse")
        return 2

    cmd = build_cmd(entry)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"posttrain_{run_id}.log"
    save_state(
        run_id,
        {
            "run_id": run_id,
            "state": "running",
            "cmd": cmd,
            "log": str(log_path.relative_to(ROOT)),
            "started_at": datetime.now().astimezone().isoformat(),
        },
    )
    print(f"[posttrain] {run_id}: {' '.join(cmd)}")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n--- posttrain {datetime.now():%Y-%m-%d %H:%M:%S} ---\n")
        log.write(" ".join(cmd) + "\n")
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    state = "done" if proc.returncode == 0 else "failed"
    save_state(
        run_id,
        {
            "run_id": run_id,
            "state": state,
            "cmd": cmd,
            "exit_code": proc.returncode,
            "log": str(log_path.relative_to(ROOT)),
            "finished_at": datetime.now().astimezone().isoformat(),
        },
    )
    print(f"[posttrain] {run_id}: {state} (exit={proc.returncode}) log={log_path}")
    return proc.returncode


def arm_waiter(entry: dict, *, poll_sec: int = 60, force: bool = False) -> None:
    """Detach a process that waits for train_done then runs posttrain."""
    run_id = entry["run_id"]
    if not posttrain_spec(entry):
        print(f"[posttrain] {run_id}: no posttrain in manifest — not arming")
        return

    st = load_state(run_id) or {}
    if st.get("state") == "done" and not force:
        print(f"[posttrain] {run_id}: already done — not re-arming")
        return
    if st.get("state") in ("waiting", "running") and pid_alive(st.get("waiter_pid")):
        print(f"[posttrain] {run_id}: waiter already alive pid={st['waiter_pid']}")
        return

    if train_done(entry):
        print(f"[posttrain] {run_id}: train already done — running now")
        run_posttrain(entry, force=True)
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    waiter_log = LOG_DIR / f"posttrain_waiter_{run_id}.log"
    cmd = [
        PY,
        "scripts/posttrain_scheduler.py",
        "wait",
        "--run-id",
        run_id,
        "--manifest",
        str(Path(entry.get("_manifest_path", "results/exp_runs/sepres_v2_20k_manifest.json"))),
        "--poll-sec",
        str(poll_sec),
    ]
    # Prefer absolute manifest path passed by caller via env
    manifest_env = os.environ.get("POSTTRAIN_MANIFEST")
    if manifest_env:
        cmd[cmd.index("--manifest") + 1] = manifest_env

    log_file = open(waiter_log, "a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env={**os.environ, "POSTTRAIN_MANIFEST": cmd[cmd.index("--manifest") + 1]},
    )
    log_file.close()
    save_state(
        run_id,
        {
            "run_id": run_id,
            "state": "waiting",
            "waiter_pid": proc.pid,
            "cmd": cmd,
            "log": str(waiter_log.relative_to(ROOT)),
            "armed_at": datetime.now().astimezone().isoformat(),
        },
    )
    print(f"[posttrain] {run_id}: armed waiter pid={proc.pid} (poll={poll_sec}s)")


def cmd_wait(manifest_path: Path, run_id: str, poll_sec: int) -> int:
    manifest = load_manifest(manifest_path)
    entry = entry_for(manifest, run_id)
    entry["_manifest_path"] = str(manifest_path)
    print(f"[posttrain-wait] {run_id} until epoch>={entry['epochs']} ...")
    while not train_done(entry):
        save_state(
            run_id,
            {
                "run_id": run_id,
                "state": "waiting",
                "waiter_pid": os.getpid(),
                "poll_sec": poll_sec,
            },
        )
        time.sleep(poll_sec)
    print(f"[posttrain-wait] {run_id} train done — launching posttrain")
    return run_posttrain(entry, force=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Shared post-train scheduler")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--run-id", required=True)
        p.add_argument(
            "--manifest",
            default="results/exp_runs/sepres_v2_20k_manifest.json",
            help="Manifest that declares posttrain for this run_id",
        )

    p_arm = sub.add_parser("arm", help="Arm detached waiter after resume/start")
    add_common(p_arm)
    p_arm.add_argument("--poll-sec", type=int, default=60)
    p_arm.add_argument("--force", action="store_true")

    p_run = sub.add_parser("run", help="Run posttrain now (train must be done)")
    add_common(p_run)
    p_run.add_argument("--force", action="store_true")

    p_wait = sub.add_parser("wait", help="Internal: poll then run (used by arm)")
    add_common(p_wait)
    p_wait.add_argument("--poll-sec", type=int, default=60)

    p_status = sub.add_parser("status", help="Show waiter/run state")
    p_status.add_argument("--run-id", required=True)

    p_cancel = sub.add_parser("cancel", help="Cancel waiter")
    p_cancel.add_argument("--run-id", required=True)

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "status":
        st = load_state(args.run_id)
        print(json.dumps(st or {"run_id": args.run_id, "state": "none"}, indent=2))
        return
    if args.cmd == "cancel":
        cancel(args.run_id)
        return

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    manifest = load_manifest(manifest_path)
    entry = entry_for(manifest, args.run_id)
    entry["_manifest_path"] = str(manifest_path)
    os.environ["POSTTRAIN_MANIFEST"] = str(manifest_path)

    if args.cmd == "arm":
        arm_waiter(entry, poll_sec=args.poll_sec, force=args.force)
    elif args.cmd == "run":
        rc = run_posttrain(entry, force=args.force)
        sys.exit(rc)
    elif args.cmd == "wait":
        rc = cmd_wait(manifest_path, args.run_id, args.poll_sec)
        sys.exit(rc)


if __name__ == "__main__":
    main()
