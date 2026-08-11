#!/usr/bin/env python3
"""Unified B5 trainer: B5a + multi-seed MSE + gated KD Stage-B (2k).

Called by ``b5_train_20k.py resume``. Default ``--max-parallel 3`` keeps the
MSE lane 3-wide overnight. KD lane starts only after every MSE entry is done,
and runs sequentially (never mixed with MSE; IMPLEMENTATION §9).

KD Stage-B (after MSE): pixel λ=0/0.2 then VGG relu3 λ=0/0.01 @ ~2k updates.
Do not use pixel's λ=0.2 for VGG. Full 20k PECSR KD is superseded.

  python scripts/run_b5_train_20k.py --skip-done --max-parallel 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
if Path("/home/hyb/miniforge3/envs/cv_env/bin/python").exists():
    PYTHON = "/home/hyb/miniforge3/envs/cv_env/bin/python"

MANIFEST = PROJECT_ROOT / "results/exp_runs/b5_train_20k_manifest.json"
LOG_DIR = PROJECT_ROOT / "results/exp_runs/logs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=str(MANIFEST))
    p.add_argument("--skip-done", action="store_true")
    p.add_argument("--run-id", default=None)
    p.add_argument(
        "--max-parallel",
        type=int,
        default=3,
        help="Max concurrent MSE trains (default 3). KD always sequential.",
    )
    p.add_argument("--poll-sec", type=int, default=20)
    p.add_argument(
        "--mse-only",
        action="store_true",
        help="Do not start KD after MSE (leave KD for a later resume).",
    )
    return p.parse_args()


def load_manifest(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_done(entry: dict) -> bool:
    log_path = PROJECT_ROOT / "results/exp_runs" / entry["run_id"] / "train_log.jsonl"
    ckpt = PROJECT_ROOT / "results/exp_runs" / entry["run_id"] / "checkpoints/best.pt"
    if not log_path.exists() or not ckpt.exists():
        return False
    rows = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        return False
    return int(rows[-1]["epoch"]) >= int(entry["epochs"])


def train_script_for(entry: dict) -> Path:
    return PROJECT_ROOT / "scripts" / entry.get("train_script", "train_sepres_v2.py")


def pgrep(pattern: str) -> list[str]:
    try:
        out = subprocess.check_output(["pgrep", "-f", pattern], text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return []
    return [p for p in out.split() if p]


def _cmdline(pid: str) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")
    except OSError:
        return ""


def train_already_running(entry: dict) -> bool:
    cfg = entry["config"]
    patterns = [
        f"train_sepres_v2.py --config {cfg}",
        f"train_ecbsr.py --config {cfg}",
        f"train_mobile_srnet_kd.py --config {cfg}",
        f"{train_script_for(entry).name} --config {cfg}",
    ]
    for pat in patterns:
        for pid in pgrep(pat):
            cmd = _cmdline(pid)
            if "python" in cmd and "/bin/bash" not in cmd:
                return True
    return False


def build_cmd(entry: dict) -> list[str]:
    cfg = entry["config"]
    cmd = [PYTHON, str(train_script_for(entry)), "--config", cfg]
    ckpt_latest = PROJECT_ROOT / "results/exp_runs" / entry["run_id"] / "checkpoints/latest.pt"
    if ckpt_latest.exists() and not is_done(entry):
        cmd += ["--resume-from", str(ckpt_latest.relative_to(PROJECT_ROOT))]
    return cmd


def live_count(entries: list[dict]) -> int:
    return sum(1 for e in entries if not is_done(e) and train_already_running(e))


def launch(entry: dict, owned: dict[str, subprocess.Popen], logs: dict) -> None:
    run_id = entry["run_id"]
    if train_already_running(entry):
        print(f"[adopt] {run_id} already live — counted toward parallel cap")
        return
    cmd = build_cmd(entry)
    ckpt_latest = PROJECT_ROOT / "results/exp_runs" / run_id / "checkpoints/latest.pt"
    log_mode = "a" if ckpt_latest.exists() and not is_done(entry) else "w"
    log_path = LOG_DIR / f"train_{run_id}.log"
    handle = log_path.open(log_mode, encoding="utf-8")
    if log_mode == "a":
        handle.write(f"\n--- resume {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    print(f"[start] {run_id} lane={entry.get('lane')} variant={entry.get('variant')}")
    print(" ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT)
    owned[run_id] = proc
    logs[run_id] = handle


def reap(owned: dict[str, subprocess.Popen], logs: dict) -> None:
    finished = []
    for run_id, proc in owned.items():
        rc = proc.poll()
        if rc is None:
            continue
        finished.append(run_id)
        h = logs.pop(run_id, None)
        if h is not None:
            h.close()
        if rc != 0:
            print(f"[FAIL] {run_id} exit={rc} — see {LOG_DIR / f'train_{run_id}.log'}")
            for other, op in list(owned.items()):
                if other != run_id and op.poll() is None:
                    op.terminate()
            sys.exit(rc)
        print(f"[done] {run_id}")
    for run_id in finished:
        del owned[run_id]


def run_lane(
    entries: list[dict],
    *,
    max_parallel: int,
    poll_sec: int,
    skip_done: bool,
) -> None:
    pending: list[dict] = []
    for e in entries:
        if skip_done and is_done(e):
            print(f"[skip] {e['run_id']} already complete")
            continue
        pending.append(e)

    owned: dict[str, subprocess.Popen] = {}
    logs: dict = {}
    print(f"lane size={len(entries)} pending={len(pending)} max_parallel={max_parallel}")

    # Initial adopt / fill
    while True:
        reap(owned, logs)
        # Drop pending that finished externally
        pending = [e for e in pending if not is_done(e)]
        live = live_count(entries)
        # Launch until cap
        i = 0
        while i < len(pending) and live < max_parallel:
            e = pending[i]
            if train_already_running(e) or is_done(e):
                i += 1
                continue
            launch(e, owned, logs)
            pending.pop(i)
            live = live_count(entries)
        if not pending and live == 0 and not owned:
            break
        # Still waiting on live orphans or owned procs
        if not pending and live > 0 and not owned:
            # Orphans only — wait until they finish
            time.sleep(poll_sec)
            continue
        if not pending and not owned and live == 0:
            break
        time.sleep(poll_sec)


def main() -> None:
    args = parse_args()
    if args.max_parallel < 1:
        raise SystemExit("--max-parallel must be >= 1")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(Path(args.manifest))

    if args.run_id:
        manifest = [e for e in manifest if e["run_id"] == args.run_id]
        if not manifest:
            raise SystemExit(f"unknown run_id: {args.run_id}")

    mse = [e for e in manifest if e.get("lane", "mse") == "mse"]
    kd = [e for e in manifest if e.get("lane") == "kd"]

    print("B5 unified train — MSE lane")
    run_lane(mse, max_parallel=args.max_parallel, poll_sec=args.poll_sec, skip_done=args.skip_done)

    if args.mse_only:
        print("MSE lane complete (--mse-only: KD not started).")
        return

    if any(not is_done(e) for e in mse):
        print("MSE lane still incomplete — not starting KD.")
        sys.exit(1)

    print("B5 unified train — KD lane (sequential, after MSE)")
    run_lane(kd, max_parallel=1, poll_sec=args.poll_sec, skip_done=args.skip_done)
    print("B5 unified train complete (MSE + KD).")


if __name__ == "__main__":
    main()
