#!/usr/bin/env python3
"""Run B5 multi-seed PECSR/ECBSR fair-budget training.

Called by b5_confirm_20k.py resume. Default trains all manifest entries
(s42 rows skip-done). Picks train_script from each manifest entry.

Parallelism (IMPLEMENTATION §9): default ``--max-parallel 2`` so the confirm
queue never collapses to 1-wide after B5a finishes. With B5a also running,
total MSE jobs ≈ 3 (probe-validated on this machine).

  python scripts/run_b5_confirm_20k.py --skip-done
  python scripts/run_b5_confirm_20k.py --skip-done --max-parallel 2
  python scripts/run_b5_confirm_20k.py --skip-done --run-id sepres_v2_c16n10_20k_s123
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

MANIFEST = PROJECT_ROOT / "results/exp_runs/b5_confirm_20k_manifest.json"
LOG_DIR = PROJECT_ROOT / "results/exp_runs/logs"
DEFAULT_RUN_IDS = [
    "sepres_v2_c16n10_20k",
    "sepres_v2_c16n10_20k_s123",
    "sepres_v2_c16n10_20k_s2026",
    "ecbsr_m10c16_20k",
    "ecbsr_m10c16_20k_s123",
    "ecbsr_m10c16_20k_s2026",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=str(MANIFEST))
    p.add_argument("--skip-done", action="store_true")
    p.add_argument("--run-id", default=None, help="Train one run_id.")
    p.add_argument(
        "--all-gated",
        action="store_true",
        help="Train every manifest entry (same as default).",
    )
    p.add_argument(
        "--max-parallel",
        type=int,
        default=2,
        help="Max concurrent MSE trains from this queue (default 2; never mix KD).",
    )
    p.add_argument(
        "--poll-sec",
        type=int,
        default=30,
        help="Seconds between slot reaping / refill checks.",
    )
    return p.parse_args()


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
    name = entry.get("train_script", "train_sepres_v2.py")
    return PROJECT_ROOT / "scripts" / name


def pgrep(pattern: str) -> list[str]:
    try:
        out = subprocess.check_output(["pgrep", "-f", pattern], text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return []
    return [p for p in out.split() if p]


def train_already_running(entry: dict) -> bool:
    """Skip launch if this config is already training (manual probe / other launcher)."""
    cfg = entry["config"]
    script = train_script_for(entry).name
    # Also match train_ecbsr when sepres wrapper runpy is used.
    patterns = [f"{script} --config {cfg}", f"train_ecbsr.py --config {cfg}", f"train_sepres_v2.py --config {cfg}"]
    for pat in patterns:
        if pgrep(pat):
            return True
    return False


def build_cmd(entry: dict) -> list[str]:
    cfg = entry["config"]
    train_script = train_script_for(entry)
    run_id = entry["run_id"]
    ckpt_latest = PROJECT_ROOT / "results/exp_runs" / run_id / "checkpoints/latest.pt"
    cmd = [PYTHON, str(train_script), "--config", cfg]
    if ckpt_latest.exists() and not is_done(entry):
        cmd += ["--resume-from", str(ckpt_latest.relative_to(PROJECT_ROOT))]
    return cmd


def main() -> None:
    args = parse_args()
    if args.max_parallel < 1:
        raise SystemExit("--max-parallel must be >= 1")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if args.run_id:
        entries = [e for e in manifest if e["run_id"] == args.run_id]
        if not entries:
            raise SystemExit(f"run_id not in manifest: {args.run_id}")
    elif args.all_gated:
        entries = manifest
    else:
        entries = [e for e in manifest if e["run_id"] in DEFAULT_RUN_IDS]
        if not entries:
            raise SystemExit(f"default run ids missing from manifest: {DEFAULT_RUN_IDS}")

    pending = []
    for entry in entries:
        run_id = entry["run_id"]
        if args.skip_done and is_done(entry):
            print(f"[skip] {run_id} already complete")
            continue
        if train_already_running(entry):
            print(f"[skip] {run_id} already has a live train process")
            continue
        pending.append(entry)

    print(
        f"B5 confirm multi-seed 20k — training "
        f"(max_parallel={args.max_parallel}, pending={len(pending)})"
    )

    # Single-job path stays simple / identical to pre-parallel behavior.
    if args.max_parallel == 1 or args.run_id:
        for entry in pending:
            run_id = entry["run_id"]
            cmd = build_cmd(entry)
            ckpt_latest = PROJECT_ROOT / "results/exp_runs" / run_id / "checkpoints/latest.pt"
            log_mode = "a" if ckpt_latest.exists() and not is_done(entry) else "w"
            log_path = LOG_DIR / f"train_{run_id}.log"
            print(f"\n[start] {run_id} variant={entry.get('variant')} script={train_script_for(entry).name}")
            print(" ".join(cmd))
            with log_path.open(log_mode, encoding="utf-8") as log:
                if log_mode == "a":
                    log.write(f"\n--- resume {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
            if proc.returncode != 0:
                print(f"[FAIL] {run_id} exit={proc.returncode} — see {log_path}")
                sys.exit(proc.returncode)
            print(f"[done] {run_id}")
        print("\nB5 confirm train complete for selected run(s). Next: mean±std table; KD only after MSE idle.")
        return

    # Parallel pool (MSE only).
    active: dict[str, subprocess.Popen] = {}
    log_handles: dict[str, object] = {}
    queue = list(pending)

    def reap() -> None:
        done_ids = []
        for run_id, proc in active.items():
            rc = proc.poll()
            if rc is None:
                continue
            done_ids.append(run_id)
            handle = log_handles.pop(run_id, None)
            if handle is not None:
                handle.close()
            if rc != 0:
                print(f"[FAIL] {run_id} exit={rc} — see {LOG_DIR / f'train_{run_id}.log'}")
                for other in list(active):
                    if other == run_id:
                        continue
                    active[other].terminate()
                sys.exit(rc)
            print(f"[done] {run_id}")
        for run_id in done_ids:
            del active[run_id]

    def launch_one(entry: dict) -> None:
        run_id = entry["run_id"]
        if train_already_running(entry):
            print(f"[skip] {run_id} became live before launch")
            return
        cmd = build_cmd(entry)
        ckpt_latest = PROJECT_ROOT / "results/exp_runs" / run_id / "checkpoints/latest.pt"
        log_mode = "a" if ckpt_latest.exists() and not is_done(entry) else "w"
        log_path = LOG_DIR / f"train_{run_id}.log"
        log = log_path.open(log_mode, encoding="utf-8")
        if log_mode == "a":
            log.write(f"\n--- resume {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        print(f"\n[start] {run_id} variant={entry.get('variant')} script={train_script_for(entry).name}")
        print(" ".join(cmd))
        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
        active[run_id] = proc
        log_handles[run_id] = log

    while queue or active:
        reap()
        while queue and len(active) < args.max_parallel:
            launch_one(queue.pop(0))
        if not queue and not active:
            break
        time.sleep(args.poll_sec)

    print("\nB5 confirm train complete for selected run(s). Next: mean±std table; KD only after MSE idle.")


if __name__ == "__main__":
    main()
