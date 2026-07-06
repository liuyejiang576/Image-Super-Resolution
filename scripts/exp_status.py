#!/usr/bin/env python3
"""Print fair-budget experiment status and optionally write results/exp_runs/status.json."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from exp_run_utils import (  # noqa: E402
    EXP_RUNS_DIR,
    all_run_statuses,
    gpu_snapshot,
    pending_run_ids,
    running_run_ids,
    summary_dict,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fair-budget experiment monitor")
    p.add_argument("--json", action="store_true", help="Print JSON summary to stdout")
    p.add_argument("--pending-count", action="store_true", help="Print number of incomplete runs")
    p.add_argument(
        "--write",
        default=str(EXP_RUNS_DIR / "status.json"),
        help="Write status JSON (default: results/exp_runs/status.json)",
    )
    p.add_argument("--no-write", action="store_true", help="Do not write status JSON file")
    p.add_argument("--watch", type=int, default=0, metavar="SEC", help="Poll every SEC seconds")
    return p.parse_args()


def _fmt_lambda(val) -> str:
    if val is None:
        return ""
    return f" λ={val}"


def print_table(statuses: list[dict]) -> None:
    gpu = gpu_snapshot()
    done = sum(1 for s in statuses if s["done"])
    running = sorted(running_run_ids())
    pending = pending_run_ids()

    print(f"Fair-budget runs: {done}/{len(statuses)} complete")
    if running:
        print(f"Active trainers: {', '.join(running)}")
    if pending and len(pending) != len(running):
        not_running = [r for r in pending if r not in running]
        if not_running:
            print(f"Queued / paused: {', '.join(not_running)}")
    if gpu:
        print(
            "GPU: "
            f"{gpu['memory_used_mib']} / {gpu['memory_total_mib']} MiB, "
            f"{gpu['utilization_pct']}% util"
        )
    print()
    print(f"{'run_id':<28} {'state':<8} {'epoch':>9} {'val_psnr':>9} {'best':>9} {'step':>7}")
    print("-" * 78)
    for s in statuses:
        lam = _fmt_lambda(s.get("lambda_kd"))
        label = f"{s['run_id']}{lam}"
        epoch = f"{s['epoch'] or 0}/{s['target_epochs']}"
        val = f"{s['val_psnr']:.3f}" if s.get("val_psnr") is not None else "—"
        best = f"{s['best_val_psnr']:.3f}" if s.get("best_val_psnr") is not None else "—"
        step = str(s.get("global_step") or "—")
        print(f"{label:<28} {s['state']:<8} {epoch:>9} {val:>9} {best:>9} {step:>7}")
    print()
    print("Logs: results/exp_runs/<run_id>/train_log.jsonl")
    print("Trainer stdout: results/exp_runs/logs/train_<run_id>.log")
    print("Launcher: results/exp_runs/parallel.log")


def emit_once(args: argparse.Namespace) -> dict:
    summary = summary_dict()
    summary["updated_at"] = datetime.now(timezone.utc).isoformat()

    if args.pending_count:
        print(len(summary["pending"]))
    elif args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_table(summary["runs"])

    if not args.no_write and args.write:
        out = Path(args.write)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    args = parse_args()
    if args.watch > 0:
        while True:
            if not args.json and not args.pending_count:
                print(f"\n=== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
            emit_once(args)
            pending = pending_run_ids()
            if not pending:
                break
            time.sleep(args.watch)
    else:
        emit_once(args)


if __name__ == "__main__":
    main()
