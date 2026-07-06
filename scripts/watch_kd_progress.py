#!/usr/bin/env python3
"""Rich progress viewer for the KD lambda-sweep runs.

Per-run and overall: task, epoch x/x, val PSNR, time spent, ETA, est. total, progress %.

  one-shot : python scripts/watch_kd_progress.py
  live      : python scripts/watch_kd_progress.py --watch 60
  json      : python scripts/watch_kd_progress.py --json
  other runs: python scripts/watch_kd_progress.py --runs kd05,kd10,kd20

"spent"  = sum of per-epoch elapsed_sec (training compute time, excludes pauses).
"ETA"    = mean(last 5 epochs' elapsed_sec) * (target_epochs - current_epoch).
"est.tot"= spent + ETA.
Overall progress % = mean of per-run progress %. Sweep ETA = max per-run ETA (slowest).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

DEFAULT_RUNS = ["kd05", "kd10", "kd20"]


def run_id(name: str) -> str:
    return f"mobile_srnet_{name}_10k"


def cfg_path(name: str) -> Path:
    return Path(f"configs/_inactive/exp/mobile_srnet_{name}_10k.yaml")


def log_path(name: str) -> Path:
    return Path(f"results/exp_runs/{run_id(name)}/train_log.jsonl")


def target_epochs(name: str) -> int:
    cfg = cfg_path(name)
    try:
        import yaml  # type: ignore

        return int(yaml.safe_load(cfg.read_text(encoding="utf-8"))["train"]["epochs"])
    except Exception:
        m = re.search(r"(?m)^\s*epochs:\s*(\d+)", cfg.read_text(encoding="utf-8"))
        return int(m.group(1)) if m else 200


def read_rows(name: str) -> list[dict]:
    f = log_path(name)
    if not f.exists():
        return []
    rows: list[dict] = []
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def running_pids(name: str) -> list[str]:
    cfg = str(cfg_path(name))
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", f"train_mobile_srnet_kd.py --config {cfg}"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return []
    return [p for p in out.split() if p]


def gpu_snapshot() -> dict | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    first = out.splitlines()[0] if out else ""
    parts = [p.strip() for p in first.split(",")]
    if len(parts) != 3:
        return None
    return {"used_mib": parts[0], "total_mib": parts[1], "util_pct": parts[2]}


def fmt_dur(s: float | None) -> str:
    if s is None:
        return "—"
    s = int(max(0.0, s))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{ s % 60:02d}s"
    if s < 86400:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    return f"{s // 86400}d{(s % 86400) // 3600}h"


def fmt_pct(p: float) -> str:
    return f"{p:.1f}%"


def _is_nan(x: float) -> bool:
    return x != x


def state_of(name: str, rows: list[dict], target: int) -> str:
    if running_pids(name):
        return "running"
    if rows and int(rows[-1]["epoch"]) >= target:
        return "done"
    if rows:
        return "paused"
    return "pending"


def run_info(name: str) -> dict:
    rows = read_rows(name)
    target = target_epochs(name)
    if rows:
        last = rows[-1]
        epoch = int(last["epoch"])
        step = int(last.get("global_step", 0))
        psnr = float(last.get("val_psnr", float("nan")))
        best = max((float(r.get("val_psnr", -999.0)) for r in rows), default=float("nan"))
        spent = sum(float(r.get("elapsed_sec", 0.0)) for r in rows)
        recent = [float(r["elapsed_sec"]) for r in rows if "elapsed_sec" in r][-5:]
        avg_ep = mean(recent) if recent else (spent / epoch if epoch else 0.0)
        eta = avg_ep * max(0, target - epoch)
    else:
        epoch, step, psnr, best, spent, avg_ep, eta = 0, 0, float("nan"), float("nan"), 0.0, 0.0, 0.0
    progress = 100.0 * epoch / target if target else 0.0
    return {
        "name": name,
        "run_id": run_id(name),
        "state": state_of(name, rows, target),
        "epoch": epoch,
        "target": target,
        "step": step,
        "psnr": psnr,
        "best": best,
        "spent_sec": spent,
        "avg_epoch_sec": avg_ep,
        "eta_sec": eta,
        "progress_pct": progress,
    }


def build(runs: list[str]) -> list[dict]:
    return [run_info(n) for n in runs]


def render(infos: list[dict], json_out: bool = False) -> None:
    if json_out:
        print(json.dumps(infos, indent=2))
        return
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    running = [i for i in infos if i["state"] == "running"]
    done = [i for i in infos if i["state"] == "done"]
    paused = [i for i in infos if i["state"] == "paused"]
    min_epoch = min((i["epoch"] for i in infos), default=0)
    max_target = max((i["target"] for i in infos), default=0)
    overall_pct = mean(i["progress_pct"] for i in infos) if infos else 0.0
    if running:
        sweep_state = "running"
    elif infos and len(done) == len(infos):
        sweep_state = "done"
    elif paused:
        sweep_state = "paused"
    else:
        sweep_state = "idle"
    gpu = gpu_snapshot()

    lines: list[str] = []
    lines.append(f"KD lambda-sweep progress  —  {now}")
    lines.append(
        f"Sweep: {len(infos)} runs · {min_epoch}/{max_target} epochs (min) · "
        f"{fmt_pct(overall_pct)} overall · {sweep_state}"
    )
    if gpu:
        lines.append(
            f"GPU:   {gpu['used_mib']} / {gpu['total_mib']} MiB · {gpu['util_pct']}% util"
        )
    lines.append("")
    hdr = (
        f"{'run':<11} {'state':<8} {'epoch':>9} {'val_psnr':>9} {'best':>9} "
        f"{'spent':>9} {'ETA':>9} {'est.tot':>9} {'prog':>6}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for i in infos:
        psnr = f"{i['psnr']:.3f}" if not _is_nan(i["psnr"]) else "—"
        best = f"{i['best']:.3f}" if not _is_nan(i["best"]) else "—"
        ep_str = f"{i['epoch']}/{i['target']}"
        lines.append(
            f"{i['name']:<11} {i['state']:<8} {ep_str:>9} {psnr:>9} {best:>9} "
            f"{fmt_dur(i['spent_sec']):>9} {fmt_dur(i['eta_sec']):>9} "
            f"{fmt_dur(i['spent_sec'] + i['eta_sec']):>9} {fmt_pct(i['progress_pct']):>6}"
        )
    lines.append("")
    if infos:
        slowest = max(infos, key=lambda i: i["eta_sec"])
        if slowest["eta_sec"] > 0 and slowest["state"] != "done":
            eta_dt = datetime.fromtimestamp(time.time() + slowest["eta_sec"]).strftime(
                "%Y-%m-%d %H:%M"
            )
            lines.append(
                f"ETA to sweep completion (slowest = {slowest['name']}): "
                f"~{fmt_dur(slowest['eta_sec'])}  →  {eta_dt}"
            )
        else:
            lines.append("All runs reached target.")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="KD lambda-sweep rich progress viewer")
    ap.add_argument(
        "--runs",
        default=",".join(DEFAULT_RUNS),
        help="comma-separated run names (default: kd05,kd10,kd20)",
    )
    ap.add_argument(
        "--watch",
        type=int,
        default=0,
        metavar="SEC",
        help="refresh every SEC seconds (live, clear-screen); exits when all done",
    )
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]

    if args.watch > 0:
        try:
            while True:
                infos = build(runs)
                if not args.json:
                    sys.stdout.write("\033[2J\033[H")
                render(infos, json_out=args.json)
                sys.stdout.flush()
                if all(i["state"] == "done" for i in infos):
                    break
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print()
            sys.stdout.flush()
    else:
        render(build(runs), json_out=args.json)


if __name__ == "__main__":
    main()
