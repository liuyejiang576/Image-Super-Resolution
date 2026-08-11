#!/usr/bin/env python3
"""Measure solo vs 2-process KD throughput to choose sequential vs parallel Stage B.

Runs the same KD training step as fair-budget configs (bs=16, AMP, SwinIR teacher).
Compares solo steps/sec vs dual concurrent workers; recommends layout if slowdown > 1.8x.

Example:
  python scripts/probe_kd_parallel.py
  python scripts/probe_kd_parallel.py --timed-steps 20 --warmup-steps 5
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
WORKER = PROJECT_ROOT / "scripts/probe_kd_throughput_worker.py"
OUT_DIR = PROJECT_ROOT / "results/exp_runs/kd_parallel_probe"

# If dual per-job slowdown exceeds this, sequential wall clock wins for 2-run Stage B.
SLOWDOWN_BREAK_EVEN = 1.8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--timed-steps", type=int, default=30)
    p.add_argument("--slowdown-threshold", type=float, default=SLOWDOWN_BREAK_EVEN)
    return p.parse_args()


def run_solo(args: argparse.Namespace) -> dict:
    out = OUT_DIR / "solo.json"
    cmd = [
        PYTHON,
        str(WORKER),
        "--worker-id",
        "0",
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--warmup-steps",
        str(args.warmup_steps),
        "--timed-steps",
        str(args.timed_steps),
        "--out",
        str(out),
    ]
    print(f"[solo] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    return json.loads(out.read_text(encoding="utf-8"))


def run_dual(args: argparse.Namespace) -> list[dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outs = [OUT_DIR / "dual_0.json", OUT_DIR / "dual_1.json"]
    procs = []
    t0 = time.perf_counter()
    for wid, out in enumerate(outs):
        cmd = [
            PYTHON,
            str(WORKER),
            "--worker-id",
            str(wid),
            "--batch-size",
            str(args.batch_size),
            "--workers",
            str(args.workers),
            "--warmup-steps",
            str(args.warmup_steps),
            "--timed-steps",
            str(args.timed_steps),
            "--seed",
            "42",
            "--out",
            str(out),
        ]
        print(f"[dual] starting worker {wid}: {' '.join(cmd)}")
        procs.append(
            subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        )

    for wid, proc in enumerate(procs):
        stdout, _ = proc.communicate()
        if proc.returncode != 0:
            print(f"[dual] worker {wid} failed (exit {proc.returncode}):")
            print(stdout.decode("utf-8", errors="replace")[-2000:])
            raise subprocess.CalledProcessError(proc.returncode, cmd)
        print(f"[dual] worker {wid} done")

    wall_sec = time.perf_counter() - t0
    results = [json.loads(p.read_text(encoding="utf-8")) for p in outs]
    for r in results:
        r["dual_wall_sec"] = wall_sec
    return results


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
        used, total, util = [x.strip() for x in out.split(",")]
        return {"used_mib": used, "total_mib": total, "util_pct": util}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main() -> None:
    args = parse_args()
    if not __import__("torch").cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== KD throughput probe (solo vs 2 concurrent) ===")
    print(f"batch_size={args.batch_size} workers={args.workers} timed_steps={args.timed_steps}")

    solo = run_solo(args)
    if solo.get("oom"):
        print("Solo run OOM — cannot probe")
        sys.exit(1)

    dual = run_dual(args)
    if any(r.get("oom") for r in dual):
        print("Dual run OOM — use sequential for Stage B")
        verdict = "SEQUENTIAL (dual OOM)"
    else:
        solo_sps = solo["steps_per_sec"]
        dual_sps = [r["steps_per_sec"] for r in dual]
        dual_min = min(dual_sps)
        dual_avg = sum(dual_sps) / len(dual_sps)
        slowdown_min = solo_sps / dual_min if dual_min > 0 else float("inf")
        slowdown_avg = solo_sps / dual_avg if dual_avg > 0 else float("inf")

        # Wall clock for 2 runs: sequential = 2/solo_sps * steps; parallel = max(elapsed)
        steps_2k = 2000
        seq_wall_h = 2 * steps_2k / solo_sps / 3600
        par_wall_h = steps_2k / dual_min / 3600

        if slowdown_min >= args.slowdown_threshold:
            verdict = "SEQUENTIAL (dual per-job slowdown >= threshold)"
            layout = "sequential"
        else:
            verdict = "PARALLEL ok (2 concurrent jobs)"
            layout = "parallel"

        print()
        print("--- Results ---")
        print(f"Solo:     {solo_sps:.3f} steps/s  peak={solo['peak_mem_gb']:.2f} GB  "
              f"sec/step={solo['sec_per_step']:.3f}")
        for r in dual:
            print(
                f"Dual w{r['worker_id']}: {r['steps_per_sec']:.3f} steps/s  "
                f"peak={r['peak_mem_gb']:.2f} GB  sec/step={r['sec_per_step']:.3f}"
            )
        print(f"Slowdown (solo / slowest dual): {slowdown_min:.2f}x  (avg dual: {slowdown_avg:.2f}x)")
        print(f"Est. Stage B 2-run wall @ 2k updates: sequential ~{seq_wall_h:.1f}h  parallel ~{par_wall_h:.1f}h")
        print(f"Threshold: {args.slowdown_threshold}x")
        print(f"Verdict: {verdict}")

        payload = {
            "solo": solo,
            "dual": dual,
            "slowdown_min": slowdown_min,
            "slowdown_avg": slowdown_avg,
            "threshold": args.slowdown_threshold,
            "stage_b_2run_wall_h": {"sequential": seq_wall_h, "parallel": par_wall_h},
            "recommendation": layout,
            "verdict": verdict,
            "gpu_after": gpu_snapshot(),
        }
        out_path = OUT_DIR / "summary.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {out_path}")
        return

    payload = {"solo": solo, "dual": dual, "recommendation": "sequential", "verdict": verdict}
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
