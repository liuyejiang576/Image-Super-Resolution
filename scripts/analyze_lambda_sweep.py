#!/usr/bin/env python3
"""Benchmark the lambda-sweep runs and assemble a 5-point sweep table.

Combines the new runs (kd05_10k, kd10_10k, kd20_10k) with the existing
kd0_10k / kd02_10k results from fair_budget_runs.json into a single
lambda vs avg_psnr / avg_lpips / val_psnr table.

Run AFTER training for kd05/kd10/kd20 has completed (epoch 200).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/hyb/miniforge3/envs/cv_env/bin/python"

# (run_id, lambda) for the 5-point sweep at 10k updates.
SWEEP = [
    ("mobile_srnet_kd0_10k", 0.0),
    ("mobile_srnet_kd02_10k", 0.2),
    ("mobile_srnet_kd05_10k", 0.5),
    ("mobile_srnet_kd10_10k", 1.0),
    ("mobile_srnet_kd20_10k", 2.0),
]
NEW_RUNS = {"mobile_srnet_kd05_10k", "mobile_srnet_kd10_10k", "mobile_srnet_kd20_10k"}


def final_val_psnr(run_id: str) -> float | None:
    log = PROJECT_ROOT / "results/exp_runs" / run_id / "train_log.jsonl"
    if not log.exists():
        return None
    last = None
    for line in log.open():
        last = json.loads(line)
    return last["val_psnr"] if last else None


def benchmark_run(run_id: str) -> dict:
    ckpt = PROJECT_ROOT / "results/exp_runs" / run_id / "checkpoints/best.pt"
    out_json = PROJECT_ROOT / "results/exp_runs" / run_id / "benchmark_metrics.json"
    if not ckpt.exists():
        return {"status": "no_checkpoint"}
    cmd = [
        PYTHON, "scripts/eval_sr.py",
        "--checkpoint", str(ckpt.relative_to(PROJECT_ROOT)),
        "--save-json", str(out_json.relative_to(PROJECT_ROOT)),
        "--compute-lpips",
    ]
    print(f"  benchmarking {run_id} ...")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
    if not out_json.exists():
        return {"status": "eval_failed"}
    with out_json.open() as f:
        metrics = json.load(f)
    psnrs = [v["psnr"] for v in metrics.values() if isinstance(v, dict)]
    lpips = [v.get("lpips") for v in metrics.values() if isinstance(v, dict)]
    lpips = [x for x in lpips if x is not None]
    return {
        "status": "done",
        "avg_psnr": sum(psnrs) / len(psnrs) if psnrs else None,
        "avg_lpips": sum(lpips) / len(lpips) if lpips else None,
        "metrics": metrics,
    }


def main() -> None:
    # Load existing fair_budget_runs.json for the already-benchmarked runs.
    fb_path = PROJECT_ROOT / "results/exp_runs/fair_budget_runs.json"
    existing = json.load(fb_path.open()) if fb_path.exists() else {}

    results = []
    for run_id, lam in SWEEP:
        print(f"[lambda={lam}] run={run_id}")
        if run_id in NEW_RUNS:
            bench = benchmark_run(run_id)
        else:
            # Reuse existing benchmark from fair_budget_runs.json.
            entry = existing.get(run_id, {})
            bench = {
                "status": entry.get("status", "missing"),
                "avg_psnr": entry.get("avg_psnr"),
                "avg_lpips": _avg_lpips_from_metrics(entry.get("metrics", {})),
                "metrics": entry.get("metrics", {}),
            }
        val_psnr = final_val_psnr(run_id)
        results.append({
            "run_id": run_id,
            "lambda_kd": lam,
            "avg_psnr": bench.get("avg_psnr"),
            "avg_lpips": bench.get("avg_lpips"),
            "val_psnr_div2k": val_psnr,
            "status": bench.get("status"),
        })

    out = {"sweep_10k": results}
    out_path = PROJECT_ROOT / "results/exp_runs/lambda_sweep_summary.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)

    print("\n=== 10k lambda-sweep (DIV2K benchmark avg) ===")
    print(f"{'lambda':>7} {'avg_psnr':>10} {'avg_lpips':>10} {'val_psnr':>10}  run_id")
    for r in results:
        ps = f"{r['avg_psnr']:.4f}" if r['avg_psnr'] is not None else "  —  "
        lp = f"{r['avg_lpips']:.4f}" if r['avg_lpips'] is not None else "  —  "
        vp = f"{r['val_psnr_div2k']:.4f}" if r['val_psnr_div2k'] is not None else "  —  "
        print(f"{r['lambda_kd']:>7} {ps:>10} {lp:>10} {vp:>10}  {r['run_id']}")
    print(f"\nWrote {out_path}")


def _avg_lpips_from_metrics(metrics: dict) -> float | None:
    lpips = [v.get("lpips") for v in metrics.values() if isinstance(v, dict)]
    lpips = [x for x in lpips if x is not None]
    return sum(lpips) / len(lpips) if lpips else None


if __name__ == "__main__":
    main()
