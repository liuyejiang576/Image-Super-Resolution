#!/usr/bin/env python3
"""Evaluate all completed experiment runs and update fair_budget summary."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/hyb/miniforge3/envs/cv_env/bin/python"


def main() -> None:
    manifest_path = PROJECT_ROOT / "results/exp_runs/fair_budget_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    summary = {}
    for entry in manifest:
        run_id = entry["run_id"]
        ckpt = PROJECT_ROOT / "results/exp_runs" / run_id / "checkpoints/best.pt"
        if not ckpt.exists():
            summary[run_id] = {"status": "pending"}
            continue
        out_json = PROJECT_ROOT / "results/exp_runs" / run_id / "benchmark_metrics.json"
        cmd = [
            PYTHON, "scripts/eval_sr.py",
            "--checkpoint", str(ckpt.relative_to(PROJECT_ROOT)),
            "--save-json", str(out_json.relative_to(PROJECT_ROOT)),
            "--compute-lpips",
        ]
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
        if out_json.exists():
            with out_json.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
            psnrs = [v["psnr"] for v in metrics.values() if isinstance(v, dict)]
            summary[run_id] = {
                "status": "done",
                "avg_psnr": sum(psnrs) / len(psnrs) if psnrs else None,
                "metrics": metrics,
            }
        else:
            summary[run_id] = {"status": "eval_failed"}

    out_path = PROJECT_ROOT / "results/exp_runs/fair_budget_runs.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_plot = PROJECT_ROOT.parent / "report" / "plot" / "plot_fair_budget.py"
    if report_plot.exists():
        subprocess.run(
            [PYTHON, str(report_plot)],
            cwd=str(report_plot.parent),
            check=False,
        )
    print(f"Wrote {out_path}")
    print("Promote to report: copy fair_budget_runs.json → ../report/assets/metrics/ (see ../report/SYNC.md)")


if __name__ == "__main__":
    main()
