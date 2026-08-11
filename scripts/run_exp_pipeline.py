#!/usr/bin/env python3
"""Run full experiment + report pipeline unattended."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/hyb/miniforge3/envs/cv_env/bin/python"
LOG_DIR = PROJECT_ROOT / "results" / "exp_runs" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def run_step(name: str, cmd: list[str], required: bool = True) -> bool:
    log_path = LOG_DIR / f"{name}.log"
    print(f"\n=== {name} ===")
    print(" ".join(cmd))
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"FAILED {name} (see {log_path})")
        if required:
            return False
    else:
        print(f"OK {name}")
    return proc.returncode == 0


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from exp_run_utils import is_run_done, load_manifest  # noqa: E402


def train_from_manifest(manifest_path: Path) -> None:
    manifest = load_manifest() if manifest_path.name == "fair_budget_manifest.json" else json.loads(manifest_path.read_text())
    for entry in manifest:
        run_id = entry["run_id"]
        cfg = entry["config"]
        if is_run_done(entry):
            print(f"Skip completed {run_id}")
            continue
        if "fsrcnn" in run_id:
            script = "scripts/train_fsrcnn.py"
            cmd = [PYTHON, script, "--config", cfg]
        elif entry.get("lambda_kd") is not None or "kd" in run_id:
            script = "scripts/train_mobile_srnet_kd.py"
            cmd = [PYTHON, script, "--config", cfg]
            if entry.get("lambda_kd") is not None:
                cmd += ["--lambda-kd", str(entry["lambda_kd"])]
        else:
            script = "scripts/train_mobile_srnet.py"
            cmd = [PYTHON, script, "--config", cfg]
        run_step(f"train_{run_id}", cmd, required=False)


def main() -> None:
    steps = [
        ("snapshot_baselines", [PYTHON, "scripts/snapshot_baselines.py"], True),
        ("gpu_probe", [PYTHON, "scripts/gpu_probe.py", "--probe-steps", "30"], True),
        ("generate_exp_configs", [PYTHON, "scripts/generate_exp_configs.py"], True),
    ]
    for name, cmd, req in steps:
        if not run_step(name, cmd, req):
            sys.exit(1)

    train_from_manifest(PROJECT_ROOT / "results/exp_runs/fair_budget_manifest.json")

    # Figures / latency: use report/plot (lab copies live under scripts/_inactive/lab_plot_dup/)
    report_plot = PROJECT_ROOT.parent / "report" / "plot"
    report_steps = [
        (
            "audit_latency_180",
            [PYTHON, str(report_plot / "audit_latency.py"), "--lr-h", "180", "--lr-w", "180"],
            False,
        ),
        (
            "audit_latency_720p",
            [
                PYTHON,
                str(report_plot / "audit_latency.py"),
                "--lr-h",
                "180",
                "--lr-w",
                "320",
                "--output",
                str(PROJECT_ROOT / "results/latency_audit/latency_audit_320x180.json"),
            ],
            False,
        ),
        # KD per-image: optional regen only
        (
            "kd_per_image",
            [PYTHON, "scripts/_inactive/kd_diag/kd_per_image_analysis.py"],
            False,
        ),
    ]
    for name, cmd, req in report_steps:
        run_step(name, cmd, req)

    run_step("eval_exp_runs", [PYTHON, "scripts/eval_exp_runs.py"], required=False)

    print("\nPipeline finished. See results/exp_runs/logs/")


if __name__ == "__main__":
    main()
