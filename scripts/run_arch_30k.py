#!/usr/bin/env python3
"""Run fair-budget 30k continuation for FSRCNN, Base, and Plus (parallel MSE).

Called by arch_30k.py resume. Do not start manually unless debugging.

  python scripts/run_arch_30k.py --skip-done
  python scripts/run_arch_30k.py --run-id mobile_srnet_plus_30k
  python scripts/run_arch_30k.py --sequential --skip-done   # fallback

MSE-only: never launches KD / SwinIR-teacher jobs (KD parallel probe showed ~2.9x
slowdown). Writes observation snapshots to results/exp_runs/arch_30k_observe.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
if Path("/home/hyb/miniforge3/envs/cv_env/bin/python").exists():
    PYTHON = "/home/hyb/miniforge3/envs/cv_env/bin/python"

MANIFEST = PROJECT_ROOT / "results/exp_runs/arch_30k_manifest.json"
LOG_DIR = PROJECT_ROOT / "results/exp_runs/logs"
EXP_RUNS = PROJECT_ROOT / "results/exp_runs"
OBSERVE_PATH = PROJECT_ROOT / "results/exp_runs/arch_30k_observe.json"

# Solo epoch/min from gpu_probe_recommendations (steps/s, batch) @ DIV2K 800 imgs.
SOLO_EP_PER_MIN = {
    "fsrcnn": 1.78,   # bs=8, 2.97 steps/s
    "base": 3.10,     # bs=24, 1.71 steps/s (40ch)
    "plus": 3.10,     # same batch/recipe as base
}

WARN_LOW_UTIL_PCT = 25
WARN_LOW_UTIL_POLLS = 3
WARN_HIGH_VRAM_MIB = 6500
WARN_SLOWDOWN_RATIO = 0.55  # parallel ep/min < 55% of solo → warn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=str(MANIFEST))
    p.add_argument("--skip-done", action="store_true")
    p.add_argument("--run-id", default=None, help="Run a single manifest entry")
    p.add_argument("--max-parallel", type=int, default=3, help="Max concurrent MSE jobs")
    p.add_argument("--poll-sec", type=int, default=30, help="Observation poll interval")
    p.add_argument("--sequential", action="store_true", help="One job at a time (legacy)")
    return p.parse_args()


def load_manifest(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def target_epochs(entry: dict) -> int:
    cfg_path = PROJECT_ROOT / entry["config"]
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return int(cfg["train"]["epochs"])


def read_train_log(run_id: str) -> list[dict]:
    log_path = EXP_RUNS / run_id / "train_log.jsonl"
    if not log_path.exists():
        return []
    rows = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def is_done(entry: dict) -> bool:
    rows = read_train_log(entry["run_id"])
    if not rows:
        return False
    ckpt = EXP_RUNS / entry["run_id"] / "checkpoints/best.pt"
    if not ckpt.exists():
        return False
    return int(rows[-1]["epoch"]) >= target_epochs(entry)


def train_script(entry: dict) -> str:
    if entry["model"] == "fsrcnn":
        return "scripts/train_fsrcnn.py"
    if entry["model"] == "mobile_srnet":
        return "scripts/train_mobile_srnet.py"
    raise ValueError(f"Unknown model type: {entry['model']}")


def assert_mse_only(entry: dict) -> None:
    if entry.get("lambda_kd") is not None or "kd" in entry["run_id"]:
        raise SystemExit(f"Refusing parallel launch for KD run: {entry['run_id']}")


def bootstrap_resume_path(entry: dict) -> Path | None:
    """Pick resume checkpoint: 30k latest if partial, else 20k latest on first start."""
    run_id = entry["run_id"]
    source_id = entry["resume_from_run_id"]
    run_dir = EXP_RUNS / run_id
    ckpt_30k = run_dir / "checkpoints/latest.pt"
    ckpt_20k = EXP_RUNS / source_id / "checkpoints/latest.pt"
    log_30k = run_dir / "train_log.jsonl"
    log_20k = EXP_RUNS / source_id / "train_log.jsonl"

    if is_done(entry):
        return None
    if ckpt_30k.exists():
        return ckpt_30k
    if not ckpt_20k.exists():
        raise SystemExit(f"Missing 20k checkpoint for bootstrap: {ckpt_20k}")

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    if not log_30k.exists() and log_20k.exists():
        shutil.copy2(log_20k, log_30k)
        print(f"[bootstrap] copied 20k log → {log_30k.relative_to(PROJECT_ROOT)}")
    print(f"[bootstrap] {run_id}: resume from {ckpt_20k.relative_to(PROJECT_ROOT)}")
    return ckpt_20k


def running_run_ids(manifest: list[dict]) -> set[str]:
    try:
        proc = subprocess.check_output(["pgrep", "-af", "scripts/train_"], text=True)
    except subprocess.CalledProcessError:
        return set()
    active: set[str] = set()
    for entry in manifest:
        cfg = entry["config"]
        if cfg in proc:
            active.add(entry["run_id"])
    return active


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
    return {"used_mib": int(parts[0]), "total_mib": int(parts[1]), "util_pct": int(parts[2])}


def variant_key(entry: dict) -> str:
    v = entry.get("variant", entry["model"])
    if v == "fsrcnn":
        return "fsrcnn"
    if v == "plus":
        return "plus"
    return "base"


def build_cmd(entry: dict, resume_path: Path) -> list[str]:
    script = train_script(entry)
    cfg = entry["config"]
    return [
        PYTHON,
        str(PROJECT_ROOT / script),
        "--config",
        cfg,
        "--resume-from",
        str(resume_path.relative_to(PROJECT_ROOT)),
    ]


def launch_job(entry: dict, resume_path: Path) -> subprocess.Popen:
    run_id = entry["run_id"]
    cmd = build_cmd(entry, resume_path)
    log_path = LOG_DIR / f"train_{run_id}.log"
    log_mode = "a" if log_path.exists() else "w"
    print(f"[start] {run_id} ({entry.get('variant', entry['model'])}): {' '.join(cmd)}")
    log = log_path.open(log_mode, encoding="utf-8")
    if log_mode == "a":
        log.write(f"\n--- parallel resume {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    return subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
    )


def ep_per_min(history: deque, now: float) -> float | None:
    """Epochs/min over the last ~5 minutes of samples."""
    if len(history) < 2:
        return None
    t0, e0 = history[0]
    t1, e1 = history[-1]
    dt_min = (t1 - t0) / 60.0
    if dt_min <= 0:
        return None
    return (e1 - e0) / dt_min


def write_observe(
    *,
    layout: str,
    max_parallel: int,
    active: dict[str, subprocess.Popen],
    manifest: list[dict],
    epoch_hist: dict[str, deque],
    poll: int,
    low_util_streak: int,
    warnings: list[str],
) -> None:
    gpu = gpu_snapshot()
    running = sorted(set(active.keys()) | running_run_ids(manifest))
    jobs: dict[str, dict] = {}
    for entry in manifest:
        rid = entry["run_id"]
        rows = read_train_log(rid)
        epoch = int(rows[-1]["epoch"]) if rows else 0
        vk = variant_key(entry)
        rate = ep_per_min(epoch_hist.get(rid, deque()), time.time())
        solo = SOLO_EP_PER_MIN.get(vk)
        slowdown = None
        if rate is not None and solo and solo > 0 and rid in running:
            slowdown = rate / solo
        jobs[rid] = {
            "variant": entry.get("variant", entry["model"]),
            "epoch": epoch,
            "target_epochs": target_epochs(entry),
            "running": rid in running,
            "ep_per_min": round(rate, 3) if rate is not None else None,
            "solo_ep_per_min": solo,
            "slowdown_vs_solo": round(slowdown, 3) if slowdown is not None else None,
        }

    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "layout": layout,
        "max_parallel": max_parallel,
        "poll": poll,
        "running": running,
        "launcher_active": sorted(active.keys()),
        "gpu": gpu,
        "low_util_streak": low_util_streak,
        "warnings": warnings,
        "jobs": jobs,
        "notes": (
            "MSE-only parallel. KD probe showed ~2.9x per-job slowdown — never mix KD here. "
            "If warnings persist, pause and retry with --sequential or lower num_workers."
        ),
    }
    OBSERVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    OBSERVE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_sequential(entries: list[dict]) -> None:
    print("Architecture 30k continuation — sequential (FSRCNN → Base → Plus)\n")
    for entry in entries:
        run_id = entry["run_id"]
        resume_path = bootstrap_resume_path(entry)
        if resume_path is None:
            print(f"[skip] {run_id} already complete")
            continue
        proc = launch_job(entry, resume_path)
        code = proc.wait()
        if code != 0:
            print(f"[FAIL] {run_id} exit={code}")
            sys.exit(code)
        print(f"[done] {run_id}")
    print("\n30k continuation complete.")


def run_parallel(entries: list[dict], max_parallel: int, poll_sec: int) -> None:
    print(f"Architecture 30k continuation — parallel MSE (max {max_parallel} jobs)")
    print("(20k + ~10k low-LR fine-tuning; observe via arch_30k.py watch)\n")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        assert_mse_only(entry)

    active: dict[str, subprocess.Popen] = {}
    epoch_hist: dict[str, deque] = {e["run_id"]: deque(maxlen=12) for e in entries}
    low_util_streak = 0
    poll = 0

    def refresh_active() -> None:
        dead = [rid for rid, p in active.items() if p.poll() is not None]
        for rid in dead:
            code = active[rid].returncode
            print(f"[parallel] finished {rid} exit={code}")
            if code != 0:
                print(f"[FAIL] {rid} — see {LOG_DIR / f'train_{rid}.log'}")
                for p in active.values():
                    if p.poll() is None:
                        p.terminate()
                sys.exit(code)
            del active[rid]

    def pending_entries() -> list[dict]:
        out = []
        for entry in entries:
            rid = entry["run_id"]
            if is_done(entry):
                continue
            if rid in active or rid in running_run_ids(entries):
                continue
            out.append(entry)
        return out

    while True:
        refresh_active()
        poll += 1

        external = running_run_ids(entries) - set(active.keys())
        slots = max_parallel - len(active) - len(external)
        launched = 0
        for entry in pending_entries():
            if slots <= 0:
                break
            resume_path = bootstrap_resume_path(entry)
            if resume_path is None:
                continue
            rid = entry["run_id"]
            active[rid] = launch_job(entry, resume_path)
            slots -= 1
            launched += 1

        now = time.time()
        warnings: list[str] = []
        for entry in entries:
            rid = entry["run_id"]
            rows = read_train_log(rid)
            epoch = int(rows[-1]["epoch"]) if rows else 0
            epoch_hist.setdefault(rid, deque(maxlen=12)).append((now, epoch))

        gpu = gpu_snapshot()
        running_count = len(set(active.keys()) | running_run_ids(entries))
        if gpu and running_count >= 2:
            if gpu["util_pct"] < WARN_LOW_UTIL_PCT:
                low_util_streak += 1
            else:
                low_util_streak = 0
            if low_util_streak >= WARN_LOW_UTIL_POLLS:
                warnings.append(
                    f"Low GPU util ({gpu['util_pct']}%) with {running_count} jobs "
                    f"for {low_util_streak} polls — CPU/data-loader bottleneck?"
                )
        else:
            low_util_streak = 0

        if gpu and gpu["used_mib"] >= WARN_HIGH_VRAM_MIB:
            warnings.append(f"High VRAM ({gpu['used_mib']} MiB) — reduce --max-parallel")

        for entry in entries:
            rid = entry["run_id"]
            if rid not in (set(active.keys()) | running_run_ids(entries)):
                continue
            hist = epoch_hist.get(rid, deque())
            if len(hist) < 3 or poll < 4:
                continue
            rate = ep_per_min(hist, now)
            solo = SOLO_EP_PER_MIN.get(variant_key(entry))
            if rate is not None and rate > 0.05 and solo and rate < solo * WARN_SLOWDOWN_RATIO:
                warnings.append(
                    f"{rid}: {rate:.2f} ep/min vs solo ~{solo:.2f} "
                    f"({100 * rate / solo:.0f}% throughput)"
                )

        write_observe(
            layout="parallel",
            max_parallel=max_parallel,
            active=active,
            manifest=entries,
            epoch_hist=epoch_hist,
            poll=poll,
            low_util_streak=low_util_streak,
            warnings=warnings,
        )

        if all(is_done(e) for e in entries):
            print("[parallel] all 30k runs complete")
            break

        active_all = sorted(set(active.keys()) | external)
        if launched == 0 and active_all:
            print(f"[parallel] poll={poll} active={active_all} gpu={gpu}")
        elif launched == 0 and not active_all:
            pending = [e["run_id"] for e in pending_entries()]
            if pending:
                print(f"[parallel] waiting to launch: {pending}")
            else:
                print("[parallel] nothing running but not all done — rechecking")

        time.sleep(poll_sec)

    print("\n30k continuation complete.")
    print("Next: eval_sr.py on 30k checkpoints, update report/assets/metrics/model_summary.json")


def main() -> None:
    args = parse_args()
    manifest = load_manifest(Path(args.manifest))

    entries = manifest
    if args.run_id:
        entries = [e for e in manifest if e["run_id"] == args.run_id]
        if not entries:
            raise SystemExit(f"run_id not in manifest: {args.run_id}")

    if args.skip_done:
        entries = [e for e in entries if not is_done(e)]
        if not entries:
            print("All requested runs already complete.")
            return

    if args.sequential or args.max_parallel <= 1:
        run_sequential(entries)
    else:
        run_parallel(entries, args.max_parallel, args.poll_sec)


if __name__ == "__main__":
    main()
