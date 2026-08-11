#!/usr/bin/env python3
"""Unified B5 control: B5a + multi-seed MSE + gated KD Stage-B — one watch.

  python scripts/b5_train_20k.py pause [--dry-run]
  python scripts/b5_train_20k.py resume [--dry-run]   # --max-parallel 3
  python scripts/b5_train_20k.py watch [--interval 60]

KD after all MSE: pixel λ=0/0.2 @ 2k, then VGG relu3 λ=0/0.01 @ 2k (method-local λ).
See IMPLEMENTATION.md §1 / §14.
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
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

PY = sys.executable
if Path("/home/hyb/miniforge3/envs/cv_env/bin/python").exists():
    PY = "/home/hyb/miniforge3/envs/cv_env/bin/python"

MANIFEST = ROOT / "results/exp_runs/b5_train_20k_manifest.json"
PAUSE_STATE = ROOT / "results/exp_runs/b5_train_20k_paused.json"
LOG_DIR = ROOT / "results/exp_runs/logs"
DEFAULT_MAX_PARALLEL = 3


def load_manifest() -> list[dict]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def manifest_by_id() -> dict[str, dict]:
    return {e["run_id"]: e for e in load_manifest()}


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


def _python_pids(pattern: str) -> list[str]:
    return [
        p
        for p in pgrep(pattern)
        if "python" in _cmdline(p) and "/bin/bash" not in _cmdline(p)
    ]


def train_pids(run_id: str) -> list[str]:
    cfg = manifest_by_id()[run_id]["config"]
    return (
        _python_pids(f"train_sepres_v2.py --config {cfg}")
        + _python_pids(f"train_ecbsr.py --config {cfg}")
        + _python_pids(f"train_mobile_srnet_kd.py --config {cfg}")
    )


def launcher_pids() -> list[str]:
    return [
        p
        for p in pgrep("run_b5_train_20k.py")
        if "python" in _cmdline(p) and "/bin/bash" not in _cmdline(p)
    ]


def legacy_launcher_pids() -> list[str]:
    """Old split launchers — must not co-exist with unified."""
    out = []
    for name in ("run_b5a_20k.py", "run_b5_confirm_20k.py", "run_pecsr_kd_20k.py"):
        out.extend(
            p
            for p in pgrep(name)
            if "python" in _cmdline(p) and "/bin/bash" not in _cmdline(p)
        )
    return out


def watch_pids() -> list[str]:
    return [
        p
        for p in pgrep("b5_train_20k.py watch")
        if p != str(os.getpid()) and "python" in _cmdline(p) and "/bin/bash" not in _cmdline(p)
    ]


def kill_pids(pids: list[str], sig: int = signal.SIGTERM) -> None:
    for pid in pids:
        try:
            os.kill(int(pid), sig)
        except ProcessLookupError:
            pass


def wait_pids_gone(get_pids, timeout_sec: int = 60) -> None:
    waited = 0
    while waited < timeout_sec:
        if not get_pids():
            return
        time.sleep(2)
        waited += 2
    kill_pids(get_pids(), signal.SIGKILL)


def last_epoch(run_id: str) -> int:
    log = ROOT / f"results/exp_runs/{run_id}/train_log.jsonl"
    if not log.exists():
        return 0
    lines = [l for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    return int(json.loads(lines[-1])["epoch"]) if lines else 0


def read_rows(run_id: str) -> list[dict]:
    f = ROOT / f"results/exp_runs/{run_id}/train_log.jsonl"
    if not f.exists():
        return []
    rows = []
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def fmt_dur(s: float) -> str:
    s = int(max(0.0, s))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def run_info(run_id: str) -> dict:
    meta = manifest_by_id()[run_id]
    target = int(meta["epochs"])
    rows = read_rows(run_id)
    lane = meta.get("lane", "mse")

    if rows:
        last = rows[-1]
        epoch = int(last["epoch"])
        psnr = float(last.get("val_psnr", float("nan")))
        best = max((float(r.get("val_psnr", -999.0)) for r in rows), default=float("nan"))
        spent = sum(float(r.get("elapsed_sec", 0.0)) for r in rows)
        recent = [float(r["elapsed_sec"]) for r in rows if "elapsed_sec" in r][-5:]
        avg_ep = mean(recent) if recent else (spent / epoch if epoch else 0.0)
        eta = avg_ep * max(0, target - epoch)
        gstep = int(last.get("global_step", 0))
    else:
        epoch, psnr, best, spent, eta, gstep = 0, float("nan"), float("nan"), 0.0, 0.0, 0

    if train_pids(run_id):
        state = "running"
    elif rows and epoch >= target:
        state = "done"
    elif lane == "kd" and run_info_quick_mse_pending():
        state = "gated"
    elif rows:
        state = "paused"
    else:
        state = "pending"

    return {
        "run_id": run_id,
        "lane": lane,
        "variant": meta.get("variant", "?"),
        "state": state,
        "epoch": epoch,
        "target": target,
        "psnr": psnr,
        "best": best,
        "spent_sec": spent,
        "eta_sec": eta,
        "global_step": gstep,
        "updates_target": int(meta.get("updates_target", 20000)),
        "progress_pct": 100.0 * epoch / target if target else 0.0,
    }


def run_info_quick_mse_pending() -> bool:
    """True if any MSE entry is not done (KD must wait)."""
    for e in load_manifest():
        if e.get("lane", "mse") != "mse":
            continue
        rid = e["run_id"]
        rows = read_rows(rid)
        if not rows:
            return True
        if int(rows[-1]["epoch"]) < int(e["epochs"]):
            return True
        ckpt = ROOT / f"results/exp_runs/{rid}/checkpoints/best.pt"
        if not ckpt.exists():
            return True
    return False


def render_watch(infos: list[dict]) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mse = [i for i in infos if i["lane"] == "mse"]
    kd = [i for i in infos if i["lane"] == "kd"]
    n_run = sum(1 for i in infos if i["state"] == "running")
    n_done = sum(1 for i in infos if i["state"] == "done")
    n_pend = sum(1 for i in infos if i["state"] in ("pending", "paused", "gated"))
    mse_run = sum(1 for i in mse if i["state"] == "running")
    mse_done = sum(1 for i in mse if i["state"] == "done")
    mse_n = len(mse)
    kd_done = sum(1 for i in kd if i["state"] == "done")
    lines = [
        f"B5 unified train — {now}",
        f"summary: running {n_run}/{len(infos)} · done {n_done} · left {n_pend} "
        f"| MSE {mse_run}/{mse_n} live · {mse_done} done · KD {kd_done}/{len(kd)} "
        f"| cap {DEFAULT_MAX_PARALLEL}-wide",
        "",
    ]
    hdr = (
        f"{'run_id':<32} {'lane':<4} {'var':<12} {'state':<8} {'epoch':>11} "
        f"{'val':>8} {'best':>8} {'ETA':>9} {'prog':>6}"
    )
    lines.extend([hdr, "-" * len(hdr)])
    for i in infos:
        psnr = f"{i['psnr']:.2f}" if i["psnr"] == i["psnr"] else "—"
        best = f"{i['best']:.2f}" if i["best"] == i["best"] else "—"
        ep_str = f"{i['epoch']}/{i['target']}"
        lines.append(
            f"{i['run_id']:<32} {i['lane']:<4} {str(i['variant']):<12} {i['state']:<8} "
            f"{ep_str:>11} {psnr:>8} {best:>8} {fmt_dur(i['eta_sec']):>9} "
            f"{i['progress_pct']:>5.1f}%"
        )
    lines.extend(
        [
            "",
            "Commands:  python scripts/b5_train_20k.py pause | resume | watch",
            "Resume:    --max-parallel 3 (MSE); KD Stage-B 2k auto after all MSE done",
            "Contract:  IMPLEMENTATION.md §14",
        ]
    )
    print("\n".join(lines))


def cmd_pause(dry_run: bool, run_ids: list[str] | None) -> None:
    ids = run_ids or [e["run_id"] for e in load_manifest()]
    print(f"Pausing B5 unified — {datetime.now():%Y-%m-%d %H:%M:%S}")
    state = {
        "paused_at": datetime.now().astimezone().isoformat(),
        "runs": [
            {
                "run_id": rid,
                "variant": manifest_by_id()[rid].get("variant"),
                "epoch": last_epoch(rid),
                "target_epochs": manifest_by_id()[rid]["epochs"],
                "resume_from": f"results/exp_runs/{rid}/checkpoints/latest.pt",
            }
            for rid in ids
        ],
    }
    print(json.dumps(state, indent=2))
    if dry_run:
        print("(dry-run: not killing anything)")
        return

    PAUSE_STATE.parent.mkdir(parents=True, exist_ok=True)
    PAUSE_STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")

    wp = watch_pids()
    if wp:
        print(f"Stopping watch (pids: {' '.join(wp)})...")
        kill_pids(wp)

    for rid in ids:
        pids = train_pids(rid)
        if pids:
            print(f"SIGTERM {rid} (pids: {' '.join(pids)})...")
            kill_pids(pids)

    wait_pids_gone(lambda: [p for rid in ids for p in train_pids(rid)])

    for label, getter in (
        ("unified launcher", launcher_pids),
        ("legacy launchers", legacy_launcher_pids),
    ):
        lp = getter()
        if lp:
            print(f"Stopping {label} (pids: {' '.join(lp)})...")
            kill_pids(lp)

    print("\nPaused.")
    print("Resume: python scripts/b5_train_20k.py resume")


def cmd_resume(dry_run: bool, max_parallel: int, mse_only: bool) -> None:
    print(f"Resuming B5 unified — {datetime.now():%Y-%m-%d %H:%M:%S}")
    any_pending = False
    for e in load_manifest():
        rid = e["run_id"]
        ep = last_epoch(rid)
        target = int(e["epochs"])
        if train_pids(rid):
            print(f"  {rid}: already training")
            any_pending = True
            continue
        if ep >= target and (ROOT / f"results/exp_runs/{rid}/checkpoints/best.pt").exists():
            print(f"  {rid}: done ({ep}/{target})")
            continue
        print(f"  {rid}: will run/adopt ({ep}/{target}) lane={e.get('lane')} variant={e.get('variant')}")
        any_pending = True

    if not any_pending:
        print("Nothing to resume.")
        return
    if dry_run:
        print("(dry-run: not launching)")
        return

    # Retire split launchers so they cannot double-schedule.
    leg = legacy_launcher_pids()
    if leg:
        print(f"Stopping legacy launchers (pids: {' '.join(leg)}) — trains keep running...")
        kill_pids(leg)
        time.sleep(2)

    if launcher_pids():
        print(f"Unified launcher already running (pids: {' '.join(launcher_pids())}).")
    else:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = open(LOG_DIR / "b5_train_20k_launcher.log", "a", encoding="utf-8")
        cmd = [
            PY,
            "-u",
            "scripts/run_b5_train_20k.py",
            "--skip-done",
            "--max-parallel",
            str(max_parallel),
        ]
        if mse_only:
            cmd.append("--mse-only")
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"Started launcher pid={proc.pid} cmd={' '.join(cmd)}")
        log_file.close()

    print("Monitor: python scripts/b5_train_20k.py watch --interval 60")


def cmd_watch(interval: int) -> None:
    ids = [e["run_id"] for e in load_manifest()]
    try:
        while True:
            infos = [run_info(rid) for rid in ids]
            sys.stdout.write("\033[2J\033[H")
            render_watch(infos)
            sys.stdout.flush()
            if all(i["state"] == "done" for i in infos):
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description="B5 unified: pause | resume | watch")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_pause = sub.add_parser("pause")
    p_pause.add_argument("--dry-run", action="store_true")
    p_pause.add_argument("--run-id", default=None)

    p_resume = sub.add_parser("resume")
    p_resume.add_argument("--dry-run", action="store_true")
    p_resume.add_argument("--max-parallel", type=int, default=DEFAULT_MAX_PARALLEL)
    p_resume.add_argument("--mse-only", action="store_true")

    p_watch = sub.add_parser("watch")
    p_watch.add_argument("--interval", type=int, default=60)

    args = ap.parse_args()
    if args.cmd == "pause":
        ids = [args.run_id] if args.run_id else None
        cmd_pause(args.dry_run, ids)
    elif args.cmd == "resume":
        cmd_resume(args.dry_run, args.max_parallel, args.mse_only)
    elif args.cmd == "watch":
        cmd_watch(args.interval)


if __name__ == "__main__":
    main()
