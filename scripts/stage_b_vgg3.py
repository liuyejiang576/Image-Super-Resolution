#!/usr/bin/env python3
"""Stage B vgg_relu3 control: pause | resume | watch.

  python scripts/stage_b_vgg3.py pause [--dry-run]
  python scripts/stage_b_vgg3.py resume [--dry-run]
  python scripts/stage_b_vgg3.py watch [--interval 60]

Sequential launcher (run_stage_b_vgg3.py) is started by resume only.
Watch in a separate terminal while training runs.
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

MANIFEST = ROOT / "results/exp_runs/stage_b_vgg3_manifest.json"
PAUSE_STATE = ROOT / "results/exp_runs/stage_b_vgg3_paused.json"
LOG_DIR = ROOT / "results/exp_runs/logs"
RUN_IDS = ["mobile_srnet_vgg3_kd0_2k", "mobile_srnet_vgg3_kd01_2k"]


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


def train_pids(run_id: str) -> list[str]:
    cfg = manifest_by_id()[run_id]["config"]
    return pgrep(f"train_mobile_srnet_kd.py --config {cfg}")


def launcher_pids() -> list[str]:
    return pgrep("run_stage_b_vgg3.py")


def watch_pids() -> list[str]:
    return [p for p in pgrep("stage_b_vgg3.py watch") if p != str(os.getpid())]


def kill_pids(pids: list[str], sig: int = signal.SIGTERM) -> None:
    for pid in pids:
        try:
            os.kill(int(pid), sig)
        except ProcessLookupError:
            pass


def wait_pids_gone(get_pids, timeout_sec: int = 30) -> None:
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


def log_path(run_id: str) -> Path:
    return ROOT / f"results/exp_runs/{run_id}/train_log.jsonl"


def read_rows(run_id: str) -> list[dict]:
    f = log_path(run_id)
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
    if rows:
        last = rows[-1]
        epoch = int(last["epoch"])
        psnr = float(last.get("val_psnr", float("nan")))
        best = max((float(r.get("val_psnr", -999.0)) for r in rows), default=float("nan"))
        spent = sum(float(r.get("elapsed_sec", 0.0)) for r in rows)
        recent = [float(r["elapsed_sec"]) for r in rows if "elapsed_sec" in r][-5:]
        avg_ep = mean(recent) if recent else (spent / epoch if epoch else 0.0)
        eta = avg_ep * max(0, target - epoch)
    else:
        epoch, psnr, best, spent, eta = 0, float("nan"), float("nan"), 0.0, 0.0

    if train_pids(run_id):
        state = "running"
    elif rows and epoch >= target:
        state = "done"
    elif rows:
        state = "paused"
    else:
        state = "pending"

    return {
        "run_id": run_id,
        "state": state,
        "epoch": epoch,
        "target": target,
        "psnr": psnr,
        "best": best,
        "spent_sec": spent,
        "eta_sec": eta,
        "progress_pct": 100.0 * epoch / target if target else 0.0,
    }


def render_watch(infos: list[dict]) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"Stage B vgg_relu3 — {now}", ""]
    hdr = (
        f"{'run_id':<28} {'state':<8} {'epoch':>9} {'val_psnr':>9} {'best':>9} "
        f"{'spent':>9} {'ETA':>9} {'prog':>6}"
    )
    lines.extend([hdr, "-" * len(hdr)])
    for i in infos:
        psnr = f"{i['psnr']:.3f}" if i["psnr"] == i["psnr"] else "—"
        best = f"{i['best']:.3f}" if i["best"] == i["best"] else "—"
        ep_str = f"{i['epoch']}/{i['target']}"
        lines.append(
            f"{i['run_id']:<28} {i['state']:<8} {ep_str:>9} {psnr:>9} {best:>9} "
            f"{fmt_dur(i['spent_sec']):>9} {fmt_dur(i['eta_sec']):>9} {i['progress_pct']:>5.1f}%"
        )
    if len(infos) >= 2 and all(i["state"] == "done" for i in infos):
        d = infos[1]["best"] - infos[0]["best"]
        if d == d:
            lines.extend([
                "",
                f"Δ(best val PSNR) treatment−control = {d:+.3f} dB  "
                f"(pass if > +0.05 dB; commit 20k if clear)",
            ])
    print("\n".join(lines))


def cmd_pause(dry_run: bool) -> None:
    print(f"Pausing Stage B vgg_relu3 — {datetime.now():%Y-%m-%d %H:%M:%S}")
    state = {
        "paused_at": datetime.now().astimezone().isoformat(),
        "runs": [
            {
                "run_id": rid,
                "lambda_kd": meta["lambda_kd"],
                "kd_method": meta.get("kd_method", "vgg_relu3"),
                "epoch": last_epoch(rid),
                "target_epochs": meta["epochs"],
                "resume_from": f"results/exp_runs/{rid}/checkpoints/latest.pt",
            }
            for rid in RUN_IDS
            for meta in [manifest_by_id()[rid]]
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

    for rid in RUN_IDS:
        pids = train_pids(rid)
        if pids:
            print(f"SIGTERM {rid} (pids: {' '.join(pids)})...")
            kill_pids(pids)

    def any_train():
        return [p for rid in RUN_IDS for p in train_pids(rid)]

    wait_pids_gone(any_train)

    lp = launcher_pids()
    if lp:
        print(f"Stopping launcher (pids: {' '.join(lp)})...")
        kill_pids(lp)

    print("\nPaused. Checkpoints at results/exp_runs/mobile_srnet_vgg3_kd*_2k/checkpoints/latest.pt")
    print("Resume: python scripts/stage_b_vgg3.py resume")


def cmd_resume(dry_run: bool) -> None:
    print(f"Resuming Stage B vgg_relu3 — {datetime.now():%Y-%m-%d %H:%M:%S}")
    any_pending = False
    for rid in RUN_IDS:
        meta = manifest_by_id()[rid]
        target = int(meta["epochs"])
        ep = last_epoch(rid)
        ckpt = ROOT / f"results/exp_runs/{rid}/checkpoints/latest.pt"
        if train_pids(rid):
            print(f"  {rid}: already training — skip")
            any_pending = True
            continue
        if ep >= target:
            print(f"  {rid}: done ({ep}/{target}) — skip")
            continue
        if ep > 0 and not ckpt.exists():
            print(f"  {rid}: partial log but no checkpoint — skip")
            continue
        print(f"  {rid}: will run ({ep}/{target}) λ={meta['lambda_kd']}")
        any_pending = True

    if not any_pending:
        print("Nothing to resume.")
        return
    if dry_run:
        print("(dry-run: not launching)")
        return

    if launcher_pids():
        print(f"Launcher already running (pids: {' '.join(launcher_pids())}).")
    else:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = open(LOG_DIR / "stage_b_vgg3_launcher.log", "a", encoding="utf-8")
        proc = subprocess.Popen(
            [PY, "scripts/run_stage_b_vgg3.py", "--skip-done"],
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"Started sequential launcher pid={proc.pid}")
        log_file.close()

    print("Monitor: python scripts/stage_b_vgg3.py watch --interval 60")


def cmd_watch(interval: int) -> None:
    try:
        while True:
            infos = [run_info(rid) for rid in RUN_IDS]
            sys.stdout.write("\033[2J\033[H")
            render_watch(infos)
            sys.stdout.flush()
            if all(i["state"] == "done" for i in infos):
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage B vgg_relu3: pause | resume | watch")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_pause = sub.add_parser("pause", help="Stop training + launcher, keep checkpoints")
    p_pause.add_argument("--dry-run", action="store_true")

    p_resume = sub.add_parser("resume", help="Start sequential launcher (--resume-from latest.pt)")
    p_resume.add_argument("--dry-run", action="store_true")

    p_watch = sub.add_parser("watch", help="Live progress table")
    p_watch.add_argument("--interval", type=int, default=60, help="Refresh seconds (default 60)")

    args = ap.parse_args()
    if args.cmd == "pause":
        cmd_pause(args.dry_run)
    elif args.cmd == "resume":
        cmd_resume(args.dry_run)
    elif args.cmd == "watch":
        cmd_watch(args.interval)


if __name__ == "__main__":
    main()
