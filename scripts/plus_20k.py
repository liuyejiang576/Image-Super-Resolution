#!/usr/bin/env python3
"""MobileSRNet-Plus 20k control: pause | resume | watch.

  python scripts/plus_20k.py pause [--dry-run]
  python scripts/plus_20k.py resume [--dry-run]
  python scripts/plus_20k.py watch [--interval 60]

Launcher (run_plus_20k.py) is started by resume only.
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

MANIFEST = ROOT / "results/exp_runs/plus_20k_manifest.json"
REFERENCE = ROOT / "results/exp_runs/plus_20k_reference.json"
PAUSE_STATE = ROOT / "results/exp_runs/plus_20k_paused.json"
LOG_DIR = ROOT / "results/exp_runs/logs"
RUN_IDS = ["mobile_srnet_plus_20k"]


def load_manifest() -> list[dict]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def manifest_by_id() -> dict[str, dict]:
    return {e["run_id"]: e for e in load_manifest()}


def load_reference() -> dict:
    if REFERENCE.exists():
        return json.loads(REFERENCE.read_text(encoding="utf-8"))
    return {"baseline_val_psnr": float("nan")}


def pgrep(pattern: str) -> list[str]:
    try:
        out = subprocess.check_output(["pgrep", "-f", pattern], text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return []
    return [p for p in out.split() if p]


def train_pids(run_id: str) -> list[str]:
    cfg = manifest_by_id()[run_id]["config"]
    return pgrep(f"train_mobile_srnet.py --config {cfg}")


def launcher_pids() -> list[str]:
    return pgrep("run_plus_20k.py")


def watch_pids() -> list[str]:
    return [p for p in pgrep("plus_20k.py watch") if p != str(os.getpid())]


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
    ref = load_reference()
    target = int(meta["epochs"])
    rows = read_rows(run_id)
    base_psnr = float(ref.get("baseline_val_psnr", float("nan")))

    if rows:
        last = rows[-1]
        epoch = int(last["epoch"])
        psnr = float(last.get("val_psnr", float("nan")))
        best = max((float(r.get("val_psnr", -999.0)) for r in rows), default=float("nan"))
        spent = sum(float(r.get("elapsed_sec", 0.0)) for r in rows)
        recent = [float(r["elapsed_sec"]) for r in rows if "elapsed_sec" in r][-5:]
        avg_ep = mean(recent) if recent else (spent / epoch if epoch else 0.0)
        eta = avg_ep * max(0, target - epoch)
        delta_base = best - base_psnr if best == best and base_psnr == base_psnr else float("nan")
    else:
        epoch, psnr, best, spent, eta, delta_base = 0, float("nan"), float("nan"), 0.0, 0.0, float("nan")

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
        "variant": meta.get("variant", "plus"),
        "state": state,
        "epoch": epoch,
        "target": target,
        "psnr": psnr,
        "best": best,
        "base_psnr": base_psnr,
        "delta_base": delta_base,
        "spent_sec": spent,
        "eta_sec": eta,
        "progress_pct": 100.0 * epoch / target if target else 0.0,
    }


def render_watch(infos: list[dict]) -> None:
    ref = load_reference()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"MobileSRNet-Plus 20k — {now}", ""]
    hdr = (
        f"{'run_id':<26} {'state':<8} {'epoch':>11} {'val_psnr':>9} {'best':>9} "
        f"{'Δbase':>8} {'spent':>9} {'ETA':>9} {'prog':>6}"
    )
    lines.extend([hdr, "-" * len(hdr)])
    for i in infos:
        psnr = f"{i['psnr']:.3f}" if i["psnr"] == i["psnr"] else "—"
        best = f"{i['best']:.3f}" if i["best"] == i["best"] else "—"
        dbase = f"{i['delta_base']:+.3f}" if i["delta_base"] == i["delta_base"] else "—"
        ep_str = f"{i['epoch']}/{i['target']}"
        lines.append(
            f"{i['run_id']:<26} {i['state']:<8} {ep_str:>11} {psnr:>9} {best:>9} "
            f"{dbase:>8} {fmt_dur(i['spent_sec']):>9} {fmt_dur(i['eta_sec']):>9} {i['progress_pct']:>5.1f}%"
        )
    lines.extend([
        "",
        f"Base reference (mobile_srnet_20k): val_psnr={ref.get('baseline_val_psnr', '?'):.4f}  "
        f"avg_benchmark={ref.get('baseline_avg_psnr_benchmarks', '?')} dB",
    ])
    if infos and infos[0]["state"] == "done":
        d = infos[0]["delta_base"]
        if d == d:
            lines.append(f"Final Δ(best vs Base val) = {d:+.3f} dB — run full benchmark eval next.")
    print("\n".join(lines))


def cmd_pause(dry_run: bool) -> None:
    print(f"Pausing MobileSRNet-Plus 20k — {datetime.now():%Y-%m-%d %H:%M:%S}")
    state = {
        "paused_at": datetime.now().astimezone().isoformat(),
        "runs": [
            {
                "run_id": rid,
                "variant": meta.get("variant", "plus"),
                "feat": meta.get("feat"),
                "num_blocks": meta.get("num_blocks"),
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

    print("\nPaused. Checkpoint: results/exp_runs/mobile_srnet_plus_20k/checkpoints/latest.pt")
    print("Resume: python scripts/plus_20k.py resume")


def cmd_resume(dry_run: bool) -> None:
    print(f"Resuming MobileSRNet-Plus 20k — {datetime.now():%Y-%m-%d %H:%M:%S}")
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
        print(f"  {rid}: will run ({ep}/{target}) variant={meta.get('variant', 'plus')}")
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
        log_file = open(LOG_DIR / "plus_20k_launcher.log", "a", encoding="utf-8")
        proc = subprocess.Popen(
            [PY, "scripts/run_plus_20k.py", "--skip-done"],
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"Started launcher pid={proc.pid}")
        log_file.close()

    print("Monitor: python scripts/plus_20k.py watch --interval 60")


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
    ap = argparse.ArgumentParser(description="MobileSRNet-Plus 20k: pause | resume | watch")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_pause = sub.add_parser("pause", help="Stop training + launcher, keep checkpoints")
    p_pause.add_argument("--dry-run", action="store_true")

    p_resume = sub.add_parser("resume", help="Start launcher (--resume-from latest.pt if partial)")
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
