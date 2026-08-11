#!/usr/bin/env python3
"""SepResSR-v2 20k control: pause | resume | watch (B4).

Default resume trains **v2_b (c16n10)** only. Gated a/c need explicit --run-id.

  python scripts/sepres_v2_20k.py pause [--dry-run]
  python scripts/sepres_v2_20k.py resume [--dry-run] [--run-id sepres_v2_c16n10_20k]
  python scripts/sepres_v2_20k.py watch [--interval 60]

See IMPLEMENTATION.md §1–§2 and §11.
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

MANIFEST = ROOT / "results/exp_runs/sepres_v2_20k_manifest.json"
PAUSE_STATE = ROOT / "results/exp_runs/sepres_v2_20k_paused.json"
LOG_DIR = ROOT / "results/exp_runs/logs"
DEFAULT_RUN_IDS = ["sepres_v2_c16n10_20k"]


def load_manifest() -> list[dict]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def manifest_by_id() -> dict[str, dict]:
    return {e["run_id"]: e for e in load_manifest()}


def resolve_run_ids(run_id: str | None, all_gated: bool) -> list[str]:
    known = set(manifest_by_id())
    if run_id:
        if run_id not in known:
            raise SystemExit(f"unknown run_id: {run_id}")
        return [run_id]
    if all_gated:
        return [e["run_id"] for e in load_manifest()]
    return list(DEFAULT_RUN_IDS)


def pgrep(pattern: str) -> list[str]:
    try:
        out = subprocess.check_output(["pgrep", "-f", pattern], text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return []
    return [p for p in out.split() if p]


def train_pids(run_id: str) -> list[str]:
    cfg = manifest_by_id()[run_id]["config"]
    return pgrep(f"train_sepres_v2.py --config {cfg}") + pgrep(f"train_ecbsr.py --config {cfg}")


def launcher_pids() -> list[str]:
    return pgrep("run_sepres_v2_20k.py")


def watch_pids() -> list[str]:
    return [p for p in pgrep("sepres_v2_20k.py watch") if p != str(os.getpid())]


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
    elif rows:
        state = "paused"
    else:
        state = "pending"

    return {
        "run_id": run_id,
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


def render_watch(infos: list[dict]) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"SepResSR-v2 20k (B4) — {now}", ""]
    hdr = (
        f"{'run_id':<24} {'var':<7} {'state':<8} {'epoch':>11} {'step':>10} "
        f"{'val_psnr':>9} {'best':>9} {'spent':>9} {'ETA':>9} {'prog':>6}"
    )
    lines.extend([hdr, "-" * len(hdr)])
    for i in infos:
        psnr = f"{i['psnr']:.3f}" if i["psnr"] == i["psnr"] else "—"
        best = f"{i['best']:.3f}" if i["best"] == i["best"] else "—"
        ep_str = f"{i['epoch']}/{i['target']}"
        step_str = f"{i['global_step']}/{i['updates_target']}"
        lines.append(
            f"{i['run_id']:<24} {i['variant']:<7} {i['state']:<8} {ep_str:>11} {step_str:>10} "
            f"{psnr:>9} {best:>9} {fmt_dur(i['spent_sec']):>9} {fmt_dur(i['eta_sec']):>9} "
            f"{i['progress_pct']:>5.1f}%"
        )
    lines.extend(
        [
            "",
            "Commands:  python scripts/sepres_v2_20k.py pause | resume | watch",
            "Default:   resume trains v2_b only; gated a/c need --run-id",
            "Contract:  IMPLEMENTATION.md §11",
        ]
    )
    print("\n".join(lines))


def cmd_pause(dry_run: bool, run_ids: list[str]) -> None:
    print(f"Pausing SepResV2 20k — {datetime.now():%Y-%m-%d %H:%M:%S}")
    state = {
        "paused_at": datetime.now().astimezone().isoformat(),
        "runs": [
            {
                "run_id": rid,
                "variant": meta.get("variant"),
                "epoch": last_epoch(rid),
                "target_epochs": meta["epochs"],
                "resume_from": f"results/exp_runs/{rid}/checkpoints/latest.pt",
            }
            for rid in run_ids
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

    for rid in run_ids:
        pids = train_pids(rid)
        if pids:
            print(f"SIGTERM {rid} (pids: {' '.join(pids)})...")
            kill_pids(pids)

    def any_train():
        return [p for rid in run_ids for p in train_pids(rid)]

    wait_pids_gone(any_train)

    lp = launcher_pids()
    if lp:
        print(f"Stopping launcher (pids: {' '.join(lp)})...")
        kill_pids(lp)

    print("\nPaused.")
    print("Resume: python scripts/sepres_v2_20k.py resume")


def cmd_resume(dry_run: bool, run_ids: list[str], all_gated: bool) -> None:
    print(f"Resuming SepResV2 20k — {datetime.now():%Y-%m-%d %H:%M:%S}")
    any_pending = False
    for rid in run_ids:
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
        print(f"  {rid}: will run ({ep}/{target}) variant={meta.get('variant')}")
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
        log_file = open(LOG_DIR / "sepres_v2_20k_launcher.log", "a", encoding="utf-8")
        cmd = [PY, "scripts/run_sepres_v2_20k.py", "--skip-done"]
        if len(run_ids) == 1 and run_ids[0] not in DEFAULT_RUN_IDS:
            cmd += ["--run-id", run_ids[0]]
        elif all_gated:
            cmd.append("--all-gated")
        elif len(run_ids) == 1:
            cmd += ["--run-id", run_ids[0]]
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"Started launcher pid={proc.pid} cmd={' '.join(cmd)}")
        log_file.close()

    print("Monitor: python scripts/sepres_v2_20k.py watch --interval 60")


def cmd_watch(interval: int, run_ids: list[str]) -> None:
    try:
        while True:
            infos = [run_info(rid) for rid in run_ids]
            sys.stdout.write("\033[2J\033[H")
            render_watch(infos)
            sys.stdout.flush()
            if all(i["state"] == "done" for i in infos):
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description="SepResSR-v2 20k: pause | resume | watch")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_run_sel(p: argparse.ArgumentParser) -> None:
        p.add_argument("--run-id", default=None, help="Single candidate (default: v2_b)")
        p.add_argument(
            "--all-gated",
            action="store_true",
            help="Include gated a/c from manifest (only after B4 gates)",
        )

    p_pause = sub.add_parser("pause", help="Stop training + launcher, keep checkpoints")
    p_pause.add_argument("--dry-run", action="store_true")
    add_run_sel(p_pause)

    p_resume = sub.add_parser("resume", help="Start launcher (v2_b by default)")
    p_resume.add_argument("--dry-run", action="store_true")
    add_run_sel(p_resume)

    p_watch = sub.add_parser("watch", help="Live progress table")
    p_watch.add_argument("--interval", type=int, default=60)
    add_run_sel(p_watch)

    args = ap.parse_args()
    run_ids = resolve_run_ids(getattr(args, "run_id", None), getattr(args, "all_gated", False))
    if args.cmd == "pause":
        # Pause all known train processes that match selected ids; if default, also
        # pause any accidentally started gated runs.
        pause_ids = run_ids if args.run_id or args.all_gated else list(manifest_by_id())
        cmd_pause(args.dry_run, pause_ids)
    elif args.cmd == "resume":
        cmd_resume(args.dry_run, run_ids, args.all_gated)
    elif args.cmd == "watch":
        watch_ids = run_ids if args.run_id or args.all_gated else list(manifest_by_id())
        cmd_watch(args.interval, watch_ids)


if __name__ == "__main__":
    main()
