#!/usr/bin/env python3
"""Architecture 30k continuation control: pause | resume | watch.

Continues FSRCNN, MobileSRNet-Base, and MobileSRNet-Plus from their 20k
checkpoints to a 30k optimizer-step budget (honest label: 20k + ~10k low-LR
fine-tuning, not a fresh 30k-from-scratch recipe).

  python scripts/arch_30k.py pause [--dry-run]
  python scripts/arch_30k.py resume [--dry-run] [--run-id RUN_ID]
  python scripts/arch_30k.py watch [--interval 60]

Launcher (run_arch_30k.py) starts up to 3 MSE jobs in parallel.
Watch in a separate terminal — shows GPU util, ep/min, and observe warnings.
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

import yaml

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

PY = sys.executable
if Path("/home/hyb/miniforge3/envs/cv_env/bin/python").exists():
    PY = "/home/hyb/miniforge3/envs/cv_env/bin/python"

MANIFEST = ROOT / "results/exp_runs/arch_30k_manifest.json"
REFERENCE = ROOT / "results/exp_runs/arch_30k_reference.json"
PAUSE_STATE = ROOT / "results/exp_runs/arch_30k_paused.json"
OBSERVE_PATH = ROOT / "results/exp_runs/arch_30k_observe.json"
LOG_DIR = ROOT / "results/exp_runs/logs"


def load_manifest() -> list[dict]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def manifest_by_id() -> dict[str, dict]:
    return {e["run_id"]: e for e in load_manifest()}


def run_ids() -> list[str]:
    return [e["run_id"] for e in load_manifest()]


def load_reference() -> dict:
    if REFERENCE.exists():
        return json.loads(REFERENCE.read_text(encoding="utf-8"))
    return {"models": {}}


def frozen_val_psnr(run_id: str) -> float:
    ref = load_reference()
    model_ref = ref.get("models", {}).get(run_id, {})
    val = model_ref.get("frozen_best_val_psnr", model_ref.get("frozen_val_psnr", float("nan")))
    return float(val)


def target_epochs(run_id: str) -> int:
    meta = manifest_by_id()[run_id]
    cfg_path = ROOT / meta["config"]
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return int(cfg["train"]["epochs"])


def updates_target(run_id: str) -> int:
    return int(manifest_by_id()[run_id].get("updates_target", 30000))


def train_script(entry: dict) -> str:
    if entry["model"] == "fsrcnn":
        return "train_fsrcnn.py"
    return "train_mobile_srnet.py"


def pgrep(pattern: str) -> list[str]:
    try:
        out = subprocess.check_output(["pgrep", "-f", pattern], text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return []
    return [p for p in out.split() if p]


def train_pids(run_id: str) -> list[str]:
    meta = manifest_by_id()[run_id]
    cfg = meta["config"]
    script = train_script(meta)
    return pgrep(f"{script} --config {cfg}")


def launcher_pids() -> list[str]:
    return pgrep("run_arch_30k.py")


def watch_pids() -> list[str]:
    return [p for p in pgrep("arch_30k.py watch") if p != str(os.getpid())]


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


def log_path(run_id: str) -> Path:
    meta = manifest_by_id()[run_id]
    cfg_path = ROOT / meta["config"]
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return ROOT / cfg["checkpoint"]["log_path"]


def read_rows(run_id: str) -> list[dict]:
    f = log_path(run_id)
    if f.exists():
        rows = []
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if rows:
            return rows
    # Before first bootstrap, show 20k history in watch/pause.
    source_id = manifest_by_id()[run_id].get("resume_from_run_id")
    if source_id:
        src_log = ROOT / f"results/exp_runs/{source_id}/train_log.jsonl"
        if src_log.exists():
            rows = []
            for line in src_log.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            return rows
    return []


def last_epoch(run_id: str) -> int:
    rows = read_rows(run_id)
    return int(rows[-1]["epoch"]) if rows else 0


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


def load_observe() -> dict | None:
    if OBSERVE_PATH.exists():
        try:
            return json.loads(OBSERVE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


def fmt_dur(s: float) -> str:
    s = int(max(0.0, s))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def run_info(run_id: str) -> dict:
    meta = manifest_by_id()[run_id]
    target_ep = target_epochs(run_id)
    target_steps = updates_target(run_id)
    rows = read_rows(run_id)
    base_psnr = frozen_val_psnr(run_id)

    if rows:
        last = rows[-1]
        epoch = int(last["epoch"])
        gstep = int(last.get("global_step", 0))
        psnr = float(last.get("val_psnr", float("nan")))
        best = max((float(r.get("val_psnr", -999.0)) for r in rows), default=float("nan"))
        spent = sum(float(r.get("elapsed_sec", 0.0)) for r in rows)
        # ETA from continuation segment only (epochs beyond frozen 20k end)
        frozen_ep = int(
            load_reference().get("models", {}).get(run_id, {}).get("frozen_epoch", 0)
        )
        cont_rows = [r for r in rows if int(r["epoch"]) > frozen_ep]
        recent = [float(r["elapsed_sec"]) for r in cont_rows if "elapsed_sec" in r][-5:]
        if not recent and rows:
            recent = [float(r["elapsed_sec"]) for r in rows if "elapsed_sec" in r][-5:]
        avg_ep = mean(recent) if recent else (spent / epoch if epoch else 0.0)
        eta = avg_ep * max(0, target_ep - epoch)
        delta_base = best - base_psnr if best == best and base_psnr == base_psnr else float("nan")
    else:
        epoch, gstep, psnr, best, spent, eta, delta_base = 0, 0, float("nan"), float("nan"), 0.0, 0.0, float("nan")

    if train_pids(run_id):
        state = "running"
    elif rows and epoch >= target_ep:
        state = "done"
    elif rows:
        state = "paused"
    else:
        state = "pending"

    return {
        "run_id": run_id,
        "variant": meta.get("variant", meta["model"]),
        "state": state,
        "epoch": epoch,
        "target_epoch": target_ep,
        "global_step": gstep,
        "target_steps": target_steps,
        "psnr": psnr,
        "best": best,
        "frozen_psnr": base_psnr,
        "delta_frozen": delta_base,
        "spent_sec": spent,
        "eta_sec": eta,
        "progress_pct": 100.0 * gstep / target_steps if target_steps else 0.0,
        "ep_per_min": None,
    }


def render_watch(infos: list[dict], observe: dict | None = None) -> None:
    ref = load_reference()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"Architecture 30k continuation — {now}",
        ref.get("label", "20k + ~10k low-LR fine-tuning"),
    ]
    gpu = gpu_snapshot()
    if gpu:
        lines.append(
            f"GPU:   {gpu['used_mib']} / {gpu['total_mib']} MiB · {gpu['util_pct']}% util"
        )
    if observe:
        layout = observe.get("layout", "?")
        mp = observe.get("max_parallel", "?")
        lines.append(f"Layout: {layout} (max {mp} MSE jobs) · poll {observe.get('poll', '—')}")
    lines.append("")

    hdr = (
        f"{'run_id':<26} {'var':<6} {'state':<8} {'epoch':>12} {'steps':>14} "
        f"{'val':>8} {'best':>8} {'Δ20k':>8} {'ep/min':>7} {'spent':>9} {'ETA':>9} {'prog':>6}"
    )
    lines.extend([hdr, "-" * len(hdr)])
    obs_jobs = (observe or {}).get("jobs", {})
    for i in infos:
        psnr = f"{i['psnr']:.3f}" if i["psnr"] == i["psnr"] else "—"
        best = f"{i['best']:.3f}" if i["best"] == i["best"] else "—"
        dfrozen = f"{i['delta_frozen']:+.3f}" if i["delta_frozen"] == i["delta_frozen"] else "—"
        ep_str = f"{i['epoch']}/{i['target_epoch']}"
        step_str = f"{i['global_step']}/{i['target_steps']}"
        rate = i.get("ep_per_min")
        if rate is None:
            oj = obs_jobs.get(i["run_id"], {})
            rate = oj.get("ep_per_min")
        epmin = f"{rate:.2f}" if rate is not None and rate == rate else "—"
        lines.append(
            f"{i['run_id']:<26} {i['variant']:<6} {i['state']:<8} {ep_str:>12} {step_str:>14} "
            f"{psnr:>8} {best:>8} {dfrozen:>8} {epmin:>7} {fmt_dur(i['spent_sec']):>9} "
            f"{fmt_dur(i['eta_sec']):>9} {i['progress_pct']:>5.1f}%"
        )

    done = sum(1 for i in infos if i["state"] == "done")
    running = sum(1 for i in infos if i["state"] == "running")
    lines.extend([
        "",
        f"Summary: {done}/{len(infos)} done, {running} running",
    ])
    if observe and observe.get("warnings"):
        lines.append("")
        lines.append("Observe warnings:")
        for w in observe["warnings"]:
            lines.append(f"  ! {w}")
    if done == len(infos):
        lines.append("All 30k runs complete — run benchmark eval + sync report metrics.")
    elif any(i["state"] == "running" for i in infos):
        lines.append("Pause:  python scripts/arch_30k.py pause")
        lines.append("Watch:  python scripts/arch_30k.py watch --interval 30")
    else:
        lines.append("Resume: python scripts/arch_30k.py resume")
    print("\n".join(lines))


def cmd_pause(dry_run: bool) -> None:
    print(f"Pausing architecture 30k continuation — {datetime.now():%Y-%m-%d %H:%M:%S}")
    state = {
        "paused_at": datetime.now().astimezone().isoformat(),
        "label": load_reference().get("label"),
        "runs": [
            {
                "run_id": rid,
                "variant": meta.get("variant", meta["model"]),
                "epoch": last_epoch(rid),
                "target_epochs": target_epochs(rid),
                "global_step": read_rows(rid)[-1].get("global_step") if read_rows(rid) else 0,
                "resume_from": f"results/exp_runs/{rid}/checkpoints/latest.pt",
                "bootstrap_from": meta.get("resume_from_run_id"),
            }
            for rid in run_ids()
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

    for rid in run_ids():
        pids = train_pids(rid)
        if pids:
            print(f"SIGTERM {rid} (pids: {' '.join(pids)})...")
            kill_pids(pids)

    def any_train():
        return [p for rid in run_ids() for p in train_pids(rid)]

    wait_pids_gone(any_train)

    lp = launcher_pids()
    if lp:
        print(f"Stopping launcher (pids: {' '.join(lp)})...")
        kill_pids(lp)

    print("\nPaused. Checkpoints under results/exp_runs/*_30k/checkpoints/latest.pt")
    print("Resume: python scripts/arch_30k.py resume")


def cmd_resume(dry_run: bool, single_run_id: str | None) -> None:
    print(f"Resuming architecture 30k continuation — {datetime.now():%Y-%m-%d %H:%M:%S}")
    ids = [single_run_id] if single_run_id else run_ids()
    if single_run_id and single_run_id not in manifest_by_id():
        raise SystemExit(f"Unknown run_id: {single_run_id}")

    any_pending = False
    for rid in ids:
        meta = manifest_by_id()[rid]
        target = target_epochs(rid)
        ep = last_epoch(rid)
        ckpt_30k = ROOT / f"results/exp_runs/{rid}/checkpoints/latest.pt"
        ckpt_20k = ROOT / f"results/exp_runs/{meta['resume_from_run_id']}/checkpoints/latest.pt"

        if train_pids(rid):
            print(f"  {rid}: already training — skip")
            any_pending = True
            continue
        if ep >= target:
            print(f"  {rid}: done ({ep}/{target}) — skip")
            continue
        if ckpt_30k.exists() or ckpt_20k.exists():
            src = "30k latest" if ckpt_30k.exists() else f"20k bootstrap ({meta['resume_from_run_id']})"
            print(f"  {rid}: will run ({ep}/{target}) variant={meta.get('variant')} from {src}")
            any_pending = True
        else:
            print(f"  {rid}: no checkpoint found — skip")

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
        log_file = open(LOG_DIR / "arch_30k_launcher.log", "a", encoding="utf-8")
        cmd = [PY, "scripts/run_arch_30k.py", "--skip-done"]
        if single_run_id:
            cmd += ["--run-id", single_run_id]
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"Started launcher pid={proc.pid}")
        log_file.close()

    print("Monitor: python scripts/arch_30k.py watch --interval 30")


def cmd_watch(interval: int) -> None:
    prev_epoch: dict[str, tuple[float, int]] = {}
    try:
        while True:
            now = time.time()
            infos = []
            for rid in run_ids():
                info = run_info(rid)
                epoch = info["epoch"]
                if rid in prev_epoch:
                    t0, e0 = prev_epoch[rid]
                    dt_min = (now - t0) / 60.0
                    if dt_min > 0 and epoch > e0:
                        info["ep_per_min"] = (epoch - e0) / dt_min
                prev_epoch[rid] = (now, epoch)
                infos.append(info)

            observe = load_observe()
            sys.stdout.write("\033[2J\033[H")
            render_watch(infos, observe)
            sys.stdout.flush()
            if all(i["state"] == "done" for i in infos):
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Architecture 30k continuation (FSRCNN + Base + Plus): pause | resume | watch"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_pause = sub.add_parser("pause", help="Stop training + launcher, keep checkpoints")
    p_pause.add_argument("--dry-run", action="store_true")

    p_resume = sub.add_parser("resume", help="Start launcher (bootstrap from 20k or resume 30k)")
    p_resume.add_argument("--dry-run", action="store_true")
    p_resume.add_argument("--run-id", default=None, help="Resume a single run only")

    p_watch = sub.add_parser("watch", help="Live progress table for all three models")
    p_watch.add_argument("--interval", type=int, default=30, help="Refresh seconds (default 30)")

    args = ap.parse_args()
    if args.cmd == "pause":
        cmd_pause(args.dry_run)
    elif args.cmd == "resume":
        cmd_resume(args.dry_run, args.run_id)
    elif args.cmd == "watch":
        cmd_watch(args.interval)


if __name__ == "__main__":
    main()
