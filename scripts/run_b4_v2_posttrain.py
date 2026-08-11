#!/usr/bin/env python3
"""B4 Gate-2 post-train pipeline: fuse → NCNN → paired phone → b4_v2_compare.json.

Official path (after ``sepres_v2_c16n10_20k`` finishes):

  export PATH=$HOME/miniforge3/bin:$HOME/android/platform-tools:$PATH
  python scripts/run_b4_v2_posttrain.py --wait
  # or, when best.pt already exists:
  python scripts/run_b4_v2_posttrain.py

Warmup / mid-train smoke (does NOT write official compare):

  python scripts/run_b4_v2_posttrain.py \\
    --checkpoint results/exp_runs/sepres_v2_c16n10_20k/checkpoints/latest.pt \\
    --smoke --skip-eval --sessions 0

Does NOT update ``deploy/models.json`` (freeze only).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

_extra = [
    str(Path.home() / "miniforge3/bin"),
    str(Path.home() / "android/platform-tools"),
]
os.environ["PATH"] = os.pathsep.join(_extra + [os.environ.get("PATH", "")])

from models.ecbsr import ECBSR, fuse_ecbsr  # noqa: E402
from models.sepres_v2 import (  # noqa: E402
    SepResV2,
    conv_macs_at_lr,
    count_fused_convs,
    expected_fused_budget,
    fuse_sepres_v2,
    fused_param_count,
)
from utils.model_loader import load_checkpoint_model  # noqa: E402

NCNN_DIR = PROJECT_ROOT / "deploy/artifacts/ncnn"
TS_DIR = PROJECT_ROOT / "deploy/artifacts/torchscript"
RESULTS_DIR = PROJECT_ROOT / "deploy/artifacts/results"
EXP_RESULTS = PROJECT_ROOT / "results/exp_runs"
PARSE_BLOBS = PROJECT_ROOT / "scripts/parse_ncnn_blobs.py"
BENCH_BIN = PROJECT_ROOT / "deploy/android/sr_bench/build/sr_bench"
DEVICE_DIR = "/data/local/tmp/sr_bench"
MODELS_JSON = PROJECT_ROOT / "deploy/models.json"
ENVELOPE_JSON = EXP_RESULTS / "b4_measurement_envelope.json"
COMPARE_JSON = EXP_RESULTS / "b4_v2_compare.json"
ADB = Path.home() / "android/platform-tools/adb"
if not ADB.exists():
    ADB = Path("adb")
LIBOMP = (
    Path.home()
    / "android/ndk/android-ndk-r26d/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/17/lib/linux/aarch64/libomp.so"
)

DEFAULT_V2_RUN = "sepres_v2_c16n10_20k"
DEFAULT_ECBSR_RUN = "ecbsr_m10c16_20k"
REALTIME_MED_MS = 33.3
SIZE_WIN_FRAC = 0.05


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="B4 Gate-2 post-train pipeline")
    p.add_argument("--run-id", default=DEFAULT_V2_RUN)
    p.add_argument("--ecbsr-run-id", default=DEFAULT_ECBSR_RUN)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Override v2 ckpt (default: <run>/checkpoints/best.pt)",
    )
    p.add_argument(
        "--ecbsr-checkpoint",
        type=Path,
        default=None,
        help="Override ECBSR ckpt (default: <ecbsr-run>/checkpoints/best.pt)",
    )
    p.add_argument("--preset", default="deploy_720p", choices=["deploy_720p", "audit_180"])
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--sessions", type=int, default=3, help="Paired phone sessions (0=skip)")
    p.add_argument("--skip-eval", action="store_true", help="Skip Set5/… benchmark eval")
    p.add_argument("--skip-push", action="store_true")
    p.add_argument(
        "--wait",
        action="store_true",
        help="Poll until best.pt exists and train.pid is gone",
    )
    p.add_argument("--wait-poll-sec", type=int, default=60)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Mid-train / pipeline smoke: write *_smoke.json, not official compare",
    )
    p.add_argument("--atol", type=float, default=1e-5)
    return p.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def adb(*args: str, capture: bool = True) -> str:
    cmd = [str(ADB), *args]
    if capture:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    subprocess.check_call(cmd)
    return ""


def adb_ok() -> bool:
    try:
        adb("get-state")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def parse_blobs(param_path: Path) -> tuple[str, str]:
    out = subprocess.check_output(
        [sys.executable, str(PARSE_BLOBS), str(param_path)], text=True
    ).strip()
    return out.split("\t")


def numerical_check(
    eval_model: nn.Module, fused: nn.Module, lr_h: int, lr_w: int, atol: float
) -> dict:
    torch.manual_seed(1)
    x = torch.randn(1, 3, lr_h, lr_w)
    eval_model.eval()
    fused.eval()
    with torch.no_grad():
        y0 = eval_model(x)
        y1 = fused(x)
    abs_err = (y0 - y1).abs()
    return {
        "max_abs": float(abs_err.max()),
        "pass": bool(abs_err.max().item() <= atol),
        "output_shape": list(y0.shape),
    }


def export_torchscript(model: nn.Module, stem: str, lr_h: int, lr_w: int) -> Path:
    TS_DIR.mkdir(parents=True, exist_ok=True)
    out = TS_DIR / f"{stem}.pt"
    dummy = torch.randn(1, 3, lr_h, lr_w)
    model.eval()
    with torch.no_grad():
        torch.jit.trace(model, dummy).save(str(out))
    return out


def convert_pnnx(ts_path: Path, inputshape: str) -> tuple[Path, Path]:
    pnnx = shutil.which("pnnx")
    if not pnnx:
        raise FileNotFoundError("pnnx not on PATH (expected ~/miniforge3/bin/pnnx)")
    subprocess.check_call(
        [pnnx, str(ts_path), f"inputshape={inputshape}", "device=cpu", "fp16=0"],
        cwd=PROJECT_ROOT,
    )
    param = ts_path.with_suffix(".ncnn.param")
    binf = ts_path.with_suffix(".ncnn.bin")
    if not param.exists() or not binf.exists():
        raise RuntimeError(f"PNNX output missing for {ts_path}")
    return param, binf


def ensure_bench(skip_push: bool) -> None:
    if not BENCH_BIN.exists():
        raise SystemExit(f"Missing {BENCH_BIN}")
    adb("shell", f"mkdir -p {DEVICE_DIR}/models", capture=False)
    if not skip_push:
        adb("push", str(BENCH_BIN), f"{DEVICE_DIR}/sr_bench", capture=False)
        adb("shell", f"chmod +x {DEVICE_DIR}/sr_bench", capture=False)
        if LIBOMP.exists():
            adb("push", str(LIBOMP), f"{DEVICE_DIR}/libomp.so", capture=False)


def run_bench_remote(
    remote_param: str,
    remote_bin: str,
    in_blob: str,
    out_blob: str,
    lr_w: int,
    lr_h: int,
    warmup: int,
    iters: int,
) -> dict:
    cmd = [
        f"{DEVICE_DIR}/sr_bench",
        "--param",
        remote_param,
        "--bin",
        remote_bin,
        "--in-blob",
        in_blob,
        "--out-blob",
        out_blob,
        "--input-w",
        str(lr_w),
        "--input-h",
        str(lr_h),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--fp16",
        "--vulkan",
    ]
    raw = adb("shell", f"LD_LIBRARY_PATH={DEVICE_DIR} " + " ".join(cmd))
    return json.loads(raw.strip().splitlines()[-1])


def wait_for_best(run_id: str, poll_sec: int) -> Path:
    run_dir = EXP_RESULTS / run_id
    best = run_dir / "checkpoints/best.pt"
    pid_file = run_dir / "train.pid"
    print(f"Waiting for {rel(best)} and idle train.pid ...")
    while True:
        pid_alive = False
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, 0)
                pid_alive = True
            except (ValueError, OSError, ProcessLookupError):
                pid_alive = False
        if best.exists() and not pid_alive:
            # Prefer completed log (607) if present
            log = run_dir / "train_log.jsonl"
            if log.exists():
                last = json.loads(log.read_text(encoding="utf-8").strip().splitlines()[-1])
                ep = int(last.get("epoch", 0))
                print(f"  ready: best.pt exists, train idle, last_epoch={ep}")
            return best
        status = "training" if pid_alive else "no-pid"
        has_best = "best=yes" if best.exists() else "best=no"
        print(f"  … {status} {has_best}; sleep {poll_sec}s")
        time.sleep(poll_sec)


def load_val_series(run_id: str) -> list[float]:
    log = EXP_RESULTS / run_id / "train_log.jsonl"
    if not log.exists():
        return []
    vals = []
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "val_psnr" in row:
            vals.append(float(row["val_psnr"]))
    return vals


def e_val_from_logs(cand_run: str, ref_run: str, last_n: int = 20) -> dict:
    c = load_val_series(cand_run)
    r = load_val_series(ref_run)
    c_tail = c[-last_n:] if len(c) >= 2 else c
    r_tail = r[-last_n:] if len(r) >= 2 else r

    def two_std(xs: list[float]) -> float:
        if len(xs) < 2:
            return 0.0
        return 2.0 * statistics.pstdev(xs)

    e_c, e_r = two_std(c_tail), two_std(r_tail)
    return {
        "E_val_db": max(e_c, e_r),
        "candidate_last_n": len(c_tail),
        "reference_last_n": len(r_tail),
        "candidate_2std": e_c,
        "reference_2std": e_r,
        "candidate_best_val_psnr": max(c) if c else None,
        "candidate_final_val_psnr": c[-1] if c else None,
        "reference_best_val_psnr": max(r) if r else None,
    }


def fuse_and_export(
    *,
    ckpt: Path,
    stem: str,
    kind: str,
    lr_h: int,
    lr_w: int,
    atol: float,
) -> dict:
    model, cfg = load_checkpoint_model(ckpt, torch.device("cpu"))
    model.eval()
    if kind == "sepres_v2":
        if not isinstance(model, SepResV2):
            raise TypeError(f"expected SepResV2, got {type(model)}")
        fused = fuse_sepres_v2(model)
        c = int(model.num_channel)
        n = int(model.num_block)
        exp = expected_fused_budget(c, n)
        budget = {
            "num_channel": c,
            "num_block": n,
            "fused_convs": count_fused_convs(fused),
            "params": fused_param_count(fused),
            "conv_macs_lr180": conv_macs_at_lr(fused, 180, 180),
            "expected_params": exp["params"],
            "expected_macs": exp["conv_macs_lr180"],
            "expected_convs": exp["fused_convs"],
        }
    elif kind == "ecbsr":
        if not isinstance(model, ECBSR):
            raise TypeError(f"expected ECBSR, got {type(model)}")
        fused = fuse_ecbsr(model)
        budget = {
            "params": sum(p.numel() for p in fused.parameters()),
            "num_block": cfg.get("model", {}).get("num_block"),
            "num_channel": cfg.get("model", {}).get("num_channel")
            or cfg.get("model", {}).get("num_channels"),
        }
    else:
        raise ValueError(kind)

    check = numerical_check(model, fused, lr_h, lr_w, atol)
    if not check["pass"]:
        raise SystemExit(f"fuse numerical failed for {stem}: {check}")

    print(f"  TorchScript {stem} ...")
    ts_path = export_torchscript(fused, stem, lr_h, lr_w)
    print(f"  PNNX {stem} ...")
    param_src, bin_src = convert_pnnx(ts_path, f"[1,3,{lr_h},{lr_w}]")
    NCNN_DIR.mkdir(parents=True, exist_ok=True)
    param = NCNN_DIR / f"{stem}.param"
    binf = NCNN_DIR / f"{stem}.bin"
    shutil.copy2(param_src, param)
    shutil.copy2(bin_src, binf)
    in_blob, out_blob = parse_blobs(param)
    bytes_total = param.stat().st_size + binf.stat().st_size
    return {
        "checkpoint": rel(ckpt),
        "stem": stem,
        "kind": kind,
        "numerical": check,
        "budget": budget,
        "torchscript": rel(ts_path),
        "ncnn_param": rel(param),
        "ncnn_bin": rel(binf),
        "ncnn_bytes": bytes_total,
        "ncnn_total_size_mb": round(bytes_total / 1024**2, 4),
        "in_blob": in_blob,
        "out_blob": out_blob,
        "config_snapshot": cfg.get("model", {}),
    }


def paired_phone_sessions(
    *,
    cand: dict,
    ref: dict,
    sessions: int,
    lr_w: int,
    lr_h: int,
    warmup: int,
    iters: int,
    skip_push: bool,
) -> list[dict]:
    if not adb_ok():
        raise SystemExit("adb unavailable for official phone sessions")
    ensure_bench(skip_push)
    cand_param = PROJECT_ROOT / cand["ncnn_param"]
    cand_bin = PROJECT_ROOT / cand["ncnn_bin"]
    ref_param = PROJECT_ROOT / ref["ncnn_param"]
    ref_bin = PROJECT_ROOT / ref["ncnn_bin"]

    out_sessions: list[dict] = []
    for s in range(sessions):
        order = ["v2", "ecbsr"] if s % 2 == 0 else ["ecbsr", "v2"]
        print(f"\n--- phone session {s + 1}/{sessions} order={' → '.join(order)} ---")
        runs = []
        for key in order:
            meta = cand if key == "v2" else ref
            param = cand_param if key == "v2" else ref_param
            binf = cand_bin if key == "v2" else ref_bin
            remote_p = f"{DEVICE_DIR}/models/{param.name}"
            remote_b = f"{DEVICE_DIR}/models/{binf.name}"
            adb("push", str(param), remote_p, capture=False)
            adb("push", str(binf), remote_b, capture=False)
            bench = run_bench_remote(
                remote_p,
                remote_b,
                meta["in_blob"],
                meta["out_blob"],
                lr_w,
                lr_h,
                warmup,
                iters,
            )
            row = {
                "model_key": key,
                "median_ms": bench["median_ms"],
                "p90_ms": bench["p90_ms"],
                "fps": bench.get("fps"),
                "peak_memory_kb": bench.get("peak_memory_kb"),
                "raw": bench,
            }
            print(f"  {key}: med={row['median_ms']:.2f} p90={row['p90_ms']:.2f}")
            runs.append(row)
        out_sessions.append(
            {
                "session_index": s,
                "order": order,
                "timestamp": datetime.now().astimezone().isoformat(),
                "runs": runs,
            }
        )
    return out_sessions


def summarize_phone(sessions: list[dict]) -> dict:
    by: dict[str, dict] = {"v2": {"median_ms": [], "p90_ms": []}, "ecbsr": {"median_ms": [], "p90_ms": []}}
    for sess in sessions:
        for r in sess["runs"]:
            by[r["model_key"]]["median_ms"].append(r["median_ms"])
            by[r["model_key"]]["p90_ms"].append(r["p90_ms"])

    def pack(xs: list[float]) -> dict:
        return {
            "values": xs,
            "mean": sum(xs) / len(xs) if xs else None,
            "range": (max(xs) - min(xs)) if xs else None,
        }

    return {
        "v2": {"median_ms": pack(by["v2"]["median_ms"]), "p90_ms": pack(by["v2"]["p90_ms"])},
        "ecbsr": {
            "median_ms": pack(by["ecbsr"]["median_ms"]),
            "p90_ms": pack(by["ecbsr"]["p90_ms"]),
        },
    }


def run_benchmark_eval(ckpt: Path, out_json: Path) -> dict | None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/eval_sr.py",
        "--checkpoint",
        rel(ckpt),
        "--save-json",
        rel(out_json),
    ]
    try:
        import lpips  # noqa: F401

        cmd.append("--compute-lpips")
    except ImportError:
        print("  (lpips not installed — PSNR/SSIM only)")
    print("  benchmark eval (Set5/14/BSD100/Urban100) ...")
    rc = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
    if rc.returncode != 0 or not out_json.exists():
        return None
    metrics = json.loads(out_json.read_text(encoding="utf-8"))
    psnrs = [v["psnr"] for v in metrics.values() if isinstance(v, dict) and "psnr" in v]
    return {
        "avg_psnr": sum(psnrs) / len(psnrs) if psnrs else None,
        "metrics": metrics,
        "path": rel(out_json),
    }


def decide(
    *,
    e_med: float,
    e_p90: float,
    e_val: float,
    cand_val: float | None,
    ref_val: float | None,
    phone: dict | None,
    cand_bytes: int,
    ref_bytes: int,
) -> dict:
    """Envelope-aware dominance vs ECBSR (unique main anchor)."""

    def cmp_higher(a: float | None, b: float | None, eps: float) -> str:
        if a is None or b is None:
            return "unknown"
        d = a - b
        if abs(d) <= eps:
            return "tie"
        return "cand_better" if d > 0 else "ref_better"

    def cmp_lower(a: float | None, b: float | None, eps: float) -> str:
        if a is None or b is None:
            return "unknown"
        d = a - b
        if abs(d) <= eps:
            return "tie"
        return "cand_better" if d < 0 else "ref_better"

    axes: dict[str, str] = {}
    axes["val_psnr"] = cmp_higher(cand_val, ref_val, e_val)

    if phone:
        c_med = phone["v2"]["median_ms"]["mean"]
        r_med = phone["ecbsr"]["median_ms"]["mean"]
        c_p90 = phone["v2"]["p90_ms"]["mean"]
        r_p90 = phone["ecbsr"]["p90_ms"]["mean"]
        axes["phone_median"] = cmp_lower(c_med, r_med, e_med)
        axes["phone_p90"] = cmp_lower(c_p90, r_p90, e_p90)
        realtime_ok = c_med is not None and c_med <= REALTIME_MED_MS
        stable_ok = c_p90 is not None and c_p90 <= REALTIME_MED_MS
    else:
        axes["phone_median"] = "skipped"
        axes["phone_p90"] = "skipped"
        realtime_ok = None
        stable_ok = None

    size_frac = (ref_bytes - cand_bytes) / ref_bytes if ref_bytes else 0.0
    if abs(size_frac) < SIZE_WIN_FRAC:
        axes["ncnn_bytes"] = "tie"
    else:
        axes["ncnn_bytes"] = "cand_better" if size_frac > 0 else "ref_better"

    scored = [v for v in axes.values() if v in ("cand_better", "ref_better", "tie")]
    cand_wins = sum(1 for v in scored if v == "cand_better")
    ref_wins = sum(1 for v in scored if v == "ref_better")
    ties = sum(1 for v in scored if v == "tie")

    if ref_wins >= 1 and cand_wins == 0 and "unknown" not in axes.values() and "skipped" not in (
        axes["phone_median"],
        axes["phone_p90"],
    ):
        # ECBSR not worse on any scored axis and better on ≥1 → dominated
        # (size-only tie-heavy cases fall through)
        verdict = "dominated_exit_ecbsr"
    elif cand_wins >= 1 and ref_wins == 0:
        verdict = "meaningful_nondominated_provisional_freeze"
    elif cand_wins >= 1 and ref_wins >= 1:
        verdict = "tradeoff_nondominated_provisional_freeze"
    elif all(v == "tie" for v in scored) or (cand_wins == 0 and ref_wins == 0):
        verdict = "tie_continue_or_exit"
    else:
        verdict = "incomplete_or_manual_review"

    # Size-only unique win: require meaningful frac (already in axis)
    if (
        verdict.startswith("meaningful")
        and cand_wins == 1
        and axes.get("ncnn_bytes") == "cand_better"
        and axes.get("val_psnr") == "tie"
        and axes.get("phone_median") == "tie"
    ):
        # allowed by Spec if ≥5%
        pass

    return {
        "axes": axes,
        "cand_wins": cand_wins,
        "ref_wins": ref_wins,
        "ties": ties,
        "size_save_frac": size_frac,
        "realtime_median_ok": realtime_ok,
        "stability_p90_ok": stable_ok,
        "verdict": verdict,
        "note": (
            "Differences inside E_val / E_med / E_p90 are ties. "
            "Median≤33.3ms is real-time budget; p90 needed for stability claim. "
            "Do not update models.json until freeze is written to checklist."
        ),
    }


def main() -> None:
    args = parse_args()
    registry = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    preset = {p["name"]: p for p in registry["input_presets"]}[args.preset]
    lr_w, lr_h = int(preset["lr_w"]), int(preset["lr_h"])

    if args.wait:
        v2_ckpt = wait_for_best(args.run_id, args.wait_poll_sec)
    else:
        v2_ckpt = (
            args.checkpoint
            if args.checkpoint
            else EXP_RESULTS / args.run_id / "checkpoints/best.pt"
        )
        if not v2_ckpt.is_absolute():
            v2_ckpt = PROJECT_ROOT / v2_ckpt
    if not v2_ckpt.exists():
        raise SystemExit(
            f"Missing v2 checkpoint: {v2_ckpt}\n"
            "Train still running? Use --wait, or --smoke with --checkpoint .../latest.pt"
        )

    ecbsr_ckpt = (
        args.ecbsr_checkpoint
        if args.ecbsr_checkpoint
        else EXP_RESULTS / args.ecbsr_run_id / "checkpoints/best.pt"
    )
    if not ecbsr_ckpt.is_absolute():
        ecbsr_ckpt = PROJECT_ROOT / ecbsr_ckpt
    if not ecbsr_ckpt.exists():
        raise SystemExit(f"Missing ECBSR checkpoint: {ecbsr_ckpt}")

    if not ENVELOPE_JSON.exists() and not args.smoke:
        raise SystemExit(f"Missing envelope: {ENVELOPE_JSON}")

    mode = "graph_smoke" if args.smoke else "trained_official"
    suffix = "_smoke" if args.smoke else ""
    v2_stem = f"sepres_v2_c16n10_fused_{args.preset}{suffix}"
    ecbsr_stem = f"ecbsr_m10c16_fused_{args.preset}{suffix if args.smoke else ''}"
    # Official ECBSR stem without dryrun suffix
    if not args.smoke:
        ecbsr_stem = f"ecbsr_m10c16_fused_{args.preset}"

    print("=== B4 Gate-2 post-train ===")
    print(f"  mode={mode}")
    print(f"  v2_ckpt={rel(v2_ckpt)}")
    print(f"  ecbsr_ckpt={rel(ecbsr_ckpt)}")
    print(f"  preset={args.preset} LR {lr_w}x{lr_h}")

    print("\n[1/4] fuse + export v2")
    v2_export = fuse_and_export(
        ckpt=v2_ckpt, stem=v2_stem, kind="sepres_v2", lr_h=lr_h, lr_w=lr_w, atol=args.atol
    )
    print(
        f"  fuse ok max_abs={v2_export['numerical']['max_abs']:.3e} "
        f"ncnn={v2_export['ncnn_total_size_mb']} MB"
    )

    print("\n[2/4] fuse + export ECBSR (paired identity)")
    ecbsr_export = fuse_and_export(
        ckpt=ecbsr_ckpt,
        stem=ecbsr_stem,
        kind="ecbsr",
        lr_h=lr_h,
        lr_w=lr_w,
        atol=args.atol,
    )
    print(
        f"  fuse ok max_abs={ecbsr_export['numerical']['max_abs']:.3e} "
        f"ncnn={ecbsr_export['ncnn_total_size_mb']} MB"
    )

    phone_sessions: list[dict] = []
    phone_summary = None
    if args.sessions > 0:
        print(f"\n[3/4] paired phone ({args.sessions} sessions)")
        phone_sessions = paired_phone_sessions(
            cand=v2_export,
            ref=ecbsr_export,
            sessions=args.sessions,
            lr_w=lr_w,
            lr_h=lr_h,
            warmup=args.warmup,
            iters=args.iters,
            skip_push=args.skip_push,
        )
        phone_summary = summarize_phone(phone_sessions)
        # Timestamped phone JSON
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        phone_path = RESULTS_DIR / f"sepres_v2_paired_phone_{ts}.json"
        phone_payload = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "task": "B4_v2_paired_phone",
            "mode": mode,
            "protocol": {
                "warmup": args.warmup,
                "iters": args.iters,
                "lr_w": lr_w,
                "lr_h": lr_h,
                "backend": "ncnn_vulkan_fp16",
            },
            "sessions": phone_sessions,
            "summary": phone_summary,
        }
        phone_path.write_text(json.dumps(phone_payload, indent=2), encoding="utf-8")
        latest_phone = RESULTS_DIR / "sepres_v2_paired_phone_latest.json"
        latest_phone.write_text(phone_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"  wrote {rel(phone_path)}")
    else:
        print("\n[3/4] phone skipped (--sessions 0)")

    bench = None
    if not args.skip_eval and not args.smoke:
        print("\n[4/4] quality eval")
        bench = run_benchmark_eval(
            v2_ckpt, EXP_RESULTS / args.run_id / "benchmark_metrics.json"
        )
    else:
        print("\n[4/4] quality eval skipped")

    envelope = {}
    if ENVELOPE_JSON.exists():
        envelope = json.loads(ENVELOPE_JSON.read_text(encoding="utf-8"))
    e_med = float(envelope.get("E_med_ms", 0.0))
    e_p90 = float(envelope.get("E_p90_ms", 0.0))
    e_pack = e_val_from_logs(args.run_id, args.ecbsr_run_id)
    e_val = float(e_pack["E_val_db"])

    # Prefer train-log best val for selection axis; bench avg is report-only.
    cand_val = e_pack.get("candidate_best_val_psnr")
    ref_val = e_pack.get("reference_best_val_psnr")

    decision = decide(
        e_med=e_med,
        e_p90=e_p90,
        e_val=e_val,
        cand_val=cand_val,
        ref_val=ref_val,
        phone=phone_summary,
        cand_bytes=int(v2_export["ncnn_bytes"]),
        ref_bytes=int(ecbsr_export["ncnn_bytes"]),
    )

    payload = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "task": "B4_v2_compare",
        "gate": "Gate-2-posttrain",
        "mode": mode,
        "official": not args.smoke,
        "run_id": args.run_id,
        "ecbsr_run_id": args.ecbsr_run_id,
        "measurement_kind": mode,
        "envelope": {
            "path": rel(ENVELOPE_JSON) if ENVELOPE_JSON.exists() else None,
            "E_med_ms": e_med,
            "E_p90_ms": e_p90,
        },
        "E_val": e_pack,
        "exports": {"v2": v2_export, "ecbsr": ecbsr_export},
        "phone_sessions": phone_sessions,
        "phone_summary": phone_summary,
        "benchmark": bench,
        "decision": decision,
        "next_gate": {
            "if_v2_b_weak": "consider v2_a only if speed/size headroom beyond envelope; v2_c only if quality hopeful",
            "if_dominated": "Q4 Exit freeze_ref=ECBSR-M10C16",
            "do_not": "update deploy/models.json until checklist freeze_ref written",
        },
    }

    EXP_RESULTS.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        out = EXP_RESULTS / "b4_v2_posttrain_smoke.json"
    else:
        out = COMPARE_JSON
        # also timestamped copy
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        (EXP_RESULTS / f"b4_v2_compare_{ts}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {rel(out)}")
    print(f"Decision: {decision['verdict']}")
    print(f"  axes={decision['axes']}")
    if args.smoke:
        print("SMOKE only — re-run without --smoke after best.pt for official compare.")


if __name__ == "__main__":
    main()
