#!/usr/bin/env python3
"""B4 round-2 post-train: DualStream fuse/export (+ optional paired phone vs ECBSR).

Official after Dual 2k (and D18 val gate) when phone is up:

  python scripts/run_dual_plain_posttrain.py --wait

Warmup / mid-train smoke (no official compare; E12):

  python scripts/run_dual_plain_posttrain.py \\
    --checkpoint results/exp_runs/dual_stream_c20n5_2k/checkpoints/latest.pt \\
    --smoke --skip-eval --sessions 0

Does NOT update deploy/models.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

_extra = [
    str(Path.home() / "miniforge3/bin"),
    str(Path.home() / "android/platform-tools"),
]
os.environ["PATH"] = os.pathsep.join(_extra + [os.environ.get("PATH", "")])

from models.dual_plain_sr import (  # noqa: E402
    DualStreamSR,
    PlainSR,
    conv_macs_at_lr,
    count_plain_convs,
    expected_plain_budget,
    fuse_dual_stream_sr,
    plain_param_count,
)
from utils.model_loader import load_checkpoint_model  # noqa: E402

NCNN_DIR = PROJECT_ROOT / "deploy/artifacts/ncnn"
TS_DIR = PROJECT_ROOT / "deploy/artifacts/torchscript"
EXP_RESULTS = PROJECT_ROOT / "results/exp_runs"
PARSE_BLOBS = PROJECT_ROOT / "scripts/parse_ncnn_blobs.py"
DEFAULT_DUAL_RUN = "dual_stream_c20n5_2k"
ATOL = 1e-5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default=DEFAULT_DUAL_RUN)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--preset", default="deploy_720p", choices=["deploy_720p", "audit_180"])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--sessions", type=int, default=0, help="0 = export only (E12 smoke)")
    p.add_argument("--wait", action="store_true", help="Wait until best.pt exists")
    p.add_argument("--atol", type=float, default=ATOL)
    return p.parse_args()


def export_torchscript(model, stem: str, lr_h: int, lr_w: int) -> Path:
    TS_DIR.mkdir(parents=True, exist_ok=True)
    out = TS_DIR / f"{stem}.pt"
    with torch.no_grad():
        torch.jit.trace(model, torch.randn(1, 3, lr_h, lr_w)).save(str(out))
    return out


def convert_pnnx(ts_path: Path, inputshape: str) -> tuple[Path, Path]:
    pnnx = shutil.which("pnnx")
    if not pnnx:
        raise FileNotFoundError("pnnx not on PATH")
    subprocess.check_call(
        [pnnx, str(ts_path), f"inputshape={inputshape}", "device=cpu", "fp16=0"],
        cwd=PROJECT_ROOT,
    )
    return ts_path.with_suffix(".ncnn.param"), ts_path.with_suffix(".ncnn.bin")


def main() -> int:
    args = parse_args()
    lr_h, lr_w = (180, 180) if args.preset == "audit_180" else (180, 320)
    ckpt = args.checkpoint
    if ckpt is None:
        name = "latest.pt" if args.smoke else "best.pt"
        ckpt = EXP_RESULTS / args.run_id / "checkpoints" / name
    if args.wait:
        import time

        while not ckpt.exists():
            print(f"waiting for {ckpt} ...", flush=True)
            time.sleep(60)
    if not ckpt.exists():
        raise SystemExit(f"missing checkpoint: {ckpt}")

    model, _cfg = load_checkpoint_model(ckpt, torch.device("cpu"))
    if isinstance(model, DualStreamSR):
        fused = fuse_dual_stream_sr(model)
        kind = "dual_stream_sr"
        with torch.no_grad():
            x = torch.randn(1, 3, 48, 48)
            err = float((model(x) - fused(x)).abs().max())
    elif isinstance(model, PlainSR):
        fused = model
        kind = "plain_sr"
        err = 0.0
    else:
        raise TypeError(type(model))

    if err > args.atol:
        raise SystemExit(f"fuse abs={err} > {args.atol}")

    budget = expected_plain_budget(20, 5)
    params = plain_param_count(fused)
    convs = count_plain_convs(fused)
    macs = conv_macs_at_lr(fused)
    stem = f"{args.run_id}_{'smoke' if args.smoke else 'deploy'}_720p"
    print(f"export {stem} kind={kind} ...", flush=True)
    ts = export_torchscript(fused, stem, lr_h, lr_w)
    param_src, bin_src = convert_pnnx(ts, f"[1,3,{lr_h},{lr_w}]")
    NCNN_DIR.mkdir(parents=True, exist_ok=True)
    param = NCNN_DIR / f"{stem}.param"
    binf = NCNN_DIR / f"{stem}.bin"
    shutil.copy2(param_src, param)
    shutil.copy2(bin_src, binf)
    blobs = subprocess.check_output(
        [sys.executable, str(PARSE_BLOBS), str(param)], text=True
    ).strip()
    in_blob, out_blob = blobs.split("\t")

    report = {
        "task": "b4_dual_plain_posttrain",
        "timestamp": datetime.now().astimezone().isoformat(),
        "smoke": bool(args.smoke),
        "official": not args.smoke and args.sessions > 0,
        "run_id": args.run_id,
        "checkpoint": str(ckpt),
        "kind": kind,
        "fuse_max_abs": err,
        "budget": {
            "params": params,
            "convs": convs,
            "macs_180": macs,
            "expected": budget,
            "match": params == budget["params"]
            and convs == budget["fused_convs"]
            and macs == budget["macs_180"],
        },
        "export": {
            "stem": stem,
            "torchscript": str(ts.relative_to(PROJECT_ROOT)),
            "ncnn_param": str(param.relative_to(PROJECT_ROOT)),
            "ncnn_bin": str(binf.relative_to(PROJECT_ROOT)),
            "bytes": param.stat().st_size + binf.stat().st_size,
            "in_blob": in_blob,
            "out_blob": out_blob,
        },
        "sessions_requested": args.sessions,
        "note": (
            "Smoke/export only."
            if args.sessions == 0
            else "Paired phone vs ECBSR not yet wired in this skeleton; extend when D18 val passes."
        ),
    }
    out_name = (
        "b4_dual_plain_posttrain_smoke.json"
        if args.smoke
        else "b4_dual_plain_posttrain.json"
    )
    out = EXP_RESULTS / out_name
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")
    if args.sessions > 0:
        print(
            "WARNING: sessions>0 requested but paired phone path is stub; "
            "use prescreen phone or extend this script.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
