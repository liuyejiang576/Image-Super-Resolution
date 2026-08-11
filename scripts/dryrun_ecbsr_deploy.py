#!/usr/bin/env python3
"""B3 dry-run: fuse ECBSR → TorchScript → PNNX → NCNN (+ optional phone smoke).

Does NOT touch deploy/models.json headline identity.
Uses mid-train latest.pt by default — numbers are pipeline smoke only, not B3 results.

  export PATH=$HOME/miniforge3/bin:$HOME/android/platform-tools:$PATH
  python scripts/dryrun_ecbsr_deploy.py --preset deploy_720p
  python scripts/dryrun_ecbsr_deploy.py --skip-bench   # convert only
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
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Prefer lab pnnx / adb on PATH for subprocesses.
_extra = [
    str(Path.home() / "miniforge3/bin"),
    str(Path.home() / "android/platform-tools"),
]
os.environ["PATH"] = os.pathsep.join(_extra + [os.environ.get("PATH", "")])

from models.ecbsr import ECBSR, fuse_ecbsr  # noqa: E402
from utils.model_loader import load_checkpoint_model  # noqa: E402

DEFAULT_CKPT = (
    PROJECT_ROOT / "results/exp_runs/ecbsr_m10c16_20k/checkpoints/latest.pt"
)
NCNN_DIR = PROJECT_ROOT / "deploy/artifacts/ncnn"
TS_DIR = PROJECT_ROOT / "deploy/artifacts/torchscript"
ONNX_DIR = PROJECT_ROOT / "deploy/artifacts/onnx"
RESULTS_DIR = PROJECT_ROOT / "deploy/artifacts/results"
PARSE_BLOBS = PROJECT_ROOT / "scripts/parse_ncnn_blobs.py"
BENCH_BIN = PROJECT_ROOT / "deploy/android/sr_bench/build/sr_bench"
DEVICE_DIR = "/data/local/tmp/sr_bench"
MODELS_JSON = PROJECT_ROOT / "deploy/models.json"
ADB = Path.home() / "android/platform-tools/adb"
if not ADB.exists():
    ADB = Path("adb")
LIBOMP = (
    Path.home()
    / "android/ndk/android-ndk-r26d/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/17/lib/linux/aarch64/libomp.so"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ECBSR deploy pipeline dry-run (B3)")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--preset", default="deploy_720p", choices=["deploy_720p", "audit_180"])
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=50, help="Smoke iters (not A0 300)")
    p.add_argument("--skip-bench", action="store_true")
    p.add_argument("--skip-push", action="store_true")
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--rtol", type=float, default=1e-4)
    return p.parse_args()


def numerical_check(
    eval_model: nn.Module, fused: nn.Module, lr_h: int, lr_w: int, atol: float, rtol: float
) -> dict:
    x = torch.randn(1, 3, lr_h, lr_w)
    eval_model.eval()
    fused.eval()
    with torch.no_grad():
        y0 = eval_model(x)
        y1 = fused(x)
    abs_err = (y0 - y1).abs()
    rel = abs_err / y0.abs().clamp_min(1e-8)
    return {
        "max_abs": float(abs_err.max()),
        "max_rel": float(rel.max()),
        "pass": bool(torch.allclose(y0, y1, atol=atol, rtol=rtol)),
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


def export_onnx(model: nn.Module, stem: str, lr_h: int, lr_w: int) -> Path:
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    out = ONNX_DIR / f"{stem}.onnx"
    dummy = torch.randn(1, 3, lr_h, lr_w)
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(out),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["lr"],
            output_names=["sr"],
        )
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


def parse_blobs(param_path: Path) -> tuple[str, str]:
    out = subprocess.check_output(
        [sys.executable, str(PARSE_BLOBS), str(param_path)], text=True
    ).strip()
    return out.split("\t")


def adb(*args: str, capture: bool = True) -> str:
    cmd = [str(ADB), *args]
    if capture:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    subprocess.check_call(cmd)
    return ""


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
        "--param", remote_param,
        "--bin", remote_bin,
        "--in-blob", in_blob,
        "--out-blob", out_blob,
        "--input-w", str(lr_w),
        "--input-h", str(lr_h),
        "--warmup", str(warmup),
        "--iters", str(iters),
        "--fp16",
        "--vulkan",
    ]
    raw = adb("shell", f"LD_LIBRARY_PATH={DEVICE_DIR} " + " ".join(cmd))
    return json.loads(raw.strip().splitlines()[-1])


def main() -> None:
    args = parse_args()
    ckpt = args.checkpoint if args.checkpoint.is_absolute() else PROJECT_ROOT / args.checkpoint
    if not ckpt.exists():
        raise SystemExit(f"Missing checkpoint: {ckpt}")

    registry = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    presets = {p["name"]: p for p in registry["input_presets"]}
    preset = presets[args.preset]
    lr_w, lr_h = int(preset["lr_w"]), int(preset["lr_h"])
    inputshape = f"[1,3,{lr_h},{lr_w}]"
    stem = f"ecbsr_m10c16_fused_{args.preset}_dryrun"

    print(f"=== B3 dry-run ECBSR deploy ===")
    print(f"  ckpt: {ckpt.relative_to(PROJECT_ROOT)}")
    print(f"  preset: {args.preset} LR {lr_w}x{lr_h}")
    print(f"  NOTE: mid-train weights OK for pipeline smoke; not B3 headline numbers.")

    model, cfg = load_checkpoint_model(ckpt, torch.device("cpu"))
    if not isinstance(model, ECBSR):
        raise TypeError(f"expected ECBSR, got {type(model)}")
    model.eval()
    fused = fuse_ecbsr(model)
    check = numerical_check(model, fused, lr_h, lr_w, args.atol, args.rtol)
    print(
        f"  fuse numerical: pass={check['pass']} max_abs={check['max_abs']:.3e} "
        f"max_rel={check['max_rel']:.3e}"
    )
    if not check["pass"]:
        raise SystemExit("fuse numerical check failed")

    params = sum(p.numel() for p in fused.parameters())
    print(f"  fused params: {params}")

    print("  TorchScript ...")
    ts_path = export_torchscript(fused, stem, lr_h, lr_w)
    # Official NCNN path is PNNX←TorchScript (same as B1). ONNX optional / env-dependent.
    print("  PNNX ...")
    param_src, bin_src = convert_pnnx(ts_path, inputshape)
    NCNN_DIR.mkdir(parents=True, exist_ok=True)
    param = NCNN_DIR / f"{stem}.param"
    binf = NCNN_DIR / f"{stem}.bin"
    shutil.copy2(param_src, param)
    shutil.copy2(bin_src, binf)
    in_blob, out_blob = parse_blobs(param)
    ncnn_mb = (param.stat().st_size + binf.stat().st_size) / 1024**2
    print(f"  NCNN: {param.name} blobs {in_blob}->{out_blob} size={ncnn_mb:.3f} MB")

    row: dict = {
        "model_id": "ecbsr_m10c16",
        "label": "ECBSR-M10C16 (fused)",
        "variant": "fused",
        "dry_run": True,
        "checkpoint": str(ckpt.relative_to(PROJECT_ROOT)),
        "ckpt_note": "mid-train or non-final; not B3 official",
        "preset": args.preset,
        "lr_w": lr_w,
        "lr_h": lr_h,
        "params": params,
        "numerical": check,
        "torchscript": str(ts_path.relative_to(PROJECT_ROOT)),
        "onnx": None,
        "ncnn_param": str(param.relative_to(PROJECT_ROOT)),
        "ncnn_bin": str(binf.relative_to(PROJECT_ROOT)),
        "ncnn_total_size_mb": round(ncnn_mb, 4),
        "convert_method": "pnnx",
        "in_blob": in_blob,
        "out_blob": out_blob,
        "config_snapshot": {
            "num_block": cfg.get("model", {}).get("num_block"),
            "num_channel": cfg.get("model", {}).get("num_channel"),
            "with_idt": cfg.get("model", {}).get("with_idt"),
            "act_type": cfg.get("model", {}).get("act_type"),
        },
    }

    if not args.skip_bench:
        try:
            adb("get-state")
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"  skip bench: adb unavailable ({exc})")
        else:
            if not BENCH_BIN.exists():
                raise SystemExit(f"Missing {BENCH_BIN}")
            adb("shell", f"mkdir -p {DEVICE_DIR}/models", capture=False)
            if not args.skip_push:
                adb("push", str(BENCH_BIN), f"{DEVICE_DIR}/sr_bench", capture=False)
                adb("shell", f"chmod +x {DEVICE_DIR}/sr_bench", capture=False)
            if LIBOMP.exists():
                adb("push", str(LIBOMP), f"{DEVICE_DIR}/libomp.so", capture=False)
            remote_param = f"{DEVICE_DIR}/models/{stem}.param"
            remote_bin = f"{DEVICE_DIR}/models/{stem}.bin"
            print("  phone smoke bench ...")
            adb("push", str(param), remote_param, capture=False)
            adb("push", str(binf), remote_bin, capture=False)
            bench = run_bench_remote(
                remote_param,
                remote_bin,
                in_blob,
                out_blob,
                lr_w,
                lr_h,
                args.warmup,
                args.iters,
            )
            print(
                f"  median={bench['median_ms']:.2f} ms  p90={bench['p90_ms']:.2f} ms  "
                f"(smoke iters={args.iters}; discard for B3 table)"
            )
            row["phone_smoke"] = bench

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"ecbsr_deploy_dryrun_{ts}.json"
    latest = RESULTS_DIR / "ecbsr_deploy_dryrun_latest.json"
    payload = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "task": "B3_ecbsr_deploy_dryrun",
        "official_b3_numbers": False,
        "result": row,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"\nWrote {out_path.relative_to(PROJECT_ROOT)}")
    print("Pipeline OK. After train: re-run with best.pt for official B3 bench.")


if __name__ == "__main__":
    main()
