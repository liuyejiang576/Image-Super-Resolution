#!/usr/bin/env python3
"""B4 round-2 Gate-0: DualStream→Plain fuse → TorchScript → PNNX → NCNN.

Latency (if phone) is ``graph_smoke`` only — not for D18 phone gate or freeze.

  export PATH=$HOME/miniforge3/bin:$HOME/android/platform-tools:$PATH
  python scripts/prescreen_dual_plain.py --skip-bench
  python scripts/prescreen_dual_plain.py
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
import yaml

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
from utils.model_loader import build_model_from_config  # noqa: E402

NCNN_DIR = PROJECT_ROOT / "deploy/artifacts/ncnn"
TS_DIR = PROJECT_ROOT / "deploy/artifacts/torchscript"
EXP_RESULTS = PROJECT_ROOT / "results/exp_runs"
PARSE_BLOBS = PROJECT_ROOT / "scripts/parse_ncnn_blobs.py"
BENCH_BIN = PROJECT_ROOT / "deploy/android/sr_bench/build/sr_bench"
DEVICE_DIR = "/data/local/tmp/sr_bench"
ADB = Path.home() / "android/platform-tools/adb"
if not ADB.exists():
    ADB = Path("adb")
LIBOMP = (
    Path.home()
    / "android/ndk/android-ndk-r26d/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/17/lib/linux/aarch64/libomp.so"
)

GROSS_MED_MS = 38.0
INIT_SEED = 0
ATOL = 1e-5
OUT = EXP_RESULTS / "b4_dual_plain_prescreen.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="deploy_720p", choices=["deploy_720p", "audit_180"])
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--skip-bench", action="store_true")
    p.add_argument("--skip-push", action="store_true")
    p.add_argument("--atol", type=float, default=ATOL)
    return p.parse_args()


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
        raise FileNotFoundError("pnnx not on PATH")
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


def adb_ok() -> bool:
    try:
        adb("get-state")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def main() -> int:
    args = parse_args()
    lr_h, lr_w = (180, 180) if args.preset == "audit_180" else (180, 320)
    torch.manual_seed(INIT_SEED)
    errors: list[str] = []

    dual_cfg = yaml.safe_load(
        (PROJECT_ROOT / "configs/exp/dual_stream_c20n5_2k.yaml").read_text(encoding="utf-8")
    )
    plain_cfg = yaml.safe_load(
        (PROJECT_ROOT / "configs/exp/plain_c20n5_2k.yaml").read_text(encoding="utf-8")
    )
    dual = build_model_from_config(dual_cfg)
    plain = build_model_from_config(plain_cfg)
    assert isinstance(dual, DualStreamSR)
    assert isinstance(plain, PlainSR)
    dual.eval()
    plain.eval()
    fused = fuse_dual_stream_sr(dual)

    x = torch.randn(1, 3, lr_h, lr_w)
    with torch.no_grad():
        y0, y1 = dual(x), fused(x)
    abs_err = float((y0 - y1).abs().max())
    if abs_err > args.atol:
        errors.append(f"fuse max_abs={abs_err}")

    budget = expected_plain_budget(20, 5)
    params = plain_param_count(fused)
    convs = count_plain_convs(fused)
    macs = conv_macs_at_lr(fused)
    if params != budget["params"] or convs != budget["fused_convs"] or macs != budget["macs_180"]:
        errors.append("budget mismatch")

    stem = "dual_plain_c20n5_graph_smoke_720p"
    print(f"TorchScript {stem} ...", flush=True)
    ts = export_torchscript(fused, stem, lr_h, lr_w)
    print("PNNX ...", flush=True)
    param_src, bin_src = convert_pnnx(ts, f"[1,3,{lr_h},{lr_w}]")
    NCNN_DIR.mkdir(parents=True, exist_ok=True)
    param = NCNN_DIR / f"{stem}.param"
    binf = NCNN_DIR / f"{stem}.bin"
    shutil.copy2(param_src, param)
    shutil.copy2(bin_src, binf)
    in_blob, out_blob = parse_blobs(param)
    bytes_total = param.stat().st_size + binf.stat().st_size

    phone: dict | None = None
    if not args.skip_bench and adb_ok():
        adb("shell", f"mkdir -p {DEVICE_DIR}/models", capture=False)
        if not args.skip_push and BENCH_BIN.exists():
            adb("push", str(BENCH_BIN), f"{DEVICE_DIR}/sr_bench", capture=False)
            adb("shell", f"chmod +x {DEVICE_DIR}/sr_bench", capture=False)
            if LIBOMP.exists():
                adb("push", str(LIBOMP), f"{DEVICE_DIR}/libomp.so", capture=False)
        adb("push", str(param), f"{DEVICE_DIR}/models/{stem}.param", capture=False)
        adb("push", str(binf), f"{DEVICE_DIR}/models/{stem}.bin", capture=False)
        cmd = (
            f"LD_LIBRARY_PATH={DEVICE_DIR} {DEVICE_DIR}/sr_bench "
            f"--param {DEVICE_DIR}/models/{stem}.param "
            f"--bin {DEVICE_DIR}/models/{stem}.bin "
            f"--in-blob {in_blob} --out-blob {out_blob} "
            f"--input-w {lr_w} --input-h {lr_h} "
            f"--warmup {args.warmup} --iters {args.iters} --fp16 --vulkan"
        )
        raw = adb("shell", cmd)
        phone = json.loads(raw.strip().splitlines()[-1])
        med = float(phone.get("median_ms", 1e9))
        if med > GROSS_MED_MS:
            errors.append(f"graph_smoke med={med:.2f}>{GROSS_MED_MS}")
    elif not args.skip_bench:
        print("adb unavailable — export only", flush=True)

    # Also export Plain random-init (same deploy graph) for packing check
    stem_p = "plain_c20n5_graph_smoke_720p"
    ts_p = export_torchscript(plain, stem_p, lr_h, lr_w)
    param_p, bin_p = convert_pnnx(ts_p, f"[1,3,{lr_h},{lr_w}]")
    shutil.copy2(param_p, NCNN_DIR / f"{stem_p}.param")
    shutil.copy2(bin_p, NCNN_DIR / f"{stem_p}.bin")

    ok = not errors
    report = {
        "task": "b4_dual_plain_prescreen",
        "timestamp": datetime.now().astimezone().isoformat(),
        "pass": ok,
        "errors": errors,
        "measurement_kind": "graph_smoke",
        "fuse_max_abs": abs_err,
        "atol": args.atol,
        "budget": {"params": params, "convs": convs, "macs_180": macs, "expected": budget},
        "exports": {
            "dual_fused": {
                "stem": stem,
                "torchscript": str(ts.relative_to(PROJECT_ROOT)),
                "ncnn_param": str(param.relative_to(PROJECT_ROOT)),
                "ncnn_bin": str(binf.relative_to(PROJECT_ROOT)),
                "bytes": bytes_total,
                "in_blob": in_blob,
                "out_blob": out_blob,
            },
            "plain": {"stem": stem_p},
        },
        "phone": phone,
        "note": "Not for D18 phone gate or freeze. Official phone after 2k via run_dual_plain_posttrain.py",
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"{'PASS' if ok else 'FAIL'} → {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
