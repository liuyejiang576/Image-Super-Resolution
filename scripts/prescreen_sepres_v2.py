#!/usr/bin/env python3
"""B4 Gate-0: SepResV2 random-init fuse → TorchScript → PNNX → NCNN (+ phone smoke).

Does NOT touch deploy/models.json. Latency is labelled ``graph_smoke`` only —
not for close ranking, 33.3 ms claims, or freeze.

  export PATH=$HOME/miniforge3/bin:$HOME/android/platform-tools:$PATH
  python scripts/prescreen_sepres_v2.py
  python scripts/prescreen_sepres_v2.py --skip-bench
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

from models.sepres_v2 import (  # noqa: E402
    SepResV2,
    conv_macs_at_lr,
    count_fused_convs,
    expected_fused_budget,
    fuse_sepres_v2,
    fused_param_count,
)
from utils.model_loader import build_model_from_config  # noqa: E402

NCNN_DIR = PROJECT_ROOT / "deploy/artifacts/ncnn"
TS_DIR = PROJECT_ROOT / "deploy/artifacts/torchscript"
RESULTS_DIR = PROJECT_ROOT / "deploy/artifacts/results"
EXP_RESULTS = PROJECT_ROOT / "results/exp_runs"
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

CANDIDATES = [
    ("v2_a", "configs/exp/sepres_v2_c16n8_20k.yaml", 16, 8),
    ("v2_b", "configs/exp/sepres_v2_c16n10_20k.yaml", 16, 10),
    ("v2_c", "configs/exp/sepres_v2_c20n6_20k.yaml", 20, 6),
]

# Gross latency gate for random-weight smoke (IMPLEMENTATION / track_b).
GROSS_MED_MS = 38.0
INIT_SEED = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="B4 SepResV2 Gate-0 export + phone smoke")
    p.add_argument("--preset", default="deploy_720p", choices=["deploy_720p", "audit_180"])
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=50, help="graph_smoke iters (not official 300)")
    p.add_argument("--skip-bench", action="store_true")
    p.add_argument("--skip-push", action="store_true")
    p.add_argument("--candidate", choices=["v2_a", "v2_b", "v2_c", "all"], default="all")
    p.add_argument("--atol", type=float, default=1e-5)
    return p.parse_args()


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


def ensure_bench_on_device(skip_push: bool) -> None:
    if not BENCH_BIN.exists():
        raise SystemExit(f"Missing {BENCH_BIN}")
    adb("shell", f"mkdir -p {DEVICE_DIR}/models", capture=False)
    if not skip_push:
        adb("push", str(BENCH_BIN), f"{DEVICE_DIR}/sr_bench", capture=False)
        adb("shell", f"chmod +x {DEVICE_DIR}/sr_bench", capture=False)
        if LIBOMP.exists():
            adb("push", str(LIBOMP), f"{DEVICE_DIR}/libomp.so", capture=False)


def prescreen_one(
    cand_id: str,
    cfg_rel: str,
    num_channel: int,
    num_block: int,
    lr_w: int,
    lr_h: int,
    preset: str,
    warmup: int,
    iters: int,
    skip_bench: bool,
    skip_push: bool,
    atol: float,
    do_push_once: list[bool],
) -> dict:
    print(f"\n=== {cand_id} C{num_channel}N{num_block} ===")
    errors: list[str] = []
    torch.manual_seed(INIT_SEED)

    with (PROJECT_ROOT / cfg_rel).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model = build_model_from_config(cfg)
    if not isinstance(model, SepResV2):
        raise TypeError(f"expected SepResV2, got {type(model)}")
    model.eval()
    fused = fuse_sepres_v2(model)

    check = numerical_check(model, fused, lr_h, lr_w, atol)
    print(f"  fuse: pass={check['pass']} max_abs={check['max_abs']:.3e}")
    if not check["pass"]:
        errors.append(f"fuse max_abs={check['max_abs']:.3e}")

    n_conv = count_fused_convs(fused)
    params = fused_param_count(fused)
    macs = conv_macs_at_lr(fused, 180, 180)
    expect = expected_fused_budget(num_channel, num_block)
    if n_conv != expect["fused_convs"]:
        errors.append(f"convs={n_conv}!={expect['fused_convs']}")
    if params != expect["params"]:
        errors.append(f"params={params}!={expect['params']}")
    if macs != expect["conv_macs_lr180"]:
        errors.append(f"macs={macs}!={expect['conv_macs_lr180']}")
    print(f"  budget: convs={n_conv} params={params} macs={macs}")

    stem = f"sepres_v2_{cand_id}_c{num_channel}n{num_block}_fused_{preset}_rand"
    inputshape = f"[1,3,{lr_h},{lr_w}]"
    export_ok = False
    param = binf = None
    in_blob = out_blob = None
    ncnn_mb = None
    ts_path = None
    try:
        print("  TorchScript ...")
        ts_path = export_torchscript(fused, stem, lr_h, lr_w)
        print("  PNNX ...")
        param_src, bin_src = convert_pnnx(ts_path, inputshape)
        NCNN_DIR.mkdir(parents=True, exist_ok=True)
        param = NCNN_DIR / f"{stem}.param"
        binf = NCNN_DIR / f"{stem}.bin"
        shutil.copy2(param_src, param)
        shutil.copy2(bin_src, binf)
        in_blob, out_blob = parse_blobs(param)
        ncnn_mb = (param.stat().st_size + binf.stat().st_size) / 1024**2
        export_ok = True
        print(f"  NCNN: {param.name} {in_blob}->{out_blob} size={ncnn_mb:.3f} MB")
    except Exception as exc:  # noqa: BLE001 — record and continue other candidates
        errors.append(f"export failed: {exc}")
        print(f"  EXPORT FAIL: {exc}")

    phone_smoke = None
    if export_ok and not skip_bench:
        if not adb_ok():
            errors.append("adb unavailable for phone smoke")
            print("  skip bench: adb unavailable")
        else:
            try:
                if do_push_once[0]:
                    ensure_bench_on_device(skip_push)
                    do_push_once[0] = False
                remote_param = f"{DEVICE_DIR}/models/{stem}.param"
                remote_bin = f"{DEVICE_DIR}/models/{stem}.bin"
                print("  phone graph_smoke ...")
                adb("push", str(param), remote_param, capture=False)
                adb("push", str(binf), remote_bin, capture=False)
                phone_smoke = run_bench_remote(
                    remote_param, remote_bin, in_blob, out_blob, lr_w, lr_h, warmup, iters
                )
                med = float(phone_smoke["median_ms"])
                p90 = float(phone_smoke["p90_ms"])
                print(f"  graph_smoke median={med:.2f} ms p90={p90:.2f} ms (NOT official)")
                if med > GROSS_MED_MS:
                    errors.append(f"gross latency median={med:.2f}>{GROSS_MED_MS}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"phone smoke failed: {exc}")
                print(f"  PHONE FAIL: {exc}")

    # Budget band for main candidates (~26K–43K fused params).
    if params < 26000 or params > 43000:
        errors.append(f"params {params} outside [26K,43K] band")

    row = {
        "id": cand_id,
        "config": cfg_rel,
        "init_seed": INIT_SEED,
        "num_channel": num_channel,
        "num_block": num_block,
        "preset": preset,
        "lr_w": lr_w,
        "lr_h": lr_h,
        "numerical": check,
        "fused_convs": n_conv,
        "fused_params": params,
        "conv_macs_lr180": macs,
        "expected": expect,
        "export_ok": export_ok,
        "torchscript": str(ts_path.relative_to(PROJECT_ROOT)) if ts_path else None,
        "ncnn_param": str(param.relative_to(PROJECT_ROOT)) if param else None,
        "ncnn_bin": str(binf.relative_to(PROJECT_ROOT)) if binf else None,
        "ncnn_total_size_mb": round(ncnn_mb, 4) if ncnn_mb is not None else None,
        "in_blob": in_blob,
        "out_blob": out_blob,
        "graph_smoke": phone_smoke,
        "graph_smoke_note": "random-init; not for ranking / 33.3ms / freeze",
        "pass": not errors,
        "errors": errors,
    }
    print(f"  => {'PASS' if row['pass'] else 'FAIL'} {errors or ''}")
    return row


def main() -> None:
    args = parse_args()
    registry = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    presets = {p["name"]: p for p in registry["input_presets"]}
    preset = presets[args.preset]
    lr_w, lr_h = int(preset["lr_w"]), int(preset["lr_h"])

    cands = CANDIDATES
    if args.candidate != "all":
        cands = [c for c in CANDIDATES if c[0] == args.candidate]

    print("=== B4 Gate-0 SepResV2 prescreen ===")
    print(f"  preset={args.preset} LR {lr_w}x{lr_h} smoke={not args.skip_bench}")
    print("  weights=random-init; do not use latency for close ranking")

    do_push_once = [True]
    rows = [
        prescreen_one(
            *c,
            lr_w=lr_w,
            lr_h=lr_h,
            preset=args.preset,
            warmup=args.warmup,
            iters=args.iters,
            skip_bench=args.skip_bench,
            skip_push=args.skip_push,
            atol=args.atol,
            do_push_once=do_push_once,
        )
        for c in cands
    ]

    payload = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "task": "B4_v2_prescreen",
        "gate": "Gate-0-export-phone-smoke",
        "official_numbers": False,
        "init_seed": INIT_SEED,
        "preset": args.preset,
        "smoke_protocol": {
            "warmup": args.warmup,
            "iters": args.iters,
            "fp16": True,
            "vulkan": True,
            "label": "graph_smoke",
        },
        "gross_median_gate_ms": GROSS_MED_MS,
        "candidates": rows,
        "all_pass": all(r["pass"] for r in rows),
        "promote_hint": {
            "v2_b": "first train if pass",
            "v2_a": "gated: only if fewer layers show speed/size potential beyond envelope",
            "v2_c": "gated: only if quality hopeful and wide/shallow graph passes",
        },
    }

    EXP_RESULTS.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_main = EXP_RESULTS / "b4_v2_prescreen.json"
    out_ts = RESULTS_DIR / f"b4_v2_prescreen_{ts}.json"
    out_latest = RESULTS_DIR / "b4_v2_prescreen_latest.json"
    text = json.dumps(payload, indent=2) + "\n"
    out_main.write_text(text, encoding="utf-8")
    out_ts.write_text(text, encoding="utf-8")
    out_latest.write_text(text, encoding="utf-8")

    print(f"\nWrote {out_main.relative_to(PROJECT_ROOT)}")
    print(f"all_pass={payload['all_pass']}")
    if not payload["all_pass"]:
        raise SystemExit("B4 prescreen FAILED")
    print("Prescreen OK. Next: phone envelope, then train v2_b.")


if __name__ == "__main__":
    main()
