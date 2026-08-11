#!/usr/bin/env python3
"""B1: fuse DW+PW → dense 3×3, numerical check, NCNN export, phone sep vs fused bench."""

from __future__ import annotations

import argparse
import json
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

from models.mobile_srnet import (  # noqa: E402
    DepthwiseSeparableBlock,
    FusedResidualBlock,
    fuse_mobile_srnet,
)
from utils.model_loader import load_checkpoint_model  # noqa: E402

MODELS_JSON = PROJECT_ROOT / "deploy/models.json"
NCNN_DIR = PROJECT_ROOT / "deploy/artifacts/ncnn"
TS_DIR = PROJECT_ROOT / "deploy/artifacts/torchscript"
RESULTS_DIR = PROJECT_ROOT / "deploy/artifacts/results"
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

# ids in models.json → B1 labels
TARGET_IDS = ("mobile_srnet_base", "mobile_srnet_plus")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="B1 fused vs separable deploy compare")
    p.add_argument("--preset", default="deploy_720p")
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--skip-bench", action="store_true")
    p.add_argument("--skip-push", action="store_true")
    return p.parse_args()


def assert_fuse_preconditions(model: nn.Module) -> None:
    for i, block in enumerate(model.body):
        if not isinstance(block, DepthwiseSeparableBlock):
            raise TypeError(f"body[{i}] is {type(block)}, not DepthwiseSeparableBlock")
        dw, pw, act = block.conv[0], block.conv[1], block.conv[2]
        if not isinstance(act, nn.ReLU6):
            raise RuntimeError(f"body[{i}]: activation after PW is {type(act)}, expected ReLU6")
        if dw.groups != dw.in_channels:
            raise RuntimeError(f"body[{i}]: first conv is not depthwise")
        # No mid-activation: Sequential is DW, PW, ReLU6 only.


def numerical_check(
    sep: nn.Module, fused: nn.Module, lr_h: int, lr_w: int, atol: float, rtol: float
) -> dict:
    x = torch.randn(1, 3, lr_h, lr_w)
    with torch.no_grad():
        y_sep = sep(x)
        y_fused = fused(x)
    abs_err = (y_sep - y_fused).abs()
    max_abs = float(abs_err.max())
    denom = y_sep.abs().clamp_min(1e-8)
    max_rel = float((abs_err / denom).max())
    ok = bool(torch.allclose(y_sep, y_fused, atol=atol, rtol=rtol))
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "atol": atol,
        "rtol": rtol,
        "pass": ok,
        "output_shape": list(y_sep.shape),
    }


def conv_macs(module: nn.Module, h: int, w: int) -> int:
    """Approximate multiply-adds for Conv2d at feature map HxW (output spatial)."""
    total = 0
    for m in module.modules():
        if not isinstance(m, nn.Conv2d):
            continue
        kh, kw = m.kernel_size
        # output spatial ≈ input for stride=1 padding=same
        cout, cin_g = m.out_channels, m.in_channels // m.groups
        total += cout * cin_g * kh * kw * h * w
    return total


def model_macs_g(model: nn.Module, lr_h: int, lr_w: int, scale: int = 4) -> float:
    # Head/body at LR; tail Conv2d also at LR then PixelShuffle (no MACs).
    return round(conv_macs(model, lr_h, lr_w) / 1e9, 4)


def export_torchscript(model: nn.Module, stem: str, lr_h: int, lr_w: int) -> Path:
    TS_DIR.mkdir(parents=True, exist_ok=True)
    out = TS_DIR / f"{stem}.pt"
    dummy = torch.randn(1, 3, lr_h, lr_w)
    model.eval()
    with torch.no_grad():
        torch.jit.trace(model, dummy).save(str(out))
    return out


def convert_pnnx(ts_path: Path, inputshape: str) -> tuple[Path, Path]:
    subprocess.check_call(
        ["pnnx", str(ts_path), f"inputshape={inputshape}", "device=cpu", "fp16=0"],
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


def collect_device_info() -> dict:
    script = PROJECT_ROOT / "deploy/collect_device_info.sh"
    return json.loads(subprocess.check_output([str(script)], text=True))


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
    registry = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    presets = {p["name"]: p for p in registry["input_presets"]}
    if args.preset not in presets:
        raise SystemExit(f"Unknown preset {args.preset}")
    preset = presets[args.preset]
    lr_w, lr_h = int(preset["lr_w"]), int(preset["lr_h"])
    inputshape = f"[1,3,{lr_h},{lr_w}]"

    entries = {e["id"]: e for e in registry["models"]}
    rows: list[dict] = []
    num_reports: dict[str, dict] = {}

    NCNN_DIR.mkdir(parents=True, exist_ok=True)

    for mid in TARGET_IDS:
        entry = entries[mid]
        ckpt = PROJECT_ROOT / entry["checkpoint"]
        print(f"\n=== load {mid} from {entry['checkpoint']} ===")
        sep_model, cfg = load_checkpoint_model(ckpt, torch.device("cpu"))
        assert_fuse_preconditions(sep_model)
        fused_model = fuse_mobile_srnet(sep_model)
        for block in fused_model.body:
            if not isinstance(block, FusedResidualBlock):
                raise TypeError("fuse_mobile_srnet did not produce FusedResidualBlock")

        check = numerical_check(sep_model, fused_model, lr_h, lr_w, args.atol, args.rtol)
        num_reports[mid] = check
        print(
            f"  numerical: pass={check['pass']} max_abs={check['max_abs']:.3e} "
            f"max_rel={check['max_rel']:.3e}"
        )
        if not check["pass"]:
            raise SystemExit(f"Numerical fuse check failed for {mid}")

        variants = {"sep": sep_model, "fused": fused_model}
        for vname, model in variants.items():
            stem = f"{mid}_{vname}_{args.preset}"
            print(f"  export {stem} ...")
            ts_path = export_torchscript(model, stem, lr_h, lr_w)
            param_src, bin_src = convert_pnnx(ts_path, inputshape)
            param = NCNN_DIR / f"{stem}.param"
            binf = NCNN_DIR / f"{stem}.bin"
            shutil.copy2(param_src, param)
            shutil.copy2(bin_src, binf)
            in_blob, out_blob = parse_blobs(param)
            macs_g = model_macs_g(model, lr_h, lr_w)
            rows.append({
                "model_id": mid,
                "label": entry["label"],
                "variant": vname,
                "checkpoint": entry["checkpoint"],
                "ckpt_budget": "20k",
                "preset": args.preset,
                "lr_w": lr_w,
                "lr_h": lr_h,
                "params": entry["params"],
                "graph_macs_g": macs_g,
                "ncnn_param": str(param.relative_to(PROJECT_ROOT)),
                "ncnn_bin": str(binf.relative_to(PROJECT_ROOT)),
                "ncnn_param_size_mb": round(param.stat().st_size / 1024**2, 4),
                "ncnn_bin_size_mb": round(binf.stat().st_size / 1024**2, 4),
                "ncnn_total_size_mb": round(
                    (param.stat().st_size + binf.stat().st_size) / 1024**2, 4
                ),
                "convert_method": "pnnx",
                "in_blob": in_blob,
                "out_blob": out_blob,
                "feat": int(cfg["model"]["feat"]),
                "num_blocks": int(cfg["model"]["num_blocks"]),
            })
            print(f"    -> {param.name} blobs {in_blob}->{out_blob} macs_g={macs_g}")

    if not args.skip_bench:
        if not BENCH_BIN.exists():
            raise SystemExit(f"Missing {BENCH_BIN}")
        device_info = collect_device_info()
        print("Device:", json.dumps(device_info, indent=2))
        adb("shell", f"mkdir -p {DEVICE_DIR}/models", capture=False)
        if not args.skip_push:
            adb("push", str(BENCH_BIN), f"{DEVICE_DIR}/sr_bench", capture=False)
            adb("shell", f"chmod +x {DEVICE_DIR}/sr_bench", capture=False)
        if not LIBOMP.exists():
            raise SystemExit(f"Missing {LIBOMP}")
        adb("push", str(LIBOMP), f"{DEVICE_DIR}/libomp.so", capture=False)

        for row in rows:
            param_local = PROJECT_ROOT / row["ncnn_param"]
            bin_local = PROJECT_ROOT / row["ncnn_bin"]
            base = param_local.stem
            remote_param = f"{DEVICE_DIR}/models/{base}.param"
            remote_bin = f"{DEVICE_DIR}/models/{base}.bin"
            print(f"\n=== bench {row['model_id']} {row['variant']} ===")
            adb("push", str(param_local), remote_param, capture=False)
            adb("push", str(bin_local), remote_bin, capture=False)
            bench = run_bench_remote(
                remote_param,
                remote_bin,
                row["in_blob"],
                row["out_blob"],
                lr_w,
                lr_h,
                args.warmup,
                args.iters,
            )
            print(
                f"  median={bench['median_ms']:.2f} ms  p90={bench['p90_ms']:.2f} ms  "
                f"fps={bench['fps']:.1f}"
            )
            row.update(bench)

    # Pairwise summary
    by_key = {(r["model_id"], r["variant"]): r for r in rows}
    summary = []
    for mid in TARGET_IDS:
        sep, fused = by_key[(mid, "sep")], by_key[(mid, "fused")]
        item = {
            "model_id": mid,
            "label": sep["label"],
            "ckpt_budget": "20k",
            "numerical": num_reports[mid],
            "sep_median_ms": sep.get("median_ms"),
            "fused_median_ms": fused.get("median_ms"),
            "sep_p90_ms": sep.get("p90_ms"),
            "fused_p90_ms": fused.get("p90_ms"),
            "sep_ncnn_mb": sep["ncnn_total_size_mb"],
            "fused_ncnn_mb": fused["ncnn_total_size_mb"],
            "sep_macs_g": sep["graph_macs_g"],
            "fused_macs_g": fused["graph_macs_g"],
        }
        if sep.get("median_ms") is not None and fused.get("median_ms") is not None:
            item["median_speedup"] = round(sep["median_ms"] / fused["median_ms"], 3)
            item["p90_speedup"] = round(sep["p90_ms"] / fused["p90_ms"], 3)
            item["fused_faster"] = fused["median_ms"] < sep["median_ms"]
        summary.append(item)

    out = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "task": "B1_fused_vs_sep",
        "ckpt_budget": "20k",
        "preset": args.preset,
        "protocol": {
            "warmup": args.warmup,
            "iters": args.iters,
            "fp16": True,
            "vulkan": True,
            "backend": "ncnn_vulkan",
        },
        "numerical_checks": num_reports,
        "summary": summary,
        "results": rows,
    }
    if not args.skip_bench:
        out["device"] = device_info

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"fused_sep_compare_{ts}.json"
    latest = RESULTS_DIR / "fused_sep_compare_latest.json"
    text = json.dumps(out, indent=2)
    out_path.write_text(text, encoding="utf-8")
    latest.write_text(text, encoding="utf-8")
    print(f"\nWrote {out_path.relative_to(PROJECT_ROOT)}")
    print(f"Also: {latest.relative_to(PROJECT_ROOT)}")
    print("\nSummary:")
    for s in summary:
        print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
