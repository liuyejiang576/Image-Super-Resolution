#!/usr/bin/env python3
"""Convert headline models to NCNN via PNNX (TorchScript) or onnx2ncnn fallback."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.model_loader import load_checkpoint_model  # noqa: E402

MODELS_JSON = PROJECT_ROOT / "deploy/models.json"
NCNN_DIR = PROJECT_ROOT / "deploy/artifacts/ncnn"
TS_DIR = PROJECT_ROOT / "deploy/artifacts/torchscript"
ONNX_DIR = PROJECT_ROOT / "deploy/artifacts/onnx"
MANIFEST_OUT = PROJECT_ROOT / "deploy/artifacts/ncnn_manifest.json"
ONNX2NCNN = Path.home() / "ncnn/build-host/tools/onnx/onnx2ncnn"
NCNNOPT = Path.home() / "ncnn/build-host/tools/ncnnoptimize"
PARSE_BLOBS = PROJECT_ROOT / "scripts/parse_ncnn_blobs.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["all", "audit_180", "deploy_720p"], default="all")
    p.add_argument("--model", default="all")
    return p.parse_args()


def export_torchscript(entry: dict, preset: dict) -> Path:
    ckpt = PROJECT_ROOT / entry["checkpoint"]
    model, _ = load_checkpoint_model(ckpt, torch.device("cpu"))
    model.eval()
    if entry.get("deploy_fuse"):
        from models.ecbsr import ECBSR, fuse_ecbsr
        from models.sepres_v2 import SepResV2, fuse_sepres_v2

        if isinstance(model, SepResV2):
            model = fuse_sepres_v2(model)
            print("  deploy_fuse: fuse_sepres_v2")
        elif isinstance(model, ECBSR):
            model = fuse_ecbsr(model)
            print("  deploy_fuse: fuse_ecbsr")
        else:
            raise TypeError(
                f"deploy_fuse set but unsupported type {type(model)} for {entry['id']}"
            )
    h, w = int(preset["lr_h"]), int(preset["lr_w"])
    dummy = torch.randn(1, 3, h, w)
    TS_DIR.mkdir(parents=True, exist_ok=True)
    out = TS_DIR / f"{entry['id']}_{preset['name']}.pt"
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


def convert_onnx2ncnn(onnx_path: Path, base: str) -> tuple[Path, Path]:
    if not ONNX2NCNN.exists():
        raise FileNotFoundError(f"onnx2ncnn not found: {ONNX2NCNN}")
    NCNN_DIR.mkdir(parents=True, exist_ok=True)
    param = NCNN_DIR / f"{base}.param"
    binf = NCNN_DIR / f"{base}.bin"
    subprocess.check_call([str(ONNX2NCNN), str(onnx_path), str(param), str(binf)])
    if NCNNOPT.exists():
        opt_param = NCNN_DIR / f"{base}.opt.param"
        opt_bin = NCNN_DIR / f"{base}.opt.bin"
        try:
            subprocess.check_call(
                [str(NCNNOPT), str(param), str(binf), str(opt_param), str(opt_bin), "0"]
            )
            opt_param.replace(param)
            opt_bin.replace(binf)
        except subprocess.CalledProcessError:
            pass  # unoptimized weights are fine
    return param, binf


def parse_blobs(param_path: Path) -> tuple[str, str]:
    out = subprocess.check_output(
        [sys.executable, str(PARSE_BLOBS), str(param_path)], text=True
    ).strip()
    return out.split("\t")


def main() -> None:
    args = parse_args()
    registry = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    presets = registry["input_presets"]
    if args.preset != "all":
        presets = [p for p in presets if p["name"] == args.preset]

    models = registry["models"]
    if args.model != "all":
        models = [m for m in models if m["id"] == args.model]

    rows = []
    for entry in models:
        for preset in presets:
            base = f"{entry['id']}_{preset['name']}"
            h, w = int(preset["lr_h"]), int(preset["lr_w"])
            inputshape = f"[1,3,{h},{w}]"
            print(f"\n=== {base} ===")

            onnx_path = ONNX_DIR / f"{base}.onnx"
            try:
                if entry["id"] == "fsrcnn":
                    raise RuntimeError("FSRCNN: use PNNX (onnx2ncnn segfaults)")
                if onnx_path.exists():
                    param, binf = convert_onnx2ncnn(onnx_path, base)
                    method = "onnx2ncnn"
                else:
                    raise FileNotFoundError(onnx_path)
            except Exception as exc:
                print(f"  onnx2ncnn skipped ({exc}); using PNNX")
                ts_path = export_torchscript(entry, preset)
                param, binf = convert_pnnx(ts_path, inputshape)
                NCNN_DIR.mkdir(parents=True, exist_ok=True)
                dst_param = NCNN_DIR / f"{base}.param"
                dst_bin = NCNN_DIR / f"{base}.bin"
                shutil.copy2(param, dst_param)
                shutil.copy2(binf, dst_bin)
                param, binf = dst_param, dst_bin
                method = "pnnx"

            in_blob, out_blob = parse_blobs(param)
            rows.append({
                "model_id": entry["id"],
                "label": entry["label"],
                "preset": preset["name"],
                "lr_w": w,
                "lr_h": h,
                "hr_w": preset["hr_w"],
                "hr_h": preset["hr_h"],
                "params": entry["params"],
                "flops_g": entry["flops_g"],
                "ncnn_param": str(param.relative_to(PROJECT_ROOT)),
                "ncnn_bin": str(binf.relative_to(PROJECT_ROOT)),
                "ncnn_param_size_mb": round(param.stat().st_size / 1024**2, 4),
                "ncnn_bin_size_mb": round(binf.stat().st_size / 1024**2, 4),
                "ncnn_total_size_mb": round((param.stat().st_size + binf.stat().st_size) / 1024**2, 4),
                "convert_method": method,
                "in_blob": in_blob,
                "out_blob": out_blob,
            })
            print(f"  -> {param.name}, {binf.name} ({method}) blobs {in_blob}->{out_blob}")

    MANIFEST_OUT.write_text(json.dumps({"models": rows}, indent=2), encoding="utf-8")
    print(f"\nWrote {MANIFEST_OUT.relative_to(PROJECT_ROOT)} ({len(rows)} models)")


if __name__ == "__main__":
    main()
