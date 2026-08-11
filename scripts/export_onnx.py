#!/usr/bin/env python3
"""Export headline SR checkpoints to ONNX for mobile deployment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.model_loader import load_checkpoint_model  # noqa: E402

DEPLOY_ROOT = PROJECT_ROOT / "deploy"
MODELS_JSON = DEPLOY_ROOT / "models.json"
ONNX_DIR = DEPLOY_ROOT / "artifacts" / "onnx"
MANIFEST_PATH = DEPLOY_ROOT / "artifacts" / "export_manifest.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export PyTorch SR models to ONNX")
    p.add_argument(
        "--model",
        choices=["all", "fsrcnn", "mobile_srnet_base", "mobile_srnet_plus"],
        default="all",
    )
    p.add_argument(
        "--preset",
        choices=["all", "audit_180", "deploy_720p"],
        default="all",
        help="Input resolution preset from deploy/models.json",
    )
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default=str(ONNX_DIR))
    return p.parse_args()


def load_registry() -> dict:
    return json.loads(MODELS_JSON.read_text(encoding="utf-8"))


def select_models(registry: dict, model_arg: str) -> list[dict]:
    models = registry["models"]
    if model_arg == "all":
        return models
    return [m for m in models if m["id"] == model_arg]


def select_presets(registry: dict, preset_arg: str) -> list[dict]:
    presets = registry["input_presets"]
    if preset_arg == "all":
        return presets
    return [p for p in presets if p["name"] == preset_arg]


def export_one(
    model_entry: dict,
    preset: dict,
    out_dir: Path,
    opset: int,
    device: torch.device,
) -> dict:
    ckpt = PROJECT_ROOT / model_entry["checkpoint"]
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint missing: {ckpt}")

    model, cfg = load_checkpoint_model(ckpt, device)
    model.eval()

    w, h = int(preset["lr_w"]), int(preset["lr_h"])
    dummy = torch.randn(1, 3, h, w, device=device)
    onnx_name = f"{model_entry['id']}_{preset['name']}.onnx"
    onnx_path = out_dir / onnx_name

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["lr"],
            output_names=["sr"],
            dynamic_axes=None,
        )

    size_mb = onnx_path.stat().st_size / (1024 ** 2)
    return {
        "model_id": model_entry["id"],
        "label": model_entry["label"],
        "checkpoint": str(model_entry["checkpoint"]),
        "preset": preset["name"],
        "lr_w": w,
        "lr_h": h,
        "hr_w": int(preset["hr_w"]),
        "hr_h": int(preset["hr_h"]),
        "onnx_path": str(onnx_path.relative_to(PROJECT_ROOT)),
        "onnx_size_mb": round(size_mb, 4),
        "opset": opset,
        "params": model_entry["params"],
        "flops_g": model_entry["flops_g"],
    }


def main() -> None:
    args = parse_args()
    registry = load_registry()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    exports: list[dict] = []

    for model_entry in select_models(registry, args.model):
        for preset in select_presets(registry, args.preset):
            print(f"Exporting {model_entry['id']} @ {preset['name']} ({preset['lr_w']}x{preset['lr_h']})...")
            info = export_one(model_entry, preset, out_dir, args.opset, device)
            exports.append(info)
            print(f"  -> {info['onnx_path']} ({info['onnx_size_mb']:.3f} MB)")

    manifest = {
        "exports": exports,
        "models_json": str(MODELS_JSON.relative_to(PROJECT_ROOT)),
        "benchmark": registry["benchmark"],
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote manifest: {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    print("Next: python scripts/verify_onnx_export.py")


if __name__ == "__main__":
    main()
