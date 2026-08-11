#!/usr/bin/env python3
"""Export headline checkpoints to TorchScript for PNNX -> NCNN conversion."""

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

MODELS_JSON = PROJECT_ROOT / "deploy/models.json"
OUT_DIR = PROJECT_ROOT / "deploy/artifacts/torchscript"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="all")
    p.add_argument("--preset", default="audit_180")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    registry = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    presets = {p["name"]: p for p in registry["input_presets"]}
    preset = presets[args.preset]

    models = registry["models"]
    if args.model != "all":
        models = [m for m in models if m["id"] == args.model]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    w, h = int(preset["lr_w"]), int(preset["lr_h"])
    exports = []

    for entry in models:
        ckpt = PROJECT_ROOT / entry["checkpoint"]
        model, _ = load_checkpoint_model(ckpt, torch.device("cpu"))
        model.eval()
        dummy = torch.randn(1, 3, h, w)
        with torch.no_grad():
            traced = torch.jit.trace(model, dummy)
        out = OUT_DIR / f"{entry['id']}_{preset['name']}.pt"
        traced.save(str(out))
        exports.append({
            "model_id": entry["id"],
            "preset": preset["name"],
            "torchscript": str(out.relative_to(PROJECT_ROOT)),
            "inputshape": f"[1,3,{h},{w}]",
        })
        print(f"Saved {out.relative_to(PROJECT_ROOT)}")

    manifest = OUT_DIR / f"torchscript_{preset['name']}.json"
    manifest.write_text(json.dumps(exports, indent=2), encoding="utf-8")
    print(f"Wrote {manifest.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
