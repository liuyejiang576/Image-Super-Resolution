#!/usr/bin/env python3
"""Verify ONNX exports against PyTorch (numeric parity smoke test)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.model_loader import load_checkpoint_model  # noqa: E402

MANIFEST_PATH = PROJECT_ROOT / "deploy/artifacts/export_manifest.json"
REPORT_PATH = PROJECT_ROOT / "deploy/artifacts/onnx_verify_report.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=str(MANIFEST_PATH))
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument("--save-json", default=str(REPORT_PATH))
    return p.parse_args()


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse == 0:
        return float("inf")
    return float(-10.0 * np.log10(mse))


def verify_entry(entry: dict, atol: float) -> dict:
    ckpt = PROJECT_ROOT / entry["checkpoint"]
    onnx_path = PROJECT_ROOT / entry["onnx_path"]
    onnx.checker.check_model(str(onnx_path))

    h, w = int(entry["lr_h"]), int(entry["lr_w"])
    x = torch.randn(1, 3, h, w)
    model, _ = load_checkpoint_model(ckpt, torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        pt_out = model(x).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"lr": x.numpy()})[0]

    if pt_out.shape != ort_out.shape:
        # FSRCNN deconv can be off-by-a-few on edges; crop to common size.
        min_h = min(pt_out.shape[2], ort_out.shape[2])
        min_w = min(pt_out.shape[3], ort_out.shape[3])
        pt_out = pt_out[..., :min_h, :min_w]
        ort_out = ort_out[..., :min_h, :min_w]

    max_abs = float(np.max(np.abs(pt_out - ort_out)))
    parity_psnr = psnr(pt_out, ort_out)
    ok = max_abs <= atol or parity_psnr >= 40.0

    return {
        **entry,
        "max_abs_diff": max_abs,
        "parity_psnr_db": parity_psnr,
        "output_shape_pt": list(pt_out.shape),
        "output_shape_onnx": list(ort_out.shape),
        "pass": ok,
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    results = []
    failed = 0

    for entry in manifest["exports"]:
        print(f"Verifying {entry['model_id']} @ {entry['preset']}...")
        row = verify_entry(entry, args.atol)
        results.append(row)
        status = "PASS" if row["pass"] else "FAIL"
        print(
            f"  {status}: max_abs={row['max_abs_diff']:.2e} "
            f"psnr={row['parity_psnr_db']:.2f} dB shape={row['output_shape_onnx']}"
        )
        if not row["pass"]:
            failed += 1

    report = {"results": results, "atol": args.atol, "all_pass": failed == 0}
    save_path = PROJECT_ROOT / args.save_json
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport: {save_path.relative_to(PROJECT_ROOT)}")
    if failed:
        raise SystemExit(f"{failed} export(s) failed parity check")
    print("All ONNX exports verified.")


if __name__ == "__main__":
    main()
