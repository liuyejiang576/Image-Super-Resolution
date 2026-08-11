#!/usr/bin/env python3
"""One-shot deployment pipeline: export ONNX -> verify -> (optional) convert NCNN."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd))
    subprocess.check_call(cmd, cwd=PROJECT_ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description="Deployment smoke pipeline (PC side)")
    ap.add_argument("--model", default="all")
    ap.add_argument("--preset", default="all")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--convert-ncnn", action="store_true", help="Run deploy/convert_ncnn.sh if onnx2ncnn exists")
    args = ap.parse_args()

    py = sys.executable
    run([py, "scripts/export_onnx.py", "--model", args.model, "--preset", args.preset])

    if not args.skip_verify:
        try:
            import onnx  # noqa: F401
            import onnxruntime  # noqa: F401
        except ImportError:
            print("\nInstall verify deps: pip install onnx onnxruntime onnxscript")
            print("Then: python scripts/verify_onnx_export.py")
            return
        run([py, "scripts/verify_onnx_export.py"])

    convert_sh = PROJECT_ROOT / "deploy/convert_ncnn.sh"
    if args.convert_ncnn and convert_sh.exists():
        run(["bash", str(convert_sh)])
    else:
        print("\nNext (if onnx2ncnn installed): ./deploy/convert_ncnn.sh")
        print("Then build + bench: see deploy/DEPLOY.md")


if __name__ == "__main__":
    main()
