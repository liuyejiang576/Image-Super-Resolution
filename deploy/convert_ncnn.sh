#!/usr/bin/env bash
# Convert verified ONNX models to NCNN param/bin for on-device benchmarking.
#
# Preferred: python scripts/convert_deployment.py
# (FSRCNN via PNNX; MobileSRNet via onnx2ncnn or PNNX fallback)
#
# Legacy wrapper kept for DEPLOY.md compatibility.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec python3 "$ROOT/scripts/convert_deployment.py" "$@"
