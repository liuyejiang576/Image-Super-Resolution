#!/usr/bin/env python3
"""Quick environment check for SR experiments."""

from __future__ import annotations

import platform

import torch


def main() -> None:
    print("Python environment check")
    print(f"Platform: {platform.platform()}")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Device[0]: {torch.cuda.get_device_name(0)}")
        print(f"CUDA runtime: {torch.version.cuda}")
    else:
        print("No GPU detected by torch.")


if __name__ == "__main__":
    main()
