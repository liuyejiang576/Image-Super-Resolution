#!/usr/bin/env python3
"""Train DualStream / Plain C20N5 probes (thin wrapper over train_ecbsr.py).

  python scripts/train_dual_plain.py --config configs/exp/dual_stream_c20n5_2k.yaml
  python scripts/train_dual_plain.py --config configs/exp/plain_c20n5_2k.yaml
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = "configs/exp/dual_stream_c20n5_2k.yaml"


def main() -> None:
    argv = sys.argv[1:]
    has_config = any(a == "--config" or a.startswith("--config=") for a in argv)
    if not has_config:
        sys.argv = [sys.argv[0], "--config", DEFAULT_CONFIG, *argv]
    runpy.run_path(str(SCRIPT_DIR / "train_ecbsr.py"), run_name="__main__")


if __name__ == "__main__":
    main()
