#!/usr/bin/env python3
"""Train SepResSR-v2 on DIV2K with periodic validation (B4).

Thin wrapper over ``train_ecbsr.py``: same fair-budget loop, different default
config. Model construction goes through ``build_model_from_config`` (type=sepres_v2).

Prefer control plane:

  python scripts/sepres_v2_20k.py resume
  python scripts/sepres_v2_20k.py watch --interval 60

Or direct:

  python scripts/train_sepres_v2.py --config configs/exp/sepres_v2_c16n10_20k.yaml
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = "configs/exp/sepres_v2_c16n10_20k.yaml"


def main() -> None:
    argv = sys.argv[1:]
    has_config = any(a == "--config" or a.startswith("--config=") for a in argv)
    if not has_config:
        sys.argv = [sys.argv[0], "--config", DEFAULT_CONFIG, *argv]
    runpy.run_path(str(SCRIPT_DIR / "train_ecbsr.py"), run_name="__main__")


if __name__ == "__main__":
    main()
