#!/usr/bin/env python3
"""Deprecated thin wrapper → ``b5_train_20k.py`` (KD gated after all MSE)."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "scripts" / "b5_train_20k.py"

print(
    "[deprecated] pecsr_kd_20k.py → use: python scripts/b5_train_20k.py "
    + " ".join(sys.argv[1:] or ["watch"])
    + "  (KD auto after MSE; do not force while MSE live)",
    file=sys.stderr,
)
sys.argv = [str(TARGET), *sys.argv[1:]]
runpy.run_path(str(TARGET), run_name="__main__")
