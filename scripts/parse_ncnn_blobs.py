#!/usr/bin/env python3
"""Parse NCNN .param files for input/output blob names."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_blobs(param_path: Path) -> tuple[str, str]:
    lines = [l.strip() for l in param_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(lines) < 3:
        raise ValueError(f"Invalid param file: {param_path}")

    in_blob = "in0"
    out_blob = "out0"

    for line in lines[2:]:
        parts = line.split()
        if not parts:
            continue
        layer_type = parts[0]
        if layer_type == "Input" and len(parts) > 1:
            in_blob = parts[1]
            continue
        if len(parts) < 4:
            continue
        try:
            n_bottom = int(parts[2])
            n_top = int(parts[3])
        except ValueError:
            continue
        blob_start = 4
        bottoms = parts[blob_start : blob_start + n_bottom]
        tops = parts[blob_start + n_bottom : blob_start + n_bottom + n_top]
        if tops:
            out_blob = tops[-1]

    return in_blob, out_blob


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("param", type=Path)
    args = ap.parse_args()
    in_blob, out_blob = parse_blobs(args.param)
    print(f"{in_blob}\t{out_blob}")


if __name__ == "__main__":
    main()
