#!/usr/bin/env python3
"""Compare benchmark metric JSON files in a readable table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="results/bicubic_metrics.json")
    parser.add_argument("--target", default="results/fsrcnn_fix_clean/benchmark_metrics.json")
    return parser.parse_args()


def load_json(path: str) -> dict:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    base = load_json(args.base)
    target = load_json(args.target)

    datasets = sorted(set(base.keys()) & set(target.keys()))
    print("Dataset   Base_PSNR   Target_PSNR   Delta   Base_SSIM   Target_SSIM   Delta")
    print("-------------------------------------------------------------------------")
    for d in datasets:
        b_psnr = float(base[d]["psnr"])
        t_psnr = float(target[d]["psnr"])
        b_ssim = float(base[d]["ssim"])
        t_ssim = float(target[d]["ssim"])
        print(
            f"{d:<8} {b_psnr:>9.4f}   {t_psnr:>11.4f}   {t_psnr - b_psnr:>+6.4f}   "
            f"{b_ssim:>9.4f}   {t_ssim:>11.4f}   {t_ssim - b_ssim:>+6.4f}"
        )


if __name__ == "__main__":
    main()
