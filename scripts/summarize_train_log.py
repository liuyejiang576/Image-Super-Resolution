#!/usr/bin/env python3
"""Summarize training JSONL log with latest and best metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", default="results/fsrcnn/train_log.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = Path(args.log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Log not found: {log_path}")

    rows = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        print("No rows found in log yet.")
        return

    latest = rows[-1]
    best = max(rows, key=lambda r: r.get("val_psnr", float("-inf")))

    print("Latest:")
    print(
        f"  epoch={latest['epoch']} step={latest['global_step']} "
        f"train_loss={latest['train_loss']:.6f} val_loss={latest['val_loss']:.6f} "
        f"val_psnr={latest['val_psnr']:.4f}"
    )

    print("Best by val_psnr:")
    print(
        f"  epoch={best['epoch']} step={best['global_step']} "
        f"train_loss={best['train_loss']:.6f} val_loss={best['val_loss']:.6f} "
        f"val_psnr={best['val_psnr']:.4f}"
    )

    print(f"Total logged epochs: {len(rows)}")


if __name__ == "__main__":
    main()
