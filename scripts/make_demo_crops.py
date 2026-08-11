#!/usr/bin/env python3
"""Make curated LR crops for the class demo (deploy/demo/crops/)."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HR_DIR = PROJECT_ROOT / "data/div2k/DIV2K_valid_HR"
OUT_DIR = PROJECT_ROOT / "deploy/demo/crops"
SIZE = 180  # audit_180 LR

# (stem, hr_name, left, top, note) — coords on HR; we take SIZE*4 then bicubic↓ to LR
CROPS = [
    ("crop01_texture", "0801.png", 400, 300, "brick / fine texture"),
    ("crop02_edges", "0802.png", 200, 400, "structure edges"),
    ("crop03_textish", "0803.png", 600, 200, "high-frequency detail"),
    ("crop04_foliage", "0804.png", 100, 200, "foliage / clutter"),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    catalog = []
    for stem, name, left, top, note in CROPS:
        hr_path = HR_DIR / name
        if not hr_path.exists():
            raise SystemExit(f"Missing {hr_path}")
        hr = Image.open(hr_path).convert("RGB")
        box = (left, top, left + SIZE * 4, top + SIZE * 4)
        if box[2] > hr.width or box[3] > hr.height:
            raise SystemExit(f"Crop out of bounds for {name}: {box} vs {hr.size}")
        patch = hr.crop(box)
        lr = patch.resize((SIZE, SIZE), Image.Resampling.BICUBIC)
        out = OUT_DIR / f"{stem}_lr{SIZE}.png"
        lr.save(out)
        catalog.append(
            {
                "file": out.name,
                "source": f"DIV2K_valid_HR/{name}",
                "hr_box": list(box),
                "lr_size": [SIZE, SIZE],
                "look_for": note,
            }
        )
        print(f"Wrote {out.relative_to(PROJECT_ROOT)}")
    (OUT_DIR / "catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(f"Catalog: {len(catalog)} crops")


if __name__ == "__main__":
    main()
