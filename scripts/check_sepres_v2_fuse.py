#!/usr/bin/env python3
"""Gate-0 smoke for SepResSR-v2: shape, fuse parity, fused budget audit.

Checks all three B4 candidates against IMPLEMENTATION §11 analytic budgets.
Does not export NCNN or run phone smoke (those are separate Gate-0 steps).

  python scripts/check_sepres_v2_fuse.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.sepres_v2 import (  # noqa: E402
    SepResV2,
    conv_macs_at_lr,
    count_fused_convs,
    expected_fused_budget,
    fuse_sepres_v2,
    fused_param_count,
)
from utils.model_loader import build_model_from_config  # noqa: E402

CANDIDATES = [
    ("v2_a", "configs/exp/sepres_v2_c16n8_20k.yaml", 16, 8),
    ("v2_b", "configs/exp/sepres_v2_c16n10_20k.yaml", 16, 10),
    ("v2_c", "configs/exp/sepres_v2_c20n6_20k.yaml", 20, 6),
]


def audit_one(
    cand_id: str,
    cfg_rel: str,
    num_channel: int,
    num_block: int,
) -> dict:
    torch.manual_seed(0)
    cfg_path = PROJECT_ROOT / cfg_rel
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = build_model_from_config(cfg)
    if not isinstance(model, SepResV2):
        raise TypeError(f"{cand_id}: expected SepResV2, got {type(model)}")
    if model.num_channel != num_channel or model.num_block != num_block:
        raise AssertionError(
            f"{cand_id}: geometry mismatch "
            f"got C={model.num_channel} N={model.num_block}, "
            f"want C={num_channel} N={num_block}"
        )

    x = torch.randn(2, 3, 48, 48)
    model.train()
    y_train = model(x)
    assert y_train.shape == (2, 3, 192, 192), y_train.shape

    model.eval()
    with torch.no_grad():
        y_eval = model(x)
        fused = fuse_sepres_v2(model)
        y_fused = fused(x)

    # Architecture invariants: MobileSRNet shell, not ECBSR backbone/shortcut.
    assert not hasattr(model, "backbone")
    assert hasattr(model, "head") and hasattr(model, "body") and hasattr(model, "tail")
    assert isinstance(model.head, torch.nn.Conv2d)
    assert isinstance(model.tail, torch.nn.Conv2d)
    assert len(list(model.body)) == num_block

    abs_fuse = (y_eval - y_fused).abs().max().item()
    n_conv = count_fused_convs(fused)
    params = fused_param_count(fused)
    macs = conv_macs_at_lr(fused, 180, 180)
    expect = expected_fused_budget(num_channel, num_block)

    errors: list[str] = []
    if abs_fuse > 1e-5:
        errors.append(f"fuse max_abs={abs_fuse:.3e} > 1e-5")
    if n_conv != expect["fused_convs"]:
        errors.append(f"fused_convs={n_conv} != {expect['fused_convs']}")
    if params != expect["params"]:
        errors.append(f"params={params} != {expect['params']}")
    if macs != expect["conv_macs_lr180"]:
        errors.append(f"macs={macs} != {expect['conv_macs_lr180']}")

    row = {
        "id": cand_id,
        "config": cfg_rel,
        "num_channel": num_channel,
        "num_block": num_block,
        "output_shape": list(y_eval.shape),
        "eval_vs_fused_max_abs": abs_fuse,
        "fused_convs": n_conv,
        "fused_params": params,
        "conv_macs_lr180": macs,
        "expected": expect,
        "pass": not errors,
        "errors": errors,
    }
    status = "PASS" if row["pass"] else "FAIL"
    print(
        f"[{status}] {cand_id} C{num_channel}N{num_block}: "
        f"shape={tuple(y_eval.shape)} fuse_abs={abs_fuse:.3e} "
        f"convs={n_conv} params={params} macs={macs}"
    )
    for err in errors:
        print(f"  ! {err}")
    return row


def main() -> None:
    rows = [audit_one(*c) for c in CANDIDATES]
    summary = {
        "gate": "B4-Gate0-model-fuse-budget",
        "candidates": rows,
        "all_pass": all(r["pass"] for r in rows),
    }
    out = PROJECT_ROOT / "results/exp_runs/b4_v2_fuse_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(PROJECT_ROOT)}")
    if not summary["all_pass"]:
        raise SystemExit("SepResV2 fuse/budget smoke FAILED")
    print("SepResV2 fuse smoke OK (all candidates)")


if __name__ == "__main__":
    main()
