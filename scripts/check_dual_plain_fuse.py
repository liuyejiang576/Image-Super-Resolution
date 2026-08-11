#!/usr/bin/env python3
"""Gate-0: DualStream→Plain fuse parity + C20N5 budget audit.

  python scripts/check_dual_plain_fuse.py
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

from models.dual_plain_sr import (  # noqa: E402
    DualStreamSR,
    PlainSR,
    conv_macs_at_lr,
    count_plain_convs,
    expected_plain_budget,
    fuse_dual_stream_sr,
    plain_param_count,
)
from utils.model_loader import build_model_from_config  # noqa: E402

ATOL = 1e-5
OUT = PROJECT_ROOT / "results/exp_runs/b4_dual_plain_fuse_smoke.json"


def main() -> int:
    torch.manual_seed(0)
    dual_cfg_path = PROJECT_ROOT / "configs/exp/dual_stream_c20n5_2k.yaml"
    plain_cfg_path = PROJECT_ROOT / "configs/exp/plain_c20n5_2k.yaml"
    with dual_cfg_path.open("r", encoding="utf-8") as f:
        dual_cfg = yaml.safe_load(f)
    with plain_cfg_path.open("r", encoding="utf-8") as f:
        plain_cfg = yaml.safe_load(f)

    dual = build_model_from_config(dual_cfg)
    plain = build_model_from_config(plain_cfg)
    assert isinstance(dual, DualStreamSR)
    assert isinstance(plain, PlainSR)

    x = torch.randn(2, 3, 48, 48)
    dual.eval()
    plain.eval()
    with torch.no_grad():
        y_dual = dual(x)
        y_aux = dual(x, return_aux=True)[1]
        fused = fuse_dual_stream_sr(dual)
        y_fused = fused(x)
        y_plain = plain(x)

    assert y_dual.shape == (2, 3, 192, 192), y_dual.shape
    assert y_aux.shape == y_dual.shape
    abs_fuse = (y_dual - y_fused).abs().max().item()

    budget = expected_plain_budget(20, 5)
    fused_params = plain_param_count(fused)
    fused_convs = count_plain_convs(fused)
    fused_macs = conv_macs_at_lr(fused)
    train_params = plain_param_count(dual)
    plain_params = plain_param_count(plain)

    ok = (
        abs_fuse <= ATOL
        and fused_params == budget["params"]
        and fused_convs == budget["fused_convs"]
        and fused_macs == budget["macs_180"]
        and plain_params == budget["params"]
        and train_params == 25186
    )

    report = {
        "task": "b4_dual_plain_fuse_smoke",
        "pass": ok,
        "atol": ATOL,
        "abs_fuse_max": abs_fuse,
        "train_params_dual": train_params,
        "deploy_params_fused": fused_params,
        "deploy_params_plain": plain_params,
        "fused_convs": fused_convs,
        "macs_180": fused_macs,
        "expected": budget,
        "expected_train_params": 25186,
        "plain_random_out_finite": bool(torch.isfinite(y_plain).all()),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not ok:
        print("FAIL", file=sys.stderr)
        return 1
    print(f"PASS → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
