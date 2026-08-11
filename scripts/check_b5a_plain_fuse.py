#!/usr/bin/env python3
"""B5a P0 Gate-0: plain3x3 body fuse parity + iso-MAC vs PECSR fused budget.

  /home/hyb/miniforge3/envs/cv_env/bin/python scripts/check_b5a_plain_fuse.py
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
    PlainBodyBlock,
    SepResV2,
    conv_macs_at_lr,
    count_fused_convs,
    expected_fused_budget,
    fuse_sepres_v2,
    fused_param_count,
)
from utils.model_loader import build_model_from_config  # noqa: E402

CFG = "configs/exp/abl_plain3x3_c16n10_20k.yaml"
C, N = 16, 10


def main() -> None:
    torch.manual_seed(0)
    with (PROJECT_ROOT / CFG).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = build_model_from_config(cfg)
    assert isinstance(model, SepResV2)
    assert model.body_kind == "plain3x3"
    assert model.num_channel == C and model.num_block == N
    assert all(isinstance(b, PlainBodyBlock) for b in model.body)

    x = torch.randn(2, 3, 48, 48)
    model.train()
    y_train = model(x)
    assert y_train.shape == (2, 3, 192, 192), y_train.shape

    model.eval()
    with torch.no_grad():
        y_eval = model(x)
        fused = fuse_sepres_v2(model)
        y_fused = fused(x)

    abs_fuse = (y_eval - y_fused).abs().max().item()
    n_conv = count_fused_convs(fused)
    params = fused_param_count(fused)
    macs = conv_macs_at_lr(fused, 180, 180)
    expect = expected_fused_budget(C, N)

    errors: list[str] = []
    if abs_fuse > 1e-5:
        errors.append(f"fuse max_abs={abs_fuse:.3e} > 1e-5")
    if n_conv != expect["fused_convs"]:
        errors.append(f"fused_convs={n_conv} != {expect['fused_convs']}")
    if params != expect["params"]:
        errors.append(f"params={params} != {expect['params']}")
    if macs != expect["conv_macs_lr180"]:
        errors.append(f"macs={macs} != {expect['conv_macs_lr180']}")

    # Train-time param count must be << ECB multi-branch (sanity, not gate fail).
    train_params = sum(p.numel() for p in model.parameters())
    pecsr_cfg = yaml.safe_load(
        (PROJECT_ROOT / "configs/exp/sepres_v2_c16n10_20k.yaml").read_text(encoding="utf-8")
    )
    pecsr = build_model_from_config(pecsr_cfg)
    pecsr_train_params = sum(p.numel() for p in pecsr.parameters())

    row = {
        "gate": "B5a-P0-plain3x3-fuse-budget",
        "config": CFG,
        "body_kind": model.body_kind,
        "eval_vs_fused_max_abs": abs_fuse,
        "fused_convs": n_conv,
        "fused_params": params,
        "conv_macs_lr180": macs,
        "expected": expect,
        "train_params_plain": train_params,
        "train_params_pecsr_ecb": pecsr_train_params,
        "pass": not errors,
        "errors": errors,
    }
    out = PROJECT_ROOT / "results/exp_runs/b5a_plain_fuse_smoke.json"
    out.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    status = "PASS" if row["pass"] else "FAIL"
    print(
        f"[{status}] P0 plain3x3: fuse_abs={abs_fuse:.3e} convs={n_conv} "
        f"params={params} macs={macs} "
        f"train_params plain={train_params} ecb={pecsr_train_params}"
    )
    for err in errors:
        print(f"  ! {err}")
    print(f"wrote {out.relative_to(PROJECT_ROOT)}")
    if not row["pass"]:
        raise SystemExit("B5a P0 fuse/budget smoke FAILED")
    print("B5a P0 fuse smoke OK")


if __name__ == "__main__":
    main()
