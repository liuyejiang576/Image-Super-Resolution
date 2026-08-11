#!/usr/bin/env python3
"""B2: module/op timing for FSRCNN, MobileSRNet Base/Plus (sep vs fused).

Default device=cpu so overnight A1a keeps the GPU. Answers:
  - Is depthwise the slow part on desktop PyTorch?
  - Is Plus slower from width, depth, or op count?
  - Does fuse reduce wall time / op count here (vs phone B1)?
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.mobile_srnet import (  # noqa: E402
    DepthwiseSeparableBlock,
    FusedResidualBlock,
    fuse_mobile_srnet,
)
from utils.model_loader import load_checkpoint_model  # noqa: E402

MODELS_JSON = PROJECT_ROOT / "deploy/models.json"
OUT_DIR = PROJECT_ROOT / "results/exp_runs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="B2 desktop profile")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--lr-w", type=int, default=320)
    p.add_argument("--lr-h", type=int, default=180)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--runs", type=int, default=50)
    return p.parse_args()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_fn(fn, device: torch.device, warmup: int, runs: int) -> dict:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        sync(device)
        samples = []
        for _ in range(runs):
            sync(device)
            t0 = time.perf_counter()
            fn()
            sync(device)
            samples.append((time.perf_counter() - t0) * 1000.0)
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "p90_ms": sorted(samples)[max(0, int(0.9 * (len(samples) - 1)))],
        "runs": runs,
    }


def count_convs(model: nn.Module) -> dict:
    n_dw = n_pw = n_dense = 0
    for m in model.modules():
        if not isinstance(m, nn.Conv2d):
            continue
        if m.groups == m.in_channels and m.in_channels > 1:
            n_dw += 1
        elif m.kernel_size == (1, 1):
            n_pw += 1
        else:
            n_dense += 1
    return {
        "conv_total": n_dw + n_pw + n_dense,
        "depthwise_3x3": n_dw,
        "pointwise_1x1": n_pw,
        "dense_conv": n_dense,
    }


def time_named_modules(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    warmup: int,
    runs: int,
) -> dict:
    """Time head / body / tail if present; else whole forward only."""
    out = {"full": time_fn(lambda: model(x), device, warmup, runs)}
    if hasattr(model, "head") and hasattr(model, "body") and hasattr(model, "tail"):
        def run_head():
            return model.head(x)

        h = model.head(x)
        def run_body():
            return model.body(h)

        b = model.body(h)
        def run_tail():
            return model.tail(b)

        out["head"] = time_fn(run_head, device, warmup, runs)
        out["body"] = time_fn(run_body, device, warmup, runs)
        out["tail"] = time_fn(run_tail, device, warmup, runs)

        # First body block breakdown if separable
        block0 = model.body[0]
        if isinstance(block0, DepthwiseSeparableBlock):
            feat = h
            dw, pw, act = block0.conv[0], block0.conv[1], block0.conv[2]
            out["block0_dw"] = time_fn(lambda: dw(feat), device, warmup, runs)
            y = dw(feat)
            out["block0_pw"] = time_fn(lambda: pw(y), device, warmup, runs)
            z = pw(y)
            out["block0_act"] = time_fn(lambda: act(z), device, warmup, runs)
            out["block0_full"] = time_fn(lambda: block0(feat), device, warmup, runs)
        elif isinstance(block0, FusedResidualBlock):
            feat = h
            out["block0_dense"] = time_fn(lambda: block0.conv[0](feat), device, warmup, runs)
            out["block0_full"] = time_fn(lambda: block0(feat), device, warmup, runs)
        out["num_blocks"] = len(model.body)
    return out


def load_phone_context() -> dict:
    ctx = {}
    for name, path in [
        ("a0_bench", PROJECT_ROOT / "deploy/artifacts/results/mobile_benchmark_latest.json"),
        ("b1_fused", PROJECT_ROOT / "deploy/artifacts/results/fused_sep_compare_latest.json"),
    ]:
        if path.exists():
            ctx[name] = json.loads(path.read_text(encoding="utf-8"))
    return ctx


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    registry = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    by_id = {e["id"]: e for e in registry["models"]}
    x = torch.randn(1, 3, args.lr_h, args.lr_w, device=device)

    rows = []
    for mid in ("fsrcnn", "mobile_srnet_base", "mobile_srnet_plus"):
        entry = by_id[mid]
        model, cfg = load_checkpoint_model(PROJECT_ROOT / entry["checkpoint"], device)
        model.eval()
        row = {
            "model_id": mid,
            "label": entry["label"],
            "variant": "native",
            "checkpoint": entry["checkpoint"],
            "params": sum(p.numel() for p in model.parameters()),
            "conv_counts": count_convs(model),
            "timing": time_named_modules(model, x, device, args.warmup, args.runs),
            "cfg_model": cfg.get("model", {}),
        }
        rows.append(row)
        print(f"\n=== {mid} native ===")
        print(json.dumps({"params": row["params"], "convs": row["conv_counts"], "full_ms": row["timing"]["full"]}, indent=2))

        if mid.startswith("mobile_srnet"):
            fused = fuse_mobile_srnet(model).to(device).eval()
            frow = {
                "model_id": mid,
                "label": entry["label"] + "-fused",
                "variant": "fused",
                "checkpoint": entry["checkpoint"],
                "params": sum(p.numel() for p in fused.parameters()),
                "conv_counts": count_convs(fused),
                "timing": time_named_modules(fused, x, device, args.warmup, args.runs),
                "cfg_model": cfg.get("model", {}),
            }
            rows.append(frow)
            print(f"=== {mid} fused ===")
            print(json.dumps({"params": frow["params"], "convs": frow["conv_counts"], "full_ms": frow["timing"]["full"]}, indent=2))

    phone = load_phone_context()
    # Build conclusions from desktop + known phone
    by_key = {(r["model_id"], r["variant"]): r for r in rows}
    base_sep = by_key[("mobile_srnet_base", "native")]
    base_fused = by_key[("mobile_srnet_base", "fused")]
    plus_sep = by_key[("mobile_srnet_plus", "native")]
    plus_fused = by_key[("mobile_srnet_plus", "fused")]
    fsrcnn = by_key[("fsrcnn", "native")]

    def ms(row, part="full"):
        return row["timing"][part]["median_ms"]

    conclusions = []
    # Q1: depthwise slow?
    if "block0_dw" in base_sep["timing"] and "block0_pw" in base_sep["timing"]:
        dw = base_sep["timing"]["block0_dw"]["median_ms"]
        pw = base_sep["timing"]["block0_pw"]["median_ms"]
        conclusions.append(
            f"Desktop PyTorch (Base block0): DW={dw:.3f}ms vs PW={pw:.3f}ms "
            f"({'DW heavier' if dw > pw else 'PW heavier'} on this backend)."
        )
    # Q2: Plus slower why?
    base_blocks = base_sep["timing"].get("num_blocks", 6)
    plus_blocks = plus_sep["timing"].get("num_blocks", 8)
    base_feat = int(base_sep["cfg_model"].get("feat", 40))
    plus_feat = int(plus_sep["cfg_model"].get("feat", 64))
    conclusions.append(
        f"Plus vs Base desktop full: {ms(plus_sep):.2f} vs {ms(base_sep):.2f} ms "
        f"(feat {plus_feat}/{base_feat}, blocks {plus_blocks}/{base_blocks}, "
        f"convs {plus_sep['conv_counts']['conv_total']}/{base_sep['conv_counts']['conv_total']})."
    )
    if "body" in base_sep["timing"] and "body" in plus_sep["timing"]:
        conclusions.append(
            f"Body dominates gap: Base body {ms(base_sep,'body'):.2f}ms, "
            f"Plus body {ms(plus_sep,'body'):.2f}ms "
            f"(head {ms(base_sep,'head'):.2f}/{ms(plus_sep,'head'):.2f}, "
            f"tail {ms(base_sep,'tail'):.2f}/{ms(plus_sep,'tail'):.2f})."
        )
    # Q3: fuse
    conclusions.append(
        f"Desktop fuse Base: {ms(base_sep):.2f} → {ms(base_fused):.2f} ms "
        f"(convs {base_sep['conv_counts']['conv_total']} → {base_fused['conv_counts']['conv_total']})."
    )
    conclusions.append(
        f"Desktop fuse Plus: {ms(plus_sep):.2f} → {ms(plus_fused):.2f} ms "
        f"(convs {plus_sep['conv_counts']['conv_total']} → {plus_fused['conv_counts']['conv_total']})."
    )
    conclusions.append(
        "Phone B1 already showed fused slower on NCNN Vulkan; desktop result is supporting "
        "context only (PyTorch ≠ NCNN)."
    )
    conclusions.append(
        "B4 guidance: do not chase DW→dense fuse for speed; prefer capacity (width/depth) "
        "or backend-friendly blocks (ECBSR-style) over folding sep."
    )

    note_lines = [
        "# B2 profile note — desktop module timing",
        "",
        f"- Device: `{args.device}` @ LR {args.lr_w}×{args.lr_h}",
        f"- Warmup/runs: {args.warmup}/{args.runs}",
        f"- Timestamp: {datetime.now().astimezone().isoformat()}",
        "- NCNN layer timing: **skipped** (`sr_bench` has no per-layer mode).",
        "",
        "## Conclusions",
        "",
    ]
    for c in conclusions:
        note_lines.append(f"- {c}")
    note_lines.extend(
        [
            "",
            "## Phone context (already measured)",
            "",
        ]
    )
    if "b1_fused" in phone:
        for s in phone["b1_fused"].get("summary", []):
            note_lines.append(
                f"- {s['model_id']}: sep median {s.get('sep_median_ms')} → "
                f"fused {s.get('fused_median_ms')} ms (phone)"
            )
    if "a0_bench" in phone:
        note_lines.append("- A0 20k phone medians:")
        for r in phone["a0_bench"].get("results", []):
            if r.get("preset") == "deploy_720p":
                note_lines.append(
                    f"  - {r['model_id']}: {r['median_ms']:.2f} ms (p90 {r['p90_ms']:.2f})"
                )

    note_lines.extend(
        [
            "",
            "## Desktop timing table (median ms)",
            "",
            "| model | variant | full | head | body | tail | convs |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for r in rows:
        t = r["timing"]
        note_lines.append(
            "| {mid} | {var} | {full:.2f} | {head} | {body} | {tail} | {n} |".format(
                mid=r["model_id"],
                var=r["variant"],
                full=t["full"]["median_ms"],
                head=f"{t['head']['median_ms']:.2f}" if "head" in t else "—",
                body=f"{t['body']['median_ms']:.2f}" if "body" in t else "—",
                tail=f"{t['tail']['median_ms']:.2f}" if "tail" in t else "—",
                n=r["conv_counts"]["conv_total"],
            )
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUT_DIR / f"b2_profile_{ts}.json"
    latest_json = OUT_DIR / "b2_profile_latest.json"
    note_path = OUT_DIR / "b2_profile_note.md"
    payload = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "task": "B2_profile",
        "device": args.device,
        "lr_w": args.lr_w,
        "lr_h": args.lr_h,
        "warmup": args.warmup,
        "runs": args.runs,
        "conclusions": conclusions,
        "results": rows,
        "phone_context_files": {
            "a0": "deploy/artifacts/results/mobile_benchmark_latest.json",
            "b1": "deploy/artifacts/results/fused_sep_compare_latest.json",
        },
        "ncnn_layer_timing": None,
    }
    text = json.dumps(payload, indent=2)
    json_path.write_text(text, encoding="utf-8")
    latest_json.write_text(text, encoding="utf-8")
    note_path.write_text("\n".join(note_lines) + "\n", encoding="utf-8")
    print("\n".join(note_lines))
    print(f"\nWrote {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {note_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
