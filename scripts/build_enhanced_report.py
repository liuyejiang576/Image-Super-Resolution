#!/usr/bin/env python3
"""Build enhanced final report with latency audit and KD analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def avg_metric(metrics: dict, key: str) -> float | None:
    vals = [v[key] for v in metrics.values() if isinstance(v, dict) and key in v]
    return sum(vals) / len(vals) if vals else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--report-path", default="results/final_report.md")
    args = parser.parse_args()
    root = Path(args.results_dir)

    entries = [
        ("Bicubic", root / "bicubic_metrics.json", None),
        ("FSRCNN", root / "fsrcnn_fix_clean/benchmark_metrics.json", root / "fsrcnn_fix_clean/profile.json"),
        ("FSRCNN-Small", root / "fsrcnn_small/benchmark_metrics.json", root / "fsrcnn_small/profile.json"),
        ("MobileSRNet", root / "mobile_srnet/benchmark_metrics.json", root / "mobile_srnet/profile.json"),
        ("MobileSRNet+KD", root / "mobile_srnet_kd/benchmark_metrics.json", root / "mobile_srnet_kd/profile.json"),
        ("SwinIR", root / "swinir/benchmark_metrics.json", root / "swinir/profile.json"),
    ]

    rows = []
    for name, metrics_path, profile_path in entries:
        metrics = load_json(metrics_path)
        if not metrics:
            continue
        prof = load_json(profile_path) if profile_path else {}
        rows.append({
            "name": name,
            "avg_psnr": avg_metric(metrics, "psnr"),
            "avg_ssim": avg_metric(metrics, "ssim"),
            "avg_lpips": avg_metric(metrics, "lpips"),
            "params": prof.get("params"),
            "flops_g": prof.get("flops_g"),
            "latency_fp32": prof.get("latency_fp32_ms"),
            "latency_fp16": prof.get("latency_fp16_ms"),
            "metrics": metrics,
        })

    latency_audit = load_json(root / "latency_audit/latency_audit.json")
    kd_summary = load_json(root / "kd_analysis/summary.json")
    baseline = load_json(root / "exp_runs/baseline_snapshot.json")

    lines = [
        "# Final Project Report",
        "",
        "## Objective",
        "",
        "Mobile-efficient 4× super-resolution under a deployment-oriented efficiency study:",
        "architecture, offline SwinIR distillation, and backend-aware benchmarking.",
        "",
        "## Locked Assumptions",
        "",
        "- Training can be expensive; only the student is deployed at inference time.",
        "- SwinIR is an offline teacher, not part of the deployment graph.",
        "- Efficiency claims use mobile-proxy metrics (params, FLOPs, latency), not phone hardware.",
        "",
        "## Main Results (Benchmark Average)",
        "",
        "| Model | Avg PSNR ↑ | Avg SSIM ↑ | Avg LPIPS ↓ | Params | FLOPs (G) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lpips = f"{r['avg_lpips']:.4f}" if r["avg_lpips"] is not None else "—"
        params = f"{r['params']:,}" if r["params"] else "—"
        flops = f"{r['flops_g']:.3f}" if r["flops_g"] else "—"
        lines.append(
            f"| {r['name']} | {r['avg_psnr']:.3f} | {r['avg_ssim']:.4f} | {lpips} | {params} | {flops} |"
        )

    if latency_audit:
        lines.extend([
            "",
            "## Audited Latency (CUDA Events, LR 180×180)",
            "",
            "| Model | FP32 median (ms) | FP16 median (ms) |",
            "|---|---:|---:|",
        ])
        for name, entry in latency_audit.items():
            if name in {"input_lr", "protocol"}:
                continue
            fp32 = entry.get("fp32", {}).get("median_ms")
            fp16 = entry.get("fp16", {})
            fp16_med = fp16.get("median_ms") if isinstance(fp16, dict) else None
            fp16_str = f"{fp16_med:.2f}" if fp16_med is not None else "—"
            lines.append(f"| {name} | {fp32:.2f} | {fp16_str} |")
        lines.append("")
        lines.append(
            "> Under audited CUDA timing, FP16 does not uniformly beat FP32. "
            "FSRCNN FP16 is not dramatically slower than FP32 (unlike the earlier un-audited table). "
            "MobileSRNet+KD shows a modest FP16 median advantage."
        )

    if kd_summary:
        ov = kd_summary.get("overall", {})
        lines.extend([
            "",
            "## KD Per-Image Analysis (219 images)",
            "",
            f"- Mean ΔPSNR: **{ov.get('mean_delta_psnr', 0):+.4f} dB**",
            f"- Median ΔPSNR: **{ov.get('median_delta_psnr', 0):+.4f} dB**",
            f"- Images improved (PSNR): **{ov.get('pct_improved_psnr', 0):.1f}%**",
            f"- Mean ΔLPIPS: **{ov.get('mean_delta_lpips', 0):+.4f}** (lower is better)",
            f"- Images improved (LPIPS): **{ov.get('pct_improved_lpips', 0):.1f}%**",
            "",
            "KD improves perceptual similarity more clearly than average PSNR. "
            "Urban100 gains are smaller than Set5/Set14 but still positive on most images.",
        ])

    lines.extend([
        "",
        "## Per-Dataset PSNR (dB)",
        "",
        "| Model | Set5 | Set14 | BSD100 | Urban100 |",
        "|---|---:|---:|---:|---:|",
    ])
    for r in rows:
        m = r["metrics"]
        lines.append(
            f"| {r['name']} | {m.get('Set5', {}).get('psnr', 0):.2f} | "
            f"{m.get('Set14', {}).get('psnr', 0):.2f} | "
            f"{m.get('BSD100', {}).get('psnr', 0):.2f} | "
            f"{m.get('Urban100', {}).get('psnr', 0):.2f} |"
        )

    lines.extend([
        "",
        "## Experiment Status",
        "",
        "- Phase 0 baseline snapshot: `results/exp_runs/baseline_snapshot.json`",
        "- Phase 1 GPU probe: `results/exp_runs/gpu_probe.csv`",
        "- Fair-budget retraining: in progress under `results/exp_runs/`",
        "- KD isolation (λ=0 vs 0.2): in progress",
        "- Qualitative panels: `results/qualitative/`",
        "",
        "## Limitations",
        "",
        "- Classical bicubic 4× SR only; real mobile degradations not modeled.",
        "- RTX 4060 CUDA is a deployment proxy, not mobile hardware.",
        "- Average PSNR is an unweighted mean over benchmark datasets.",
        "- SwinIR quality gap remains large; KD partially closes it.",
    ])

    report_path = Path(args.report_path)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
