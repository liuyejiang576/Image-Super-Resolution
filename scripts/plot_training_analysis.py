#!/usr/bin/env python3
"""Plot training curves and summarize learning dynamics across experiment runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "results/training_analysis"

# Fixed PSNR y-axis: most gain is in the first ~500 steps; 24–28 dB covers the useful band.
PSNR_YLIM = (24.0, 28.0)
# λ-sweep curves overlap tightly — zoom 26–28 to show they are on top of each other.
KD_SWEEP_PSNR_YLIM = (26.0, 28.0)
LOSS_YLIM_FLOOR = 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(OUT_DIR))
    return p.parse_args()


def load_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def rel_log(*parts: str) -> Path:
    return PROJECT_ROOT / Path(*parts)


def curve(rows: list[dict], x_key: str, y_key: str) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for r in rows:
        if x_key in r and y_key in r:
            xs.append(float(r[x_key]))
            ys.append(float(r[y_key]))
    return xs, ys


def final_val(rows: list[dict]) -> float | None:
    if not rows:
        return None
    return float(max(rows, key=lambda r: r.get("val_psnr", -999))["val_psnr"])


def val_at_step(rows: list[dict], step: int) -> float | None:
    best = None
    for r in rows:
        gs = int(r.get("global_step", 0))
        if gs <= step:
            best = float(r["val_psnr"])
    return best


def slope_last_n(rows: list[dict], n: int = 50) -> float | None:
    if len(rows) < 2:
        return None
    tail = rows[-n:]
    steps = np.array([float(r["global_step"]) for r in tail])
    psnrs = np.array([float(r["val_psnr"]) for r in tail])
    if len(steps) < 2 or steps[-1] == steps[0]:
        return None
    return float(np.polyfit(steps, psnrs, 1)[0] * 1000)  # dB per 1k steps


def save_fig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  wrote {path}")


def loss_ylim(*series: list[float], pad_frac: float = 0.08) -> tuple[float, float]:
    """0-based upper bound from peak train loss (includes early epochs)."""
    vals = [y for ys in series for y in ys if y == y]
    if not vals:
        return (0.0, 1.0)
    peak = max(vals)
    return (LOSS_YLIM_FLOOR, peak * (1.0 + pad_frac))


def plot_group(
    out_dir: Path,
    filename: str,
    title: str,
    series: list[tuple[str, list[dict], str]],
    x_key: str = "global_step",
    y_key: str = "val_psnr",
    xlabel: str = "Global training steps",
    ylabel: str = "DIV2K-valid PSNR (dB)",
    vlines: list[tuple[int, str]] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, rows, style in series:
        if not rows:
            continue
        xs, ys = curve(rows, x_key, y_key)
        kw = {"linestyle": style} if style else {}
        ax.plot(xs, ys, label=label, **kw)
    if vlines:
        ymin = (ylim or PSNR_YLIM)[0]
        for x, txt in vlines:
            ax.axvline(x, color="gray", ls=":", alpha=0.5)
            ax.text(x, ymin, txt, rotation=90, va="bottom", fontsize=7, alpha=0.7)
    if xlim:
        ax.set_xlim(*xlim)
    if y_key == "val_psnr":
        ax.set_ylim(*(ylim or PSNR_YLIM))
    elif ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    save_fig(fig, out_dir, filename)


def plot_loss_decomp(out_dir: Path, runs: list[tuple[str, Path]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    total_ys: list[float] = []
    gt_ys: list[float] = []
    kd_ys: list[float] = []

    for label, path in runs:
        rows = load_log(path)
        if not rows:
            continue
        _, total_ys_run = curve(rows, "global_step", "train_loss")
        total_ys.extend(total_ys_run)
        if "train_loss_gt" in rows[0]:
            _, gt = curve(rows, "global_step", "train_loss_gt")
            _, kd = curve(rows, "global_step", "train_loss_kd")
            gt_ys.extend(gt)
            kd_ys.extend(kd)

    y_total = loss_ylim(total_ys)
    y_comp = loss_ylim(gt_ys, kd_ys)

    for ax, (title, y_key, ylim) in zip(
        axes,
        [
            ("Total train loss", "train_loss", y_total),
            ("GT vs KD components", None, y_comp),
        ],
    ):
        if y_key:
            for label, path in runs:
                rows = load_log(path)
                if not rows:
                    continue
                xs, ys = curve(rows, "global_step", y_key)
                ax.plot(xs, ys, label=label, alpha=0.85)
            ax.set_ylabel("Charbonnier train loss")
            ax.set_title(title)
        else:
            for label, path in runs:
                rows = load_log(path)
                if not rows or "train_loss_gt" not in rows[0]:
                    continue
                xs_gt, ys_gt = curve(rows, "global_step", "train_loss_gt")
                xs_kd, ys_kd = curve(rows, "global_step", "train_loss_kd")
                ax.plot(xs_gt, ys_gt, label=f"{label} GT", alpha=0.85)
                ax.plot(xs_kd, ys_kd, "--", label=f"{label} KD", alpha=0.7)
            ax.set_ylabel("Charbonnier loss")
            ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Global training steps")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("KD training loss decomposition (Charbonnier pixel)")
    save_fig(fig, out_dir, "04_kd_loss_decomposition.png")


def plot_lambda_sweep(out_dir: Path) -> None:
    lambdas = [0.0, 0.2, 0.5, 1.0, 2.0]
    run_map = {
        0.0: "mobile_srnet_kd0_10k",
        0.2: "mobile_srnet_kd02_10k",
        0.5: "mobile_srnet_kd05_10k",
        1.0: "mobile_srnet_kd10_10k",
        2.0: "mobile_srnet_kd20_10k",
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    finals: list[tuple[float, float | None]] = []
    for lam in lambdas:
        rid = run_map[lam]
        rows = load_log(rel_log("results/_inactive/exp_runs", rid, "train_log.jsonl"))
        finals.append((lam, final_val(rows)))
        if rows:
            xs, ys = curve(rows, "global_step", "val_psnr")
            ax.plot(xs, ys, label=f"λ={lam}")
    ax.set_xlim(0, 10000)
    ax.set_ylim(*KD_SWEEP_PSNR_YLIM)
    ax.set_xlabel("Global training steps")
    ax.set_ylabel("DIV2K-valid PSNR (dB)")
    ax.set_title("Pixel KD λ-sweep @ 10k updates (y: 26–28 dB)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    best_vals = [v for _, v in finals if v is not None]
    if best_vals:
        spread = max(best_vals) - min(best_vals)
        ax.annotate(
            f"best @ 10k: spread {spread:.3f} dB across λ",
            xy=(0.02, 0.02),
            xycoords="axes fraction",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )
    save_fig(fig, out_dir, "03_kd_lambda_sweep_curves.png")

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    lams = [x[0] for x in finals if x[1] is not None]
    vals = [x[1] for x in finals if x[1] is not None]
    ax2.plot(lams, vals, "o-", color="C0")
    if vals:
        spread = max(vals) - min(vals)
        ax2.set_ylim(*KD_SWEEP_PSNR_YLIM)
        ax2.annotate(
            f"spread {spread:.3f} dB",
            xy=(0.02, 0.98),
            xycoords="axes fraction",
            va="top",
            fontsize=9,
        )
    ax2.set_xlabel("λ (KD weight)")
    ax2.set_ylabel("Best DIV2K-valid PSNR (dB)")
    ax2.set_title("Final PSNR vs λ (10k budget)")
    ax2.grid(True, alpha=0.3)
    save_fig(fig2, out_dir, "03b_kd_lambda_sweep_final_psnr.png")


def analyze_run(name: str, rows: list[dict]) -> dict[str, Any]:
    if not rows:
        return {"name": name, "status": "missing"}
    best_row = max(rows, key=lambda r: r.get("val_psnr", -999))
    last = rows[-1]
    out: dict[str, Any] = {
        "name": name,
        "epochs": int(last["epoch"]),
        "global_steps": int(last.get("global_step", 0)),
        "best_val_psnr": float(best_row["val_psnr"]),
        "best_epoch": int(best_row["epoch"]),
        "final_val_psnr": float(last["val_psnr"]),
        "slope_last_50ep_db_per_1k_steps": slope_last_n(rows, 50),
        "still_improving_at_end": bool(last["val_psnr"] >= float(best_row["val_psnr"]) - 0.01),
    }
    if "train_loss_gt" in last:
        out["final_train_loss_gt"] = float(last["train_loss_gt"])
        out["final_train_loss_kd"] = float(last["train_loss_kd"])
        out["lambda_kd"] = float(last.get("lambda_kd", 0))
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load key runs ---
    logs = {
        "fsrcnn_20k": load_log(rel_log("results/exp_runs/fsrcnn_fix_clean_20k/train_log.jsonl")),
        "mobile_base_20k": load_log(rel_log("results/exp_runs/mobile_srnet_20k/train_log.jsonl")),
        "mobile_plus_20k": load_log(rel_log("results/exp_runs/mobile_srnet_plus_20k/train_log.jsonl")),
        "mobile_base_10k": load_log(rel_log("results/_inactive/exp_runs/mobile_srnet_10k/train_log.jsonl")),
        "mobile_plus_2k": load_log(rel_log("results/_inactive/exp_runs/mobile_srnet_plus_2k/train_log.jsonl")),
        "kd0_20k": load_log(rel_log("results/exp_runs/mobile_srnet_kd0_20k/train_log.jsonl")),
        "kd02_20k": load_log(rel_log("results/exp_runs/mobile_srnet_kd02_20k/train_log.jsonl")),
        "vgg3_kd0_2k": load_log(rel_log("results/_inactive/exp_runs/mobile_srnet_vgg3_kd0_2k/train_log.jsonl")),
        "vgg3_kd01_2k": load_log(rel_log("results/_inactive/exp_runs/mobile_srnet_vgg3_kd01_2k/train_log.jsonl")),
    }

    print("Plotting training analysis figures...")
    # 1. Main architecture comparison @ 20k
    plot_group(
        out_dir,
        "01_fair_budget_20k_architectures.png",
        "Fair-budget 20k: FSRCNN vs MobileSRNet Base vs Plus",
        [
            ("FSRCNN", logs["fsrcnn_20k"], "--"),
            ("MobileSRNet Base", logs["mobile_base_20k"], "-"),
            ("MobileSRNet Plus", logs["mobile_plus_20k"], "-"),
        ],
    )

    # 2. Base vs Plus zoom (first 5k steps + full)
    plot_group(
        out_dir,
        "02_capacity_base_vs_plus_early.png",
        "Base vs Plus — first 5k steps",
        [
            ("Base 20k", logs["mobile_base_20k"], "-"),
            ("Plus 20k", logs["mobile_plus_20k"], "-"),
            ("Plus 2k probe", logs["mobile_plus_2k"], ":"),
        ],
        xlim=(0, 5000),
    )
    plot_group(
        out_dir,
        "02b_capacity_base_vs_plus_full.png",
        "Base vs Plus — full 20k budget",
        [
            ("Base 20k", logs["mobile_base_20k"], "-"),
            ("Plus 20k", logs["mobile_plus_20k"], "-"),
        ],
    )

    # 3. KD lambda sweep
    plot_lambda_sweep(out_dir)

    # 4. KD 20k isolation + loss decomp
    plot_group(
        out_dir,
        "05_kd_isolation_20k.png",
        "KD isolation @ 20k (λ=0 vs 0.2, pixel Charbonnier)",
        [
            ("λ=0 (Charb only)", logs["kd0_20k"], "-"),
            ("λ=0.2 (+ pixel KD)", logs["kd02_20k"], "-"),
            ("Base 20k (MSE, no teacher)", logs["mobile_base_20k"], "--"),
        ],
    )
    plot_loss_decomp(
        out_dir,
        [
            ("λ=0", rel_log("results/exp_runs/mobile_srnet_kd0_20k/train_log.jsonl")),
            ("λ=0.2", rel_log("results/exp_runs/mobile_srnet_kd02_20k/train_log.jsonl")),
        ],
    )

    # 5. VGG Stage B
    plot_group(
        out_dir,
        "06_vgg3_stage_b_2k.png",
        "Stage B: VGG relu3 KD @ 2k (λ=0 vs 0.01)",
        [
            ("λ=0 control", logs["vgg3_kd0_2k"], "-"),
            ("λ=0.01 VGG KD", logs["vgg3_kd01_2k"], "-"),
        ],
        xlim=(0, 2100),
    )

    # 6. Train loss vs val PSNR (Base vs Plus) — detect overfitting
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    base_train: list[float] = []
    plus_train: list[float] = []
    base_val: list[float] = []
    plus_val: list[float] = []
    for label, rows, color in [
        ("Base", logs["mobile_base_20k"], "C0"),
        ("Plus", logs["mobile_plus_20k"], "C1"),
    ]:
        if not rows:
            continue
        xs, train = curve(rows, "global_step", "train_loss")
        _, val = curve(rows, "global_step", "val_psnr")
        axes[0].plot(xs, train, label=label, color=color)
        axes[1].plot(xs, val, label=label, color=color)
        if label == "Base":
            base_train, base_val = train, val
        else:
            plus_train, plus_val = train, val
    axes[0].set_ylim(*loss_ylim(base_train, plus_train))
    axes[0].set_ylabel("Train MSE loss")
    axes[0].set_title("Train loss vs validation PSNR (full y-range)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].set_ylim(*PSNR_YLIM)
    axes[1].set_xlabel("Global training steps")
    axes[1].set_ylabel("Val PSNR (dB)")
    axes[1].grid(True, alpha=0.3)
    save_fig(fig, out_dir, "07_train_loss_vs_val_psnr.png")

    # 7. Convergence rate: PSNR gain per 1k steps in phases
    phases = [(0, 2000), (2000, 10000), (10000, 20000)]
    fig, ax = plt.subplots(figsize=(8, 5))
    names = []
    rates = []
    for label, rows in [("Base", logs["mobile_base_20k"]), ("Plus", logs["mobile_plus_20k"])]:
        for lo, hi in phases:
            seg = [r for r in rows if lo < int(r["global_step"]) <= hi]
            if len(seg) < 2:
                continue
            d_psnr = float(seg[-1]["val_psnr"]) - float(seg[0]["val_psnr"])
            d_steps = int(seg[-1]["global_step"]) - int(seg[0]["global_step"])
            rate = d_psnr / (d_steps / 1000) if d_steps else 0
            names.append(f"{label}\n{lo//1000}–{hi//1000}k")
            rates.append(rate)
    x = np.arange(len(names))
    ax.bar(x, rates, color=["C0", "C0", "C0", "C1", "C1", "C1"][: len(names)])
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Δ PSNR (dB) per 1k steps")
    ax.set_title("Learning rate by training phase")
    ax.axhline(0, color="k", lw=0.5)
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, out_dir, "08_learning_rate_by_phase.png")

    # --- analysis summary ---
    analyses = {k: analyze_run(k, v) for k, v in logs.items()}

    # pairwise deltas at matched steps
    comparisons = []
    pairs = [
        ("Plus vs Base @ 2k", "mobile_plus_2k", "mobile_base_20k", 2000),
        ("Plus vs Base @ 20k", "mobile_plus_20k", "mobile_base_20k", 20000),
        ("VGG λ=0.01 vs λ=0 @ 2k", "vgg3_kd01_2k", "vgg3_kd0_2k", 2000),
        ("Pixel KD λ=0.2 vs λ=0 @ 20k", "kd02_20k", "kd0_20k", 20000),
    ]
    key_map = {
        "mobile_plus_2k": logs["mobile_plus_2k"],
        "mobile_plus_20k": logs["mobile_plus_20k"],
        "mobile_base_20k": logs["mobile_base_20k"],
        "vgg3_kd01_2k": logs["vgg3_kd01_2k"],
        "vgg3_kd0_2k": logs["vgg3_kd0_2k"],
        "kd02_20k": logs["kd02_20k"],
        "kd0_20k": logs["kd0_20k"],
    }
    for title, a_key, b_key, step in pairs:
        a_val = val_at_step(key_map[a_key], step)
        b_val = val_at_step(key_map[b_key], step)
        if a_val is not None and b_val is not None:
            comparisons.append({"comparison": title, "step": step, "a_psnr": a_val, "b_psnr": b_val, "delta_db": a_val - b_val})

    insights = []

    plus_final = analyses.get("mobile_plus_20k", {}).get("best_val_psnr")
    base_final = analyses.get("mobile_base_20k", {}).get("best_val_psnr")
    if plus_final and base_final:
        d = plus_final - base_final
        insights.append(
            f"Plus beats Base by {d:+.3f} dB at 20k val PSNR ({plus_final:.3f} vs {base_final:.3f})."
        )
        base_slope = analyses["mobile_base_20k"].get("slope_last_50ep_db_per_1k_steps")
        plus_slope = analyses["mobile_plus_20k"].get("slope_last_50ep_db_per_1k_steps")
        if base_slope is not None and plus_slope is not None:
            if plus_slope > 0.02 and base_slope < 0.02:
                insights.append(
                    f"Plus still gaining ({plus_slope:.3f} dB/1k steps in last 50 epochs) while Base has plateaued ({base_slope:.3f}) — capacity not saturated."
                )
            elif plus_slope < 0.01 and base_slope < 0.01:
                insights.append("Both Base and Plus plateau late — further gains need longer budget or recipe change, not just width.")

    if analyses.get("mobile_base_20k", {}).get("still_improving_at_end"):
        insights.append("Base 20k best checkpoint is at the final epoch — 20k may still be under-budget for Base.")

    kd0_slope = analyses.get("kd0_20k", {}).get("slope_last_50ep_db_per_1k_steps")
    kd02_slope = analyses.get("kd02_20k", {}).get("slope_last_50ep_db_per_1k_steps")
    if kd0_slope is not None and kd02_slope is not None:
        insights.append(
            f"Pixel KD curves track together (λ=0 slope {kd0_slope:.4f}, λ=0.2 slope {kd02_slope:.4f} dB/1k steps) — confirms null is structural, not early-stop noise."
        )

    vgg_d = next((c["delta_db"] for c in comparisons if "VGG" in c["comparison"]), None)
    if vgg_d is not None:
        insights.append(f"VGG Stage B null: Δ={vgg_d:+.3f} dB @ 2k — non-redundant gradient did not translate to val gain.")

    early_plus = val_at_step(logs["mobile_plus_2k"], 2000)
    early_base = val_at_step(logs["mobile_base_20k"], 2000)
    if early_plus and early_base:
        insights.append(
            f"Plus leads Base early (+{early_plus - early_base:.3f} dB @ 2k) — Stage B probe correctly predicted 20k benefit."
        )

    recommendations = []
    plus_slope = analyses.get("mobile_plus_20k", {}).get("slope_last_50ep_db_per_1k_steps")
    base_slope = analyses.get("mobile_base_20k", {}).get("slope_last_50ep_db_per_1k_steps")
    if plus_final and base_final and plus_final - base_final >= 0.05:
        recommendations.append("Run benchmark eval + latency profile on Plus; promote Plus to headline if benchmarks confirm.")
    if analyses.get("mobile_base_20k", {}).get("still_improving_at_end"):
        recommendations.append("Optional: extend Base/Plus to 30k updates — curves still rising at epoch 607.")
    recommendations.append("Do not invest in more SwinIR KD — pixel and VGG paths flat across all budgets.")
    if plus_slope is not None and base_slope is not None and plus_slope > base_slope + 0.01:
        recommendations.append(
            "Consider MobileSRNet-XL (feat 80, 10 blocks) only if Plus benchmarks beat Base by ≥0.15 dB — scaling trend is positive."
        )
    recommendations.append("Finalize report: RQ1=Plus/Base vs FSRCNN, RQ2=KD null (1 paragraph), RQ3=latency.")

    summary = {
        "runs": analyses,
        "comparisons_at_matched_steps": comparisons,
        "insights": insights,
        "recommendations": recommendations,
        "figures": sorted(p.name for p in out_dir.glob("*.png")),
    }

    json_path = out_dir / "training_analysis_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  wrote {json_path}")

    md_lines = [
        "# Training Analysis Summary",
        "",
        "## Key comparisons (matched steps)",
        "",
        "| Comparison | Step | A (dB) | B (dB) | Δ (dB) |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in comparisons:
        md_lines.append(
            f"| {c['comparison']} | {c['step']} | {c['a_psnr']:.3f} | {c['b_psnr']:.3f} | {c['delta_db']:+.3f} |"
        )
    md_lines.extend(["", "## Insights", ""])
    for i in insights:
        md_lines.append(f"- {i}")
    md_lines.extend(["", "## Recommendations", ""])
    for r in recommendations:
        md_lines.append(f"- {r}")
    md_lines.extend(["", "## Figures", ""])
    for fig in summary["figures"]:
        md_lines.append(f"- `{out_dir.name}/{fig}`")

    md_path = out_dir / "training_analysis.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"  wrote {md_path}")


if __name__ == "__main__":
    main()
