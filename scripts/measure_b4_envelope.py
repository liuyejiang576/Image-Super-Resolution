#!/usr/bin/env python3
"""B4 Gate-1: paired multi-session phone envelope for Lite-sep vs ECBSR-fused.

Official protocol: warmup=50, iters=300, NCNN Vulkan FP16, LR 320×180.
Alternates order across sessions. Differences inside E_med / E_p90 are ties.

  export PATH=$HOME/miniforge3/bin:$HOME/android/platform-tools:$PATH
  python scripts/measure_b4_envelope.py
  python scripts/measure_b4_envelope.py --sessions 3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_extra = [
    str(Path.home() / "miniforge3/bin"),
    str(Path.home() / "android/platform-tools"),
]
os.environ["PATH"] = os.pathsep.join(_extra + [os.environ.get("PATH", "")])

NCNN_DIR = PROJECT_ROOT / "deploy/artifacts/ncnn"
RESULTS_DIR = PROJECT_ROOT / "deploy/artifacts/results"
EXP_RESULTS = PROJECT_ROOT / "results/exp_runs"
PARSE_BLOBS = PROJECT_ROOT / "scripts/parse_ncnn_blobs.py"
BENCH_BIN = PROJECT_ROOT / "deploy/android/sr_bench/build/sr_bench"
DEVICE_DIR = "/data/local/tmp/sr_bench"
ADB = Path.home() / "android/platform-tools/adb"
if not ADB.exists():
    ADB = Path("adb")
LIBOMP = (
    Path.home()
    / "android/ndk/android-ndk-r26d/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/17/lib/linux/aarch64/libomp.so"
)

# Existing deploy graphs (do not re-export).
MODELS = {
    "lite_sep": {
        "label": "SepResSR-Lite sep",
        "param": NCNN_DIR / "mobile_srnet_base_sep_deploy_720p.param",
        "bin": NCNN_DIR / "mobile_srnet_base_sep_deploy_720p.bin",
        # Fallback if sep-named artifact missing:
        "fallback_param": NCNN_DIR / "mobile_srnet_base_deploy_720p.param",
        "fallback_bin": NCNN_DIR / "mobile_srnet_base_deploy_720p.bin",
    },
    "ecbsr_fused": {
        "label": "ECBSR-M10C16 fused",
        "param": NCNN_DIR / "ecbsr_m10c16_fused_deploy_720p_dryrun.param",
        "bin": NCNN_DIR / "ecbsr_m10c16_fused_deploy_720p_dryrun.bin",
        "fallback_param": None,
        "fallback_bin": None,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="B4 phone measurement envelope")
    p.add_argument("--sessions", type=int, default=3)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--lr-w", type=int, default=320)
    p.add_argument("--lr-h", type=int, default=180)
    p.add_argument("--skip-push", action="store_true")
    return p.parse_args()


def adb(*args: str, capture: bool = True) -> str:
    cmd = [str(ADB), *args]
    if capture:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    subprocess.check_call(cmd)
    return ""


def resolve_paths(key: str) -> tuple[Path, Path]:
    meta = MODELS[key]
    param, binf = meta["param"], meta["bin"]
    if param.exists() and binf.exists():
        return param, binf
    fb_p, fb_b = meta.get("fallback_param"), meta.get("fallback_bin")
    if fb_p and fb_b and fb_p.exists() and fb_b.exists():
        return fb_p, fb_b
    raise FileNotFoundError(f"Missing NCNN for {key}: {param}")


def parse_blobs(param_path: Path) -> tuple[str, str]:
    out = subprocess.check_output(
        [sys.executable, str(PARSE_BLOBS), str(param_path)], text=True
    ).strip()
    return out.split("\t")


def ensure_bench(skip_push: bool) -> None:
    if not BENCH_BIN.exists():
        raise SystemExit(f"Missing {BENCH_BIN}")
    adb("shell", f"mkdir -p {DEVICE_DIR}/models", capture=False)
    if not skip_push:
        adb("push", str(BENCH_BIN), f"{DEVICE_DIR}/sr_bench", capture=False)
        adb("shell", f"chmod +x {DEVICE_DIR}/sr_bench", capture=False)
        if LIBOMP.exists():
            adb("push", str(LIBOMP), f"{DEVICE_DIR}/libomp.so", capture=False)


def bench_one(
    key: str,
    param: Path,
    binf: Path,
    in_blob: str,
    out_blob: str,
    lr_w: int,
    lr_h: int,
    warmup: int,
    iters: int,
) -> dict:
    stem = param.name
    remote_param = f"{DEVICE_DIR}/models/{stem}"
    remote_bin = f"{DEVICE_DIR}/models/{binf.name}"
    adb("push", str(param), remote_param, capture=False)
    adb("push", str(binf), remote_bin, capture=False)
    cmd = [
        f"{DEVICE_DIR}/sr_bench",
        "--param", remote_param,
        "--bin", remote_bin,
        "--in-blob", in_blob,
        "--out-blob", out_blob,
        "--input-w", str(lr_w),
        "--input-h", str(lr_h),
        "--warmup", str(warmup),
        "--iters", str(iters),
        "--fp16",
        "--vulkan",
    ]
    raw = adb("shell", f"LD_LIBRARY_PATH={DEVICE_DIR} " + " ".join(cmd))
    bench = json.loads(raw.strip().splitlines()[-1])
    return {
        "model_key": key,
        "label": MODELS[key]["label"],
        "ncnn_param": str(param.relative_to(PROJECT_ROOT)),
        "ncnn_bin": str(binf.relative_to(PROJECT_ROOT)),
        "median_ms": float(bench["median_ms"]),
        "p90_ms": float(bench["p90_ms"]),
        "fps": float(bench.get("fps", 0.0)),
        "peak_memory_kb": bench.get("peak_memory_kb"),
        "warmup": warmup,
        "iters": iters,
        "raw": bench,
    }


def cross_session_range(values: list[float]) -> float:
    return max(values) - min(values) if values else float("nan")


def main() -> None:
    args = parse_args()
    try:
        adb("get-state")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit(f"adb unavailable: {exc}") from exc

    resolved = {k: resolve_paths(k) for k in MODELS}
    blobs = {k: parse_blobs(p) for k, (p, _) in resolved.items()}
    ensure_bench(args.skip_push)

    print("=== B4 phone envelope (Lite-sep vs ECBSR-fused) ===")
    print(f"  sessions={args.sessions} protocol={args.warmup}/{args.iters} LR {args.lr_w}x{args.lr_h}")

    sessions: list[dict] = []
    for s in range(args.sessions):
        # Alternate order: even Lite→ECBSR, odd ECBSR→Lite
        order = ["lite_sep", "ecbsr_fused"] if s % 2 == 0 else ["ecbsr_fused", "lite_sep"]
        print(f"\n--- session {s + 1}/{args.sessions} order={' → '.join(order)} ---")
        runs = []
        for key in order:
            param, binf = resolved[key]
            in_b, out_b = blobs[key]
            print(f"  bench {key} ...")
            row = bench_one(
                key, param, binf, in_b, out_b, args.lr_w, args.lr_h, args.warmup, args.iters
            )
            print(f"    med={row['median_ms']:.2f} p90={row['p90_ms']:.2f}")
            runs.append(row)
        sessions.append(
            {
                "session_index": s,
                "order": order,
                "timestamp": datetime.now().astimezone().isoformat(),
                "runs": runs,
                "excluded": False,
                "exclusion_reason": None,
            }
        )

    by_model: dict[str, dict] = {}
    for key in MODELS:
        meds = []
        p90s = []
        for sess in sessions:
            if sess.get("excluded"):
                continue
            for r in sess["runs"]:
                if r["model_key"] == key:
                    meds.append(r["median_ms"])
                    p90s.append(r["p90_ms"])
        by_model[key] = {
            "label": MODELS[key]["label"],
            "median_ms": meds,
            "p90_ms": p90s,
            "median_range": cross_session_range(meds),
            "p90_range": cross_session_range(p90s),
            "median_mean": sum(meds) / len(meds) if meds else float("nan"),
            "p90_mean": sum(p90s) / len(p90s) if p90s else float("nan"),
        }

    e_med = max(by_model[k]["median_range"] for k in MODELS)
    e_p90 = max(by_model[k]["p90_range"] for k in MODELS)

    payload = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "task": "B4_measurement_envelope",
        "gate": "Gate-1-phone-envelope",
        "protocol": {
            "warmup": args.warmup,
            "iters": args.iters,
            "lr_w": args.lr_w,
            "lr_h": args.lr_h,
            "fp16": True,
            "vulkan": True,
            "backend": "ncnn_vulkan",
            "device_dir": DEVICE_DIR,
        },
        "sessions": sessions,
        "by_model": by_model,
        "E_med_ms": e_med,
        "E_p90_ms": e_p90,
        "definition": (
            "E_med / E_p90 = max same-model cross-session range over Lite-sep and ECBSR-fused. "
            "Candidate vs ECBSR differences inside the corresponding envelope are ties."
        ),
        "historical_note": (
            "Prior Lite A0 vs B1 median gap ~3.1 ms is a lower-bound sanity check; "
            "this file supersedes that estimate for B4 decisions."
        ),
    }

    EXP_RESULTS.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    text = json.dumps(payload, indent=2) + "\n"
    out_main = EXP_RESULTS / "b4_measurement_envelope.json"
    out_ts = RESULTS_DIR / f"b4_measurement_envelope_{ts}.json"
    out_latest = RESULTS_DIR / "b4_measurement_envelope_latest.json"
    out_main.write_text(text, encoding="utf-8")
    out_ts.write_text(text, encoding="utf-8")
    out_latest.write_text(text, encoding="utf-8")

    print("\n=== envelope summary ===")
    for key, stats in by_model.items():
        print(
            f"  {key}: med {stats['median_ms']} range={stats['median_range']:.3f} | "
            f"p90 {stats['p90_ms']} range={stats['p90_range']:.3f}"
        )
    print(f"  E_med={e_med:.3f} ms  E_p90={e_p90:.3f} ms")
    print(f"Wrote {out_main.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
