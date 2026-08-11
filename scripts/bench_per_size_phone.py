#!/usr/bin/env python3
"""Measure real phone latency at every unique benchmark LR size.

Pairs with per-image PSNR via shared (dataset, stem) → (w,h) → measured med.

  export PATH=$HOME/miniforge3/bin:$HOME/android/platform-tools:$PATH
  conda activate cv_env
  python scripts/bench_per_size_phone.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "deploy/artifacts/results"
NCNN_DIR = PROJECT_ROOT / "deploy/artifacts/ncnn"
BENCH_BIN = PROJECT_ROOT / "deploy/android/sr_bench/build/sr_bench"
DEVICE_DIR = "/data/local/tmp/sr_bench"
DATA_ROOT = PROJECT_ROOT.parent / "data" / "benchmarks"
PARSE_BLOBS = PROJECT_ROOT / "scripts/parse_ncnn_blobs.py"

_extra = [
    str(Path.home() / "miniforge3/bin"),
    str(Path.home() / "android/platform-tools"),
]
os.environ["PATH"] = os.pathsep.join(_extra + [os.environ.get("PATH", "")])

ADB = Path.home() / "android/platform-tools/adb"
if not ADB.exists():
    ADB = Path("adb")
LIBOMP = (
    Path.home()
    / "android/ndk/android-ndk-r26d/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/17/lib/linux/aarch64/libomp.so"
)

# Official fused audit graphs (fully convolutional — any LR size).
MODELS = [
    {
        "label": "FSRCNN",
        "model_id": "fsrcnn",
        "param": NCNN_DIR / "fsrcnn_audit_180.param",
        "bin": NCNN_DIR / "fsrcnn_audit_180.bin",
    },
    {
        "label": "PECSR",
        "model_id": "pecsr",
        "param": NCNN_DIR / "sepres_v2_c16n10_fused_audit_180.param",
        "bin": NCNN_DIR / "sepres_v2_c16n10_fused_audit_180.bin",
    },
    {
        "label": "ECBSR",
        "model_id": "ecbsr",
        "param": NCNN_DIR / "ecbsr_m10c16_fused_audit_180.param",
        "bin": NCNN_DIR / "ecbsr_m10c16_fused_audit_180.bin",
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-LR-size phone latency sweep")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--skip-push", action="store_true")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip (model,w,h) already in latest JSON",
    )
    return p.parse_args()


def adb(*args: str, capture: bool = True) -> str:
    cmd = [str(ADB), *args]
    if capture:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    subprocess.check_call(cmd)
    return ""


def adb_ok() -> bool:
    try:
        adb("get-state")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def parse_blobs(param_path: Path) -> tuple[str, str]:
    out = subprocess.check_output(
        [sys.executable, str(PARSE_BLOBS), str(param_path)], text=True
    ).strip()
    return out.split("\t")


def collect_image_sizes() -> tuple[list[dict], list[tuple[int, int]]]:
    """Return image rows + unique (w,h) sorted."""
    rows: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for ds in ("Set5", "Set14", "BSD100", "Urban100"):
        d = DATA_ROOT / ds / "image_SRF_4"
        for p in sorted(d.glob("*_LR.png")):
            stem = p.name.replace("_LR.png", "")
            with Image.open(p) as im:
                w, h = im.size
            rows.append({"dataset": ds, "stem": stem, "lr_w": w, "lr_h": h})
            seen.add((w, h))
    sizes = sorted(seen, key=lambda wh: (wh[0] * wh[1], wh[0], wh[1]))
    return rows, sizes


def patch_ecbsr_param(src: Path, dst: Path, lr_w: int, lr_h: int) -> None:
    """ECBSR residual Reshape is baked at export size; rewrite to (w,h)."""
    text = src.read_text(encoding="utf-8")
    old = "Reshape                  reshape_25               1 1 4 5 0=180 1=180 2=48"
    new = f"Reshape                  reshape_25               1 1 4 5 0={lr_w} 1={lr_h} 2=48"
    if old not in text:
        raise RuntimeError(f"expected fixed 180 reshape in {src}")
    dst.write_text(text.replace(old, new), encoding="utf-8")


def run_bench(
    remote_param: str,
    remote_bin: str,
    in_blob: str,
    out_blob: str,
    lr_w: int,
    lr_h: int,
    warmup: int,
    iters: int,
) -> dict:
    cmd = [
        f"{DEVICE_DIR}/sr_bench",
        "--param",
        remote_param,
        "--bin",
        remote_bin,
        "--in-blob",
        in_blob,
        "--out-blob",
        out_blob,
        "--input-w",
        str(lr_w),
        "--input-h",
        str(lr_h),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--fp16",
        "--vulkan",
    ]
    raw = adb("shell", f"LD_LIBRARY_PATH={DEVICE_DIR} " + " ".join(cmd))
    return json.loads(raw.strip().splitlines()[-1])


def main() -> None:
    args = parse_args()
    if not adb_ok():
        raise SystemExit("adb device not ready — deploy/check_usb.sh")
    if not BENCH_BIN.exists():
        raise SystemExit(f"missing {BENCH_BIN}")
    for m in MODELS:
        if not m["param"].exists() or not m["bin"].exists():
            raise SystemExit(f"missing NCNN for {m['label']}: {m['param']}")

    images, sizes = collect_image_sizes()
    print(f"images={len(images)} unique_sizes={len(sizes)} models={len(MODELS)}")

    latest_path = RESULTS_DIR / "per_size_phone_latest.json"
    done: dict[str, dict] = {}
    if args.resume and latest_path.exists():
        prev = json.loads(latest_path.read_text(encoding="utf-8"))
        for row in prev.get("size_results", []):
            key = f"{row['label']}|{row['lr_w']}x{row['lr_h']}"
            done[key] = row
        print(f"resume: {len(done)} existing cells")

    adb("shell", f"mkdir -p {DEVICE_DIR}/models", capture=False)
    if not args.skip_push:
        adb("push", str(BENCH_BIN), f"{DEVICE_DIR}/sr_bench", capture=False)
        adb("shell", f"chmod +x {DEVICE_DIR}/sr_bench", capture=False)
        if LIBOMP.exists():
            adb("push", str(LIBOMP), f"{DEVICE_DIR}/libomp.so", capture=False)

    size_results: list[dict] = list(done.values())
    total = len(MODELS) * len(sizes)
    n_done = 0

    for m in MODELS:
        in_blob, out_blob = parse_blobs(m["param"])
        remote_b = f"{DEVICE_DIR}/models/{m['bin'].name}"
        adb("push", str(m["bin"]), remote_b, capture=False)
        remote_p_default = f"{DEVICE_DIR}/models/{m['param'].name}"
        if m["label"] != "ECBSR":
            adb("push", str(m["param"]), remote_p_default, capture=False)
        print(f"\n=== {m['label']} blobs {in_blob}->{out_blob} ===")

        for w, h in sizes:
            key = f"{m['label']}|{w}x{h}"
            n_done += 1
            if key in done:
                print(f"  [{n_done}/{total}] skip {w}x{h}")
                continue

            if m["label"] == "ECBSR":
                local_param = RESULTS_DIR / f"_tmp_ecbsr_{w}x{h}.param"
                patch_ecbsr_param(m["param"], local_param, w, h)
                remote_p = f"{DEVICE_DIR}/models/ecbsr_{w}x{h}.param"
                adb("push", str(local_param), remote_p, capture=False)
                local_param.unlink(missing_ok=True)
            else:
                remote_p = remote_p_default

            bench = run_bench(
                remote_p, remote_b, in_blob, out_blob, w, h, args.warmup, args.iters
            )
            row = {
                "label": m["label"],
                "model_id": m["model_id"],
                "lr_w": w,
                "lr_h": h,
                "median_ms": bench["median_ms"],
                "p90_ms": bench["p90_ms"],
                "fps": bench.get("fps"),
                "peak_memory_kb": bench.get("peak_memory_kb"),
            }
            size_results.append(row)
            done[key] = row
            print(
                f"  [{n_done}/{total}] {w}x{h}: "
                f"med={row['median_ms']:.2f} p90={row['p90_ms']:.2f}"
            )
            # checkpoint after each cell
            payload = {
                "timestamp": datetime.now().astimezone().isoformat(),
                "task": "per_size_phone_latency",
                "protocol": {
                    "warmup": args.warmup,
                    "iters": args.iters,
                    "fp16": True,
                    "vulkan": True,
                    "backend": "ncnn_vulkan_fp16",
                    "note": "one measured median per unique LR (w,h); images share size cells",
                },
                "size_results": size_results,
                "images": images,
            }
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"per_size_phone_{ts}.json"
    shutil.copy2(latest_path, out_path)
    print(f"\nWrote {out_path.relative_to(PROJECT_ROOT)}")
    print(f"Also {latest_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
