#!/usr/bin/env python3
"""Run on-device NCNN benchmark via adb (push models, bench, collect JSON)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NCNN_MANIFEST = PROJECT_ROOT / "deploy/artifacts/ncnn_manifest.json"
BENCH_BIN = PROJECT_ROOT / "deploy/android/sr_bench/build/sr_bench"
DEVICE_DIR = "/data/local/tmp/sr_bench"
RESULTS_DIR = PROJECT_ROOT / "deploy/artifacts/results"
PARSE_BLOBS = PROJECT_ROOT / "scripts/parse_ncnn_blobs.py"
ADB = Path.home() / "android/platform-tools/adb"
if not ADB.exists():
    ADB = Path("adb")
LIBOMP = (
    Path.home()
    / "android/ndk/android-ndk-r26d/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/17/lib/linux/aarch64/libomp.so"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mobile NCNN benchmark via adb")
    p.add_argument("--preset", choices=["all", "audit_180", "deploy_720p"], default="all")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--vulkan", action="store_true", default=True)
    p.add_argument("--no-vulkan", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-push", action="store_true", help="Skip pushing sr_bench/libomp (already on device)")
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
    in_blob, out_blob = out.split("\t")
    return in_blob, out_blob


def collect_device_info() -> dict:
    script = PROJECT_ROOT / "deploy/collect_device_info.sh"
    return json.loads(subprocess.check_output([str(script)], text=True))


def run_bench_remote(
    remote_param: str,
    remote_bin: str,
    in_blob: str,
    out_blob: str,
    lr_w: int,
    lr_h: int,
    warmup: int,
    iters: int,
    fp16: bool,
    vulkan: bool,
) -> dict:
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
    ]
    if fp16:
        cmd.append("--fp16")
    if vulkan:
        cmd.append("--vulkan")

    raw = adb("shell", f"LD_LIBRARY_PATH={DEVICE_DIR} " + " ".join(cmd))
    line = raw.strip().splitlines()[-1]
    return json.loads(line)


def main() -> None:
    args = parse_args()
    fp16 = args.fp16 and not args.no_fp16
    vulkan = args.vulkan and not args.no_vulkan

    if not NCNN_MANIFEST.exists():
        raise SystemExit(f"Missing {NCNN_MANIFEST} — run export + convert first")
    if not BENCH_BIN.exists():
        raise SystemExit(f"Missing {BENCH_BIN} — run deploy/build_android_bench.sh")
    if not adb_ok():
        raise SystemExit("No adb device — see deploy/DEPLOY.md")

    manifest = json.loads(NCNN_MANIFEST.read_text(encoding="utf-8"))
    models = manifest["models"]
    if args.preset != "all":
        models = [m for m in models if m["preset"] == args.preset]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device_info = collect_device_info()
    (RESULTS_DIR / "device_info.json").write_text(
        json.dumps(device_info, indent=2), encoding="utf-8"
    )
    print("Device:", json.dumps(device_info, indent=2))

    if args.dry_run:
        print(f"Would benchmark {len(models)} model(s)")
        return

    adb("shell", f"mkdir -p {DEVICE_DIR}/models", capture=False)
    if not args.skip_push:
        adb("push", str(BENCH_BIN), f"{DEVICE_DIR}/sr_bench", capture=False)
        adb("shell", f"chmod +x {DEVICE_DIR}/sr_bench", capture=False)
    if not LIBOMP.exists():
        raise SystemExit(f"Missing {LIBOMP} — required at runtime on device")
    adb("push", str(LIBOMP), f"{DEVICE_DIR}/libomp.so", capture=False)

    rows: list[dict] = []
    for meta in models:
        param_local = PROJECT_ROOT / meta["ncnn_param"]
        bin_local = PROJECT_ROOT / meta["ncnn_bin"]
        base = param_local.stem
        remote_param = f"{DEVICE_DIR}/models/{base}.param"
        remote_bin = f"{DEVICE_DIR}/models/{base}.bin"

        print(f"\n=== {meta['model_id']} @ {meta['preset']} ({meta['lr_w']}x{meta['lr_h']}) ===")
        adb("push", str(param_local), remote_param, capture=False)
        adb("push", str(bin_local), remote_bin, capture=False)

        in_blob = meta.get("in_blob") or parse_blobs(param_local)[0]
        out_blob = meta.get("out_blob") or parse_blobs(param_local)[1]
        print(f"  blobs: {in_blob} -> {out_blob}")

        bench = run_bench_remote(
            remote_param, remote_bin, in_blob, out_blob,
            int(meta["lr_w"]), int(meta["lr_h"]),
            args.warmup, args.iters, fp16, vulkan,
        )
        print(
            f"  median={bench['median_ms']:.2f} ms  p90={bench['p90_ms']:.2f} ms  "
            f"fps={bench['fps']:.1f}  peak_mem={bench.get('peak_memory_kb', -1)} kB"
        )
        rows.append({**meta, **bench})

    out = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "device": device_info,
        "protocol": {
            "warmup": args.warmup,
            "iters": args.iters,
            "fp16": fp16,
            "vulkan": vulkan,
            "backend": "ncnn_vulkan" if vulkan else "ncnn_cpu",
        },
        "results": rows,
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"mobile_benchmark_{ts}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    latest = RESULTS_DIR / "mobile_benchmark_latest.json"
    latest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path.relative_to(PROJECT_ROOT)}")
    print(f"Also: {latest.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
