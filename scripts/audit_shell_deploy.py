#!/usr/bin/env python3
"""Factorial shell deploy audit: residual vs endpoints at matched Conv-MACs.

Builds four fused deploy graphs from PECSR / ECBSR 20k checkpoints, exports via
PNNX → NCNN, counts operators, and benches host + phone (NCNN FP16).

Variants (single-factor relative to PECSR shell):
  pecsr              plain endpoints, no global residual          (PECSR)
  pecsr_plus_skip    plain endpoints + global LR residual         (P1)
  ecb_ends_no_skip   ECB endpoints, no global residual            (P2-ish)
  ecbsr              ECB endpoints + global LR residual           (ECBSR)

Usage (cv_env + adb on PATH):

  export PATH=$HOME/miniforge3/bin:$HOME/android/platform-tools:$PATH
  conda activate cv_env
  python scripts/audit_shell_deploy.py
  python scripts/audit_shell_deploy.py --sessions 3   # phone envelope
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

_extra = [
    str(Path.home() / "miniforge3/bin"),
    str(Path.home() / "android/platform-tools"),
]
os.environ["PATH"] = os.pathsep.join(_extra + [os.environ.get("PATH", "")])

from models.ecbsr import ECBSR, FusedECBSR, fuse_ecbsr  # noqa: E402
from models.sepres_v2 import (  # noqa: E402
    FusedSepResV2,
    SepResV2,
    conv_macs_at_lr,
    fuse_sepres_v2,
)
from utils.model_loader import load_checkpoint_model  # noqa: E402

MODELS_JSON = PROJECT_ROOT / "deploy/models.json"
TS_DIR = PROJECT_ROOT / "deploy/artifacts/torchscript"
NCNN_DIR = PROJECT_ROOT / "deploy/artifacts/ncnn"
RESULTS_DIR = PROJECT_ROOT / "deploy/artifacts/results"
REPORT_METRICS = PROJECT_ROOT.parent / "report/assets/metrics"
PARSE_BLOBS = PROJECT_ROOT / "scripts/parse_ncnn_blobs.py"
BENCH_BIN = PROJECT_ROOT / "deploy/android/sr_bench/build/sr_bench"
HOST_BENCH = PROJECT_ROOT / "deploy/android/sr_bench/build-host/sr_bench_host"
DEVICE_DIR = "/data/local/tmp/sr_bench"
ADB = Path.home() / "android/platform-tools/adb"
if not ADB.exists():
    ADB = Path("adb")
LIBOMP = (
    Path.home()
    / "android/ndk/android-ndk-r26d/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/17/lib/linux/aarch64/libomp.so"
)

PECSR_CKPT = PROJECT_ROOT / "results/exp_runs/sepres_v2_c16n10_20k/checkpoints/best.pt"
ECBSR_CKPT = PROJECT_ROOT / "results/exp_runs/ecbsr_m10c16_20k/checkpoints/best.pt"


class FusedWithGlobalSkip(nn.Module):
    """Plain-endpoint fused stack + ECBSR-style channel-repeat residual."""

    def __init__(self, base: FusedSepResV2) -> None:
        super().__init__()
        self.scale_factor = base.scale_factor
        self.num_channels = base.num_channels
        self.head = base.head
        self.body = base.body
        self.tail = base.tail
        self.upsampler = base.upsampler

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        r2 = self.scale_factor * self.scale_factor
        shortcut = x.unsqueeze(2).expand(b, c, r2, h, w).reshape(b, c * r2, h, w)
        y = self.head(x)
        y = self.body(y)
        y = self.tail(y)
        return self.upsampler(y + shortcut)


class FusedECBEndsNoSkip(nn.Module):
    """Fused ECBSR backbone without the global residual (endpoint-only delta)."""

    def __init__(self, fused: FusedECBSR) -> None:
        super().__init__()
        self.scale_factor = fused.scale_factor
        self.num_channels = fused.num_channels
        self.backbone = fused.backbone
        self.upsampler = fused.upsampler

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsampler(self.backbone(x))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shell residual/endpoint deploy audit")
    p.add_argument("--preset", default="audit_180", choices=["audit_180", "deploy_720p"])
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--sessions", type=int, default=3, help="Paired phone sessions (0=skip phone)")
    p.add_argument("--host-iters", type=int, default=100)
    p.add_argument("--host-warmup", type=int, default=30)
    p.add_argument("--skip-export", action="store_true", help="Reuse existing NCNN stems")
    p.add_argument("--skip-host", action="store_true")
    p.add_argument("--skip-cuda", action="store_true")
    p.add_argument("--cuda-iters", type=int, default=500)
    p.add_argument("--cuda-warmup", type=int, default=100)
    return p.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def adb(*args: str, capture: bool = True) -> str:
    cmd = [str(ADB), *args]
    if capture:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    subprocess.check_call(cmd)
    return ""


def adb_ok() -> bool:
    try:
        return adb("get-state") == "device"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def parse_blobs(param_path: Path) -> tuple[str, str]:
    out = subprocess.check_output(
        [sys.executable, str(PARSE_BLOBS), str(param_path)], text=True
    ).strip()
    return out.split("\t")


def export_torchscript(model: nn.Module, stem: str, lr_h: int, lr_w: int) -> Path:
    TS_DIR.mkdir(parents=True, exist_ok=True)
    out = TS_DIR / f"{stem}.pt"
    dummy = torch.randn(1, 3, lr_h, lr_w)
    model.eval()
    with torch.no_grad():
        torch.jit.trace(model, dummy).save(str(out))
    return out


def convert_pnnx(ts_path: Path, inputshape: str) -> tuple[Path, Path]:
    pnnx = shutil.which("pnnx")
    if not pnnx:
        raise FileNotFoundError("pnnx not on PATH")
    subprocess.check_call(
        [pnnx, str(ts_path), f"inputshape={inputshape}", "device=cpu", "fp16=0"],
        cwd=PROJECT_ROOT,
    )
    param = ts_path.with_suffix(".ncnn.param")
    binf = ts_path.with_suffix(".ncnn.bin")
    if not param.exists() or not binf.exists():
        raise RuntimeError(f"PNNX output missing for {ts_path}")
    return param, binf


def audit_param(param_path: Path, lr_h: int, lr_w: int) -> dict:
    lines = [ln for ln in param_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    types = [ln.split()[0] for ln in lines[2:]]
    counts = dict(Counter(types))
    residual_ops = ["Split", "ExpandDims", "Tile", "Reshape", "BinaryOp"]
    residual_present = {op: int(counts.get(op, 0)) for op in residual_ops}
    # FP16 traffic estimate for residual expand write + BinaryOp add (2R1W)
    shortcut_elems = 3 * 16 * lr_h * lr_w  # 3 * 4^2
    residual_traffic_fp16_mb = (shortcut_elems + 3 * shortcut_elems) * 2 / 1e6
    return {
        "layer_rows": len(types),
        "type_counts": counts,
        "n_convolution": int(counts.get("Convolution", 0)),
        "n_prelu": int(counts.get("PReLU", 0)),
        "n_pixelshuffle": int(counts.get("PixelShuffle", 0)),
        "residual_ops": residual_present,
        "residual_op_total": int(sum(residual_present.values())),
        "has_global_residual_graph": bool(counts.get("BinaryOp", 0) > 0),
        "residual_shortcut_elems": shortcut_elems,
        "residual_traffic_fp16_mb_approx": round(residual_traffic_fp16_mb, 4)
        if counts.get("BinaryOp", 0)
        else 0.0,
    }


def build_variants() -> dict[str, nn.Module]:
    pecsr_raw, _ = load_checkpoint_model(PECSR_CKPT, torch.device("cpu"))
    ecbsr_raw, _ = load_checkpoint_model(ECBSR_CKPT, torch.device("cpu"))
    if not isinstance(pecsr_raw, SepResV2):
        raise TypeError(type(pecsr_raw))
    if not isinstance(ecbsr_raw, ECBSR):
        raise TypeError(type(ecbsr_raw))

    pecsr = fuse_sepres_v2(pecsr_raw)
    ecbsr = fuse_ecbsr(ecbsr_raw)
    return {
        "pecsr": pecsr,
        "pecsr_plus_skip": FusedWithGlobalSkip(pecsr),
        "ecb_ends_no_skip": FusedECBEndsNoSkip(ecbsr),
        "ecbsr": ecbsr,
    }


VARIANT_META = {
    "pecsr": {
        "label": "PECSR",
        "endpoints": "plain",
        "global_residual": False,
        "factor": "baseline_shell",
    },
    "pecsr_plus_skip": {
        "label": "PECSR+skip",
        "endpoints": "plain",
        "global_residual": True,
        "factor": "P1_residual_only",
    },
    "ecb_ends_no_skip": {
        "label": "ECB-ends no-skip",
        "endpoints": "ecb",
        "global_residual": False,
        "factor": "P2_endpoints_only",
    },
    "ecbsr": {
        "label": "ECBSR",
        "endpoints": "ecb",
        "global_residual": True,
        "factor": "full_ecbsr_shell",
    },
}


def export_variant(
    vid: str, model: nn.Module, stem: str, lr_h: int, lr_w: int
) -> dict:
    print(f"  export {vid} → {stem}")
    ts = export_torchscript(model, stem, lr_h, lr_w)
    param_src, bin_src = convert_pnnx(ts, f"[1,3,{lr_h},{lr_w}]")
    NCNN_DIR.mkdir(parents=True, exist_ok=True)
    param = NCNN_DIR / f"{stem}.param"
    binf = NCNN_DIR / f"{stem}.bin"
    shutil.copy2(param_src, param)
    shutil.copy2(bin_src, binf)
    in_blob, out_blob = parse_blobs(param)
    graph = audit_param(param, lr_h, lr_w)
    macs = conv_macs_at_lr(model, lr_h, lr_w)
    params = sum(p.numel() for p in model.parameters())
    return {
        "id": vid,
        **VARIANT_META[vid],
        "stem": stem,
        "torchscript": rel(ts),
        "ncnn_param": rel(param),
        "ncnn_bin": rel(binf),
        "ncnn_bytes": param.stat().st_size + binf.stat().st_size,
        "in_blob": in_blob,
        "out_blob": out_blob,
        "params": params,
        "conv_macs": macs,
        "conv_macs_g": round(macs / 1e9, 4),
        "graph": graph,
    }


def run_host_bench(
    entry: dict, lr_w: int, lr_h: int, warmup: int, iters: int
) -> dict:
    if not HOST_BENCH.exists():
        raise FileNotFoundError(HOST_BENCH)
    param = PROJECT_ROOT / entry["ncnn_param"]
    binf = PROJECT_ROOT / entry["ncnn_bin"]
    cmd = [
        str(HOST_BENCH),
        "--param",
        str(param),
        "--bin",
        str(binf),
        "--input-w",
        str(lr_w),
        "--input-h",
        str(lr_h),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--fp16",
    ]
    raw = subprocess.check_output(cmd, text=True).strip().splitlines()[-1]
    return json.loads(raw)


def ensure_phone_bench(skip_push: bool = False) -> None:
    if not BENCH_BIN.exists():
        raise SystemExit(f"Missing {BENCH_BIN}")
    adb("shell", f"mkdir -p {DEVICE_DIR}/models", capture=False)
    if not skip_push:
        adb("push", str(BENCH_BIN), f"{DEVICE_DIR}/sr_bench", capture=False)
        adb("shell", f"chmod +x {DEVICE_DIR}/sr_bench", capture=False)
        if LIBOMP.exists():
            adb("push", str(LIBOMP), f"{DEVICE_DIR}/libomp.so", capture=False)


def run_phone_bench(entry: dict, lr_w: int, lr_h: int, warmup: int, iters: int) -> dict:
    param = PROJECT_ROOT / entry["ncnn_param"]
    binf = PROJECT_ROOT / entry["ncnn_bin"]
    remote_p = f"{DEVICE_DIR}/models/{param.name}"
    remote_b = f"{DEVICE_DIR}/models/{binf.name}"
    adb("push", str(param), remote_p, capture=False)
    adb("push", str(binf), remote_b, capture=False)
    cmd = [
        f"{DEVICE_DIR}/sr_bench",
        "--param",
        remote_p,
        "--bin",
        remote_b,
        "--in-blob",
        entry["in_blob"],
        "--out-blob",
        entry["out_blob"],
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


def paired_phone_sessions(
    entries: list[dict],
    *,
    lr_w: int,
    lr_h: int,
    warmup: int,
    iters: int,
    sessions: int,
) -> dict:
    """Alternate order each session to reduce thermal bias."""
    ensure_phone_bench(skip_push=False)
    device = {
        "model": adb("shell", "getprop ro.product.model"),
        "device": adb("shell", "getprop ro.product.device"),
        "platform": adb("shell", "getprop ro.board.platform"),
    }
    per_id: dict[str, list[dict]] = {e["id"]: [] for e in entries}
    order_ids = [e["id"] for e in entries]
    id_to_entry = {e["id"]: e for e in entries}

    for s in range(sessions):
        order = order_ids if s % 2 == 0 else list(reversed(order_ids))
        print(f"  phone session {s + 1}/{sessions} order={order}")
        for vid in order:
            result = run_phone_bench(id_to_entry[vid], lr_w, lr_h, warmup, iters)
            per_id[vid].append(result)
            print(
                f"    {vid}: med={result['median_ms']:.2f} p90={result['p90_ms']:.2f}"
            )
        time.sleep(2.0)

    summary = {}
    for vid, rows in per_id.items():
        meds = [r["median_ms"] for r in rows]
        p90s = [r["p90_ms"] for r in rows]
        summary[vid] = {
            "sessions": len(rows),
            "median_ms_mean": statistics.mean(meds),
            "median_ms_std": statistics.pstdev(meds) if len(meds) > 1 else 0.0,
            "p90_ms_mean": statistics.mean(p90s),
            "raw": rows,
        }
    return {"device": device, "backend": "ncnn_vulkan_fp16", "variants": summary}


def cuda_latency(model: nn.Module, lr_h: int, lr_w: int, warmup: int, iters: int) -> dict:
    if not torch.cuda.is_available():
        return {"skipped": True, "reason": "no_cuda"}
    device = torch.device("cuda")
    model = model.to(device).eval()
    x = torch.randn(1, 3, lr_h, lr_w, device=device)
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        samples = []
        for _ in range(iters):
            starter.record()
            model(x)
            ender.record()
            torch.cuda.synchronize()
            samples.append(starter.elapsed_time(ender))
    samples.sort()
    return {
        "skipped": False,
        "device": torch.cuda.get_device_name(0),
        "precision": "fp32",
        "median_ms": statistics.median(samples),
        "p90_ms": samples[max(0, int(0.9 * (len(samples) - 1)))],
        "iters": iters,
    }


def deltas_vs_pecsr(payload: dict) -> dict:
    """Attribute phone / host / graph deltas to residual vs endpoints."""
    vars_ = {v["id"]: v for v in payload["variants"]}
    phone = payload.get("phone", {}).get("variants", {})
    host = {h["id"]: h for h in payload.get("host", [])}

    def phone_med(vid: str) -> float | None:
        if vid not in phone:
            return None
        return float(phone[vid]["median_ms_mean"])

    def host_med(vid: str) -> float | None:
        if vid not in host:
            return None
        return float(host[vid]["median_ms"])

    pecsr_p = phone_med("pecsr")
    out: dict = {"phone_ms": {}, "host_ms": {}, "graph": {}}

    if pecsr_p is not None:
        for vid in ("pecsr_plus_skip", "ecb_ends_no_skip", "ecbsr"):
            m = phone_med(vid)
            if m is None:
                continue
            out["phone_ms"][vid] = {
                "median_ms": m,
                "delta_vs_pecsr_ms": m - pecsr_p,
                "delta_vs_pecsr_pct": 100.0 * (m - pecsr_p) / pecsr_p,
            }
        # Factor isolation (additive approx)
        if phone_med("pecsr_plus_skip") is not None:
            out["phone_ms"]["residual_cost_ms"] = phone_med("pecsr_plus_skip") - pecsr_p
        if phone_med("ecb_ends_no_skip") is not None:
            out["phone_ms"]["endpoint_cost_ms"] = phone_med("ecb_ends_no_skip") - pecsr_p
        if phone_med("ecbsr") is not None and phone_med("pecsr_plus_skip") is not None:
            # residual on ECB-ends path
            out["phone_ms"]["residual_on_ecb_ends_ms"] = (
                phone_med("ecbsr") - phone_med("ecb_ends_no_skip")
            )

    pecsr_h = host_med("pecsr")
    if pecsr_h is not None:
        for vid in ("pecsr_plus_skip", "ecb_ends_no_skip", "ecbsr"):
            m = host_med(vid)
            if m is None:
                continue
            out["host_ms"][vid] = {
                "median_ms": m,
                "delta_vs_pecsr_ms": m - pecsr_h,
                "delta_vs_pecsr_pct": 100.0 * (m - pecsr_h) / pecsr_h,
            }

    for vid, v in vars_.items():
        g = v["graph"]
        out["graph"][vid] = {
            "layer_rows": g["layer_rows"],
            "n_convolution": g["n_convolution"],
            "n_prelu": g["n_prelu"],
            "residual_op_total": g["residual_op_total"],
            "has_global_residual_graph": g["has_global_residual_graph"],
        }
    return out


def main() -> None:
    args = parse_args()
    registry = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    presets = {p["name"]: p for p in registry["input_presets"]}
    preset = presets[args.preset]
    lr_w, lr_h = int(preset["lr_w"]), int(preset["lr_h"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print("[1/5] build fused shell variants")
    models = build_variants()
    for vid, m in models.items():
        print(
            f"  {vid}: params={sum(p.numel() for p in m.parameters())} "
            f"macs_g={conv_macs_at_lr(m, lr_h, lr_w)/1e9:.3f}"
        )

    print("[2/5] export / load NCNN")
    variants: list[dict] = []
    for vid, model in models.items():
        stem = f"shell_{vid}_{args.preset}"
        if args.skip_export:
            param = NCNN_DIR / f"{stem}.param"
            binf = NCNN_DIR / f"{stem}.bin"
            if not param.exists():
                raise SystemExit(f"missing {param}; drop --skip-export")
            in_blob, out_blob = parse_blobs(param)
            graph = audit_param(param, lr_h, lr_w)
            entry = {
                "id": vid,
                **VARIANT_META[vid],
                "stem": stem,
                "ncnn_param": rel(param),
                "ncnn_bin": rel(binf),
                "ncnn_bytes": param.stat().st_size + binf.stat().st_size,
                "in_blob": in_blob,
                "out_blob": out_blob,
                "params": sum(p.numel() for p in model.parameters()),
                "conv_macs": conv_macs_at_lr(model, lr_h, lr_w),
                "conv_macs_g": round(conv_macs_at_lr(model, lr_h, lr_w) / 1e9, 4),
                "graph": graph,
            }
        else:
            entry = export_variant(vid, model, stem, lr_h, lr_w)
        variants.append(entry)
        g = entry["graph"]
        print(
            f"  {vid}: layers={g['layer_rows']} conv={g['n_convolution']} "
            f"prelu={g['n_prelu']} residual_ops={g['residual_op_total']}"
        )

    payload: dict = {
        "task": "shell_deploy_factorial_audit",
        "created_utc": stamp,
        "preset": args.preset,
        "lr": [lr_w, lr_h],
        "protocol": {
            "phone_warmup": args.warmup,
            "phone_iters": args.iters,
            "phone_sessions": args.sessions,
            "host_warmup": args.host_warmup,
            "host_iters": args.host_iters,
            "backend_phone": "ncnn_vulkan_fp16",
            "backend_host": "ncnn_fp16_cpu",
            "note": (
                "Latency uses fused deploy graphs; weights from PECSR/ECBSR 20k. "
                "pecsr_plus_skip reuses PECSR weights + residual wiring (latency-only). "
                "ecb_ends_no_skip reuses ECBSR fused backbone without skip."
            ),
        },
        "checkpoints": {
            "pecsr": rel(PECSR_CKPT),
            "ecbsr": rel(ECBSR_CKPT),
        },
        "variants": variants,
    }

    print("[3/5] host NCNN FP16 bench")
    if args.skip_host:
        payload["host"] = []
    else:
        host_rows = []
        for entry in variants:
            row = run_host_bench(
                entry, lr_w, lr_h, args.host_warmup, args.host_iters
            )
            host_rows.append(
                {
                    "id": entry["id"],
                    "median_ms": row["median_ms"],
                    "p90_ms": row["p90_ms"],
                    "peak_memory_kb": row.get("peak_memory_kb"),
                    "raw": row,
                }
            )
            print(
                f"  {entry['id']}: med={row['median_ms']:.2f} "
                f"p90={row['p90_ms']:.2f} rss={row.get('peak_memory_kb')}"
            )
        payload["host"] = host_rows

    print("[4/5] CUDA PyTorch latency (optional)")
    if args.skip_cuda:
        payload["cuda"] = {}
    else:
        cuda_rows = {}
        # rebuild fresh models on CPU then time on CUDA
        models_cuda = build_variants()
        for vid, model in models_cuda.items():
            cuda_rows[vid] = cuda_latency(
                model, lr_h, lr_w, args.cuda_warmup, args.cuda_iters
            )
            if not cuda_rows[vid].get("skipped"):
                print(
                    f"  {vid}: med={cuda_rows[vid]['median_ms']:.3f} ms "
                    f"({cuda_rows[vid]['device']})"
                )
        payload["cuda"] = cuda_rows

    print("[5/5] phone NCNN Vulkan FP16")
    if args.sessions <= 0:
        payload["phone"] = {"skipped": True, "reason": "sessions=0"}
    elif not adb_ok():
        payload["phone"] = {"skipped": True, "reason": "adb_not_device"}
        print("  SKIP: adb not in device state")
    else:
        payload["phone"] = paired_phone_sessions(
            variants,
            lr_w=lr_w,
            lr_h=lr_h,
            warmup=args.warmup,
            iters=args.iters,
            sessions=args.sessions,
        )

    payload["attribution"] = deltas_vs_pecsr(payload)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_METRICS.mkdir(parents=True, exist_ok=True)
    out_stamp = RESULTS_DIR / f"shell_deploy_audit_{stamp}.json"
    out_latest = RESULTS_DIR / "shell_deploy_audit_latest.json"
    report_out = REPORT_METRICS / "shell_deploy_audit.json"
    text = json.dumps(payload, indent=2)
    out_stamp.write_text(text, encoding="utf-8")
    out_latest.write_text(text, encoding="utf-8")
    report_out.write_text(text, encoding="utf-8")
    print(f"\nWrote {rel(out_latest)}")
    print(f"Wrote {report_out}")

    attr = payload["attribution"]
    print("\n=== Attribution (vs PECSR) ===")
    print(json.dumps(attr.get("phone_ms", {}), indent=2))
    print(json.dumps(attr.get("graph", {}), indent=2))


if __name__ == "__main__":
    main()
