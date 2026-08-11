#!/usr/bin/env python3
"""On-device PECSR demo via adb: push LR → NCNN infer on phone → pull HR + side-by-side.

Same adb / DEVICE_DIR / libomp conventions as scripts/bench_mobile.py.
Spec: progress/track_d.md (Demo1).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARSE_BLOBS = PROJECT_ROOT / "scripts/parse_ncnn_blobs.py"
BENCH_BIN = PROJECT_ROOT / "deploy/android/sr_bench/build/sr_bench"
DEVICE_DIR = "/data/local/tmp/sr_bench"
DEMO_DIR = PROJECT_ROOT / "deploy/demo"
ADB = Path.home() / "android/platform-tools/adb"
if not ADB.exists():
    ADB = Path("adb")
LIBOMP = (
    Path.home()
    / "android/ndk/android-ndk-r26d/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/17/lib/linux/aarch64/libomp.so"
)

# Friendly CLI name → models.json id / display label
MODEL_ALIASES = {
    "pecsr": ("sepres_v2_c16n10", "PECSR"),
    "ecbsr": ("ecbsr_m10c16", "ECBSR"),
    "fsrcnn": ("fsrcnn", "FSRCNN"),
}

# Prefer fused phone graphs used in Track B benches (not the incomplete ncnn_manifest alone).
NCNN_FILES = {
    ("sepres_v2_c16n10", "audit_180"): (
        "deploy/artifacts/ncnn/sepres_v2_c16n10_fused_audit_180_smoke.param",
        "deploy/artifacts/ncnn/sepres_v2_c16n10_fused_audit_180_smoke.bin",
    ),
    ("sepres_v2_c16n10", "deploy_720p"): (
        "deploy/artifacts/ncnn/sepres_v2_c16n10_fused_deploy_720p.param",
        "deploy/artifacts/ncnn/sepres_v2_c16n10_fused_deploy_720p.bin",
    ),
    ("ecbsr_m10c16", "audit_180"): (
        "deploy/artifacts/ncnn/ecbsr_m10c16_fused_audit_180_smoke.param",
        "deploy/artifacts/ncnn/ecbsr_m10c16_fused_audit_180_smoke.bin",
    ),
    ("ecbsr_m10c16", "deploy_720p"): (
        "deploy/artifacts/ncnn/ecbsr_m10c16_fused_deploy_720p.param",
        "deploy/artifacts/ncnn/ecbsr_m10c16_fused_deploy_720p.bin",
    ),
    ("fsrcnn", "audit_180"): (
        "deploy/artifacts/ncnn/fsrcnn_audit_180.param",
        "deploy/artifacts/ncnn/fsrcnn_audit_180.bin",
    ),
    ("fsrcnn", "deploy_720p"): (
        "deploy/artifacts/ncnn/fsrcnn_deploy_720p.param",
        "deploy/artifacts/ncnn/fsrcnn_deploy_720p.bin",
    ),
}

PRESET_SIZE = {
    "audit_180": (180, 180),
    "deploy_720p": (320, 180),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="On-device mobile SR demo (adb + NCNN)")
    p.add_argument("--model", choices=list(MODEL_ALIASES), default="pecsr")
    p.add_argument("--lr", type=Path, help="LR PNG path (required unless --offline)")
    p.add_argument("--out-dir", type=Path, default=DEMO_DIR / "out")
    p.add_argument("--preset", choices=list(PRESET_SIZE), default="audit_180")
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--vulkan", action="store_true", default=True)
    p.add_argument("--no-vulkan", action="store_true")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--skip-push-bin", action="store_true", help="Skip pushing sr_bench/libomp")
    p.add_argument(
        "--offline",
        action="store_true",
        help="Show pre-recorded failsafe stills (no adb). Marks chrome as pre-recorded.",
    )
    p.add_argument(
        "--failsafe-stem",
        default="crop01",
        help="Stem under deploy/demo/failsafe/ for --offline (expects STEM_lr.png + STEM_hr.png + STEM_meta.json)",
    )
    p.add_argument("--show", action="store_true", help="Open side-by-side with default viewer")
    p.add_argument("--save-failsafe", action="store_true", help="Also copy this run into deploy/demo/failsafe/")
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


def device_model() -> str:
    try:
        model = adb("shell", "getprop", "ro.product.model")
        return model.strip() or "Android"
    except subprocess.CalledProcessError:
        return "Android"


def chrome_line(
    label: str,
    device: str,
    backend: str,
    lr_w: int,
    lr_h: int,
    ms: float | None,
    *,
    live: bool,
) -> str:
    ms_s = f"{ms:.1f} ms" if ms is not None else "n/a"
    live_tag = "on device" if live else "pre-recorded"
    return f"{label} · {device} · {live_tag} · {backend} · LR {lr_w}×{lr_h} · {ms_s}"


def make_side_by_side(
    lr: Image.Image,
    hr: Image.Image,
    title: str,
    out_path: Path,
) -> None:
    # Upscale LR nearest for fair visual panel next to HR.
    lr_up = lr.resize(hr.size, Image.Resampling.NEAREST)
    gap = 16
    bar_h = 48
    w = lr_up.width + gap + hr.width
    h = max(lr_up.height, hr.height) + bar_h
    canvas = Image.new("RGB", (w, h), (18, 18, 18))
    canvas.paste(lr_up, (0, bar_h))
    canvas.paste(hr, (lr_up.width + gap, bar_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font
    draw.text((12, 12), title, fill=(240, 240, 240), font=font)
    draw.text((12, bar_h + 8), "LR (nearest↑)", fill=(200, 200, 200), font=font_sm)
    draw.text((lr_up.width + gap + 12, bar_h + 8), "SR (on device)" if "pre-recorded" not in title else "SR (pre-recorded)", fill=(200, 200, 200), font=font_sm)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def prepare_lr(src: Path, out_path: Path, lr_w: int, lr_h: int) -> Path:
    im = Image.open(src).convert("RGB")
    if im.size != (lr_w, lr_h):
        # Center-crop then resize to preset (demo crops should already match).
        scale = max(lr_w / im.width, lr_h / im.height)
        nw, nh = int(im.width * scale + 0.5), int(im.height * scale + 0.5)
        im = im.resize((nw, nh), Image.Resampling.BICUBIC)
        left = max(0, (nw - lr_w) // 2)
        top = max(0, (nh - lr_h) // 2)
        im = im.crop((left, top, left + lr_w, top + lr_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)
    return out_path


def run_offline(args: argparse.Namespace) -> None:
    stem = args.failsafe_stem
    fs = DEMO_DIR / "failsafe"
    lr_p = fs / f"{stem}_lr.png"
    hr_p = fs / f"{stem}_hr.png"
    meta_p = fs / f"{stem}_meta.json"
    if not lr_p.exists() or not hr_p.exists():
        raise SystemExit(f"Missing failsafe pair under {fs} ({stem}_lr/_hr.png)")
    meta = {}
    if meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
    label = meta.get("label", MODEL_ALIASES[args.model][1])
    device = meta.get("device", "n/a")
    backend = meta.get("backend", "NCNN (recorded)")
    lr_w, lr_h = meta.get("lr_w", PRESET_SIZE[args.preset][0]), meta.get("lr_h", PRESET_SIZE[args.preset][1])
    ms = meta.get("latency_ms")
    title = chrome_line(label, device, backend, lr_w, lr_h, ms, live=False)
    print(title)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    side = out_dir / f"{stem}_side_by_side_offline.png"
    make_side_by_side(Image.open(lr_p), Image.open(hr_p), title, side)
    print(f"Wrote {side.relative_to(PROJECT_ROOT)}")
    if args.show:
        Image.open(side).show()


def main() -> None:
    args = parse_args()
    fp16 = args.fp16 and not args.no_fp16
    vulkan = args.vulkan and not args.no_vulkan
    backend = f"NCNN {'Vulkan' if vulkan else 'CPU'}{' FP16' if fp16 else ' FP32'}"

    if args.offline:
        run_offline(args)
        return

    if args.lr is None:
        raise SystemExit("--lr is required (or pass --offline)")

    model_id, label = MODEL_ALIASES[args.model]
    key = (model_id, args.preset)
    if key not in NCNN_FILES:
        raise SystemExit(f"No NCNN mapping for {key}")
    param_rel, bin_rel = NCNN_FILES[key]
    param_local = PROJECT_ROOT / param_rel
    bin_local = PROJECT_ROOT / bin_rel
    if not param_local.exists() or not bin_local.exists():
        raise SystemExit(f"Missing NCNN files:\n  {param_local}\n  {bin_local}")
    if not BENCH_BIN.exists():
        raise SystemExit(f"Missing {BENCH_BIN} — run deploy/build_android_bench.sh")
    if not adb_ok():
        raise SystemExit("No adb device — see deploy/DEPLOY.md (or use --offline)")

    lr_w, lr_h = PRESET_SIZE[args.preset]
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.lr.stem
    lr_ready = prepare_lr(args.lr, out_dir / f"{stem}_lr_{lr_w}x{lr_h}.png", lr_w, lr_h)

    in_blob, out_blob = parse_blobs(param_local)
    device = device_model()

    adb("shell", f"mkdir -p {DEVICE_DIR}/models {DEVICE_DIR}/demo", capture=False)
    if not args.skip_push_bin:
        adb("push", str(BENCH_BIN), f"{DEVICE_DIR}/sr_bench", capture=False)
        adb("shell", f"chmod +x {DEVICE_DIR}/sr_bench", capture=False)
        if not LIBOMP.exists():
            raise SystemExit(f"Missing {LIBOMP}")
        adb("push", str(LIBOMP), f"{DEVICE_DIR}/libomp.so", capture=False)

    remote_param = f"{DEVICE_DIR}/models/{param_local.name}"
    remote_bin = f"{DEVICE_DIR}/models/{bin_local.name}"
    remote_in = f"{DEVICE_DIR}/demo/in.png"
    remote_out = f"{DEVICE_DIR}/demo/out.png"
    adb("push", str(param_local), remote_param, capture=False)
    adb("push", str(bin_local), remote_bin, capture=False)
    adb("push", str(lr_ready), remote_in, capture=False)

    cmd = [
        f"{DEVICE_DIR}/sr_bench",
        "--param", remote_param,
        "--bin", remote_bin,
        "--in-blob", in_blob,
        "--out-blob", out_blob,
        "--in", remote_in,
        "--out", remote_out,
        "--input-w", str(lr_w),
        "--input-h", str(lr_h),
        "--warmup", str(args.warmup),
    ]
    if fp16:
        cmd.append("--fp16")
    if vulkan:
        cmd.append("--vulkan")

    raw = adb("shell", f"LD_LIBRARY_PATH={DEVICE_DIR} " + " ".join(cmd))
    # JSON is last line; latency also on stderr (may be merged into stdout by adb).
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    json_line = next((ln for ln in reversed(lines) if ln.strip().startswith("{")), None)
    if not json_line:
        raise SystemExit(f"No JSON from device:\n{raw}")
    result = json.loads(json_line)
    ms = float(result["latency_ms"])

    hr_local = out_dir / f"{stem}_hr_{label.lower()}.png"
    adb("pull", remote_out, str(hr_local), capture=False)

    title = chrome_line(label, device, backend, lr_w, lr_h, ms, live=True)
    print(title)
    side = out_dir / f"{stem}_side_by_side.png"
    make_side_by_side(Image.open(lr_ready), Image.open(hr_local), title, side)
    print(f"Wrote {side.relative_to(PROJECT_ROOT)}")
    print(f"HR: {hr_local.relative_to(PROJECT_ROOT)}")

    meta = {
        "label": label,
        "model_id": model_id,
        "device": device,
        "backend": backend,
        "preset": args.preset,
        "lr_w": lr_w,
        "lr_h": lr_h,
        "latency_ms": ms,
        "live": True,
        "ncnn_param": param_rel,
        "result": result,
    }
    (out_dir / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.save_failsafe:
        fs = DEMO_DIR / "failsafe"
        fs.mkdir(parents=True, exist_ok=True)
        fs_stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", stem)[:40] or "crop"
        Image.open(lr_ready).save(fs / f"{fs_stem}_lr.png")
        Image.open(hr_local).save(fs / f"{fs_stem}_hr.png")
        meta_off = {**meta, "live": False, "note": "pre-recorded from on-device run"}
        (fs / f"{fs_stem}_meta.json").write_text(json.dumps(meta_off, indent=2), encoding="utf-8")
        print(f"Failsafe saved under {fs.relative_to(PROJECT_ROOT)} stem={fs_stem}")

    if args.show:
        Image.open(side).show()


if __name__ == "__main__":
    main()
