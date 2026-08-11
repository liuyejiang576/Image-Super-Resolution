# Mobile deployment guide

On-device benchmarking for FSRCNN, MobileSRNet-Base/Plus, **ECBSR (M10C16, fused)**, and **PECSR / SepResV2 C16N10 (freeze_ref, fused)** using **NCNN + Vulkan FP16** on Android. Checkpoints: headline **20k** `best.pt` (`deploy/models.json`; v2/ECBSR set `deploy_fuse: true`). Report display name for the baseline is **ECBSR** (geometry defined once as M10C16).

**Class demo:** on-device PNG→PNG via this NCNN/adb spine — Spec in [`../../progress/track_d.md`](../../progress/track_d.md); runbook `deploy/demo/README.md`.

---

## What you need

| Where | What |
|---|---|
| **PC (WSL)** | Python env, `adb`, Android NDK, ~2 GB disk for NCNN build |
| **Phone** | Android 7+, USB cable, **Developer options → USB debugging** ON |
| **Time** | ~30 min first-time setup; ~5 min per benchmark run after that |

---

## Part 1 — Phone setup (do this first)

### 1. Enable Developer Options

1. **Settings → About phone**
2. Tap **Build number** 7 times → "You are now a developer"

### 2. Enable USB debugging

1. **Settings → System → Developer options** (location varies by OEM)
2. Turn on **USB debugging**
3. Optional but helpful: **Stay awake** (screen on while charging)

### 3. Connect USB and authorize PC

1. Plug phone into PC with a data-capable USB cable
2. On phone: tap **Allow USB debugging** when prompted (check **Always allow**)
3. Set USB mode to **File transfer / MTP** if asked

### 4. Verify from WSL

```bash
sudo apt update && sudo apt install -y adb
adb devices
```

You should see something like:

```
List of devices attached
XXXXXXXX    device
```

If it says `unauthorized`, unplug/replug and accept the prompt on the phone.

**WSL2 USB note:** Windows must forward USB to WSL. Easiest options:

- **Windows 11:** Settings → Bluetooth & devices → USB → attach device to WSL, **or**
- Install [usbipd-win](https://github.com/dorssel/usbipd-win) on Windows, then:

```powershell
# In PowerShell (Admin) on Windows:
usbipd list
usbipd bind --busid <BUSID>
usbipd attach --wsl --busid <BUSID>
```

Then `adb devices` in WSL should show the phone.

### 5. Keep the phone cool and honest

- Unplug other heavy apps; keep screen on
- Run benchmarks with phone **not** in power-save mode
- Note phone model in results (script collects this automatically)
- Scope claims: *"On [your phone] via NCNN Vulkan FP16, …"* — not "all mobile devices"

---

## Part 2 — PC: export models (no phone yet)

From `Image-Super-Resolution/`:

```bash
# Optional: verify deps
pip install onnx onnxruntime onnxscript

# Export all models × both input sizes (180² and 320×180)
python scripts/deploy_smoke.py

# Or step by step:
python scripts/export_onnx.py --model all --preset all
python scripts/verify_onnx_export.py
```

Outputs:

- `deploy/artifacts/onnx/*.onnx`
- `deploy/artifacts/export_manifest.json`
- `deploy/artifacts/onnx_verify_report.json`

---

## Part 3 — PC: build NCNN tools + convert models

### Build host `onnx2ncnn` (one time)

```bash
git clone --depth=1 https://github.com/Tencent/ncnn.git ~/ncnn
cd ~/ncnn && mkdir -p build-host && cd build-host
cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_BUILD_TOOLS=ON ..
cmake --build . -j"$(nproc)"
export PATH="$HOME/ncnn/build-host/tools/onnx:$PATH"
```

### Convert ONNX → NCNN

```bash
cd /home/hyb/CV_project/Image-Super-Resolution
chmod +x deploy/convert_ncnn.sh
./deploy/convert_ncnn.sh
```

Produces `deploy/artifacts/ncnn/*.param` + `*.bin` and `ncnn_manifest.json`.

---

## Part 4 — PC: build Android benchmark binary (one time)

Install **Android NDK** (Android Studio → SDK Manager → NDK, or standalone).

```bash
export ANDROID_NDK=$HOME/Android/Sdk/ndk/26.1.10909125   # adjust version
chmod +x deploy/build_android_bench.sh
./deploy/build_android_bench.sh
```

Binary: `deploy/android/sr_bench/build/sr_bench` (arm64).

---

## Part 5 — Run benchmark on phone

Phone connected, `adb devices` shows `device`:

```bash
# All models, both resolutions
python scripts/bench_mobile.py

# Primary deploy size only (LR 320×180)
python scripts/bench_mobile.py --preset deploy_720p

# CPU fallback if Vulkan fails on your device
python scripts/bench_mobile.py --no-vulkan
```

Results:

- `deploy/artifacts/results/mobile_benchmark_latest.json`
- `deploy/artifacts/results/device_info.json`

Each row includes: median/p90 latency (ms), FPS, peak memory (kB), model size, input size, backend flags.

### Protocol (matches report plan)

| Setting | Value |
|---|---|
| Warmup | 50 |
| Timed iterations | 300 |
| Batch | 1 |
| Precision | FP16 |
| Backend | NCNN Vulkan (default) |
| Inputs | 180×180 (audit) and 320×180 (720p deploy) |

---

## Part 6 — Put numbers in the report

1. Copy `deploy/artifacts/results/mobile_benchmark_latest.json` → `report/assets/metrics/` (or merge into `model_summary.json` per `report/SYNC.md`)
2. Add device table to `report.tex` RQ3 with **scoped** claims
3. Compare against RTX-4060 proxy latency in `report/assets/metrics/`

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `adb: no devices` | USB forwarding (WSL), cable, authorize prompt on phone |
| `onnx2ncnn not found` | Build NCNN host tools; `export PATH=...` |
| Vulkan bench crashes | `python scripts/bench_mobile.py --no-vulkan` |
| Inference failed (blob names) | Re-run convert; `python scripts/parse_ncnn_blobs.py deploy/artifacts/ncnn/MODEL.param` |
| FSRCNN ONNX shape mismatch | `verify_onnx_export.py` crops to common spatial size; OK if PASS |
| Permission denied on `/data/local/tmp` | Use a rooted phone or stay with `/data/local/tmp` (works on most dev devices) |

---

## Script map

| Script | Purpose |
|---|---|
| `scripts/export_onnx.py` | PyTorch → ONNX |
| `scripts/verify_onnx_export.py` | Numeric parity check |
| `scripts/deploy_smoke.py` | Export + verify one command |
| `deploy/convert_ncnn.sh` | ONNX → NCNN |
| `deploy/build_android_bench.sh` | Cross-compile `sr_bench` |
| `scripts/bench_mobile.py` | adb push + benchmark + JSON |
| `scripts/demo_mobile_sr.py` | Class presenter: adb PNG→PNG + side-by-side (Track D) |
| `scripts/make_demo_crops.py` | Curate `deploy/demo/crops/` from DIV2K val |
| `deploy/collect_device_info.sh` | Phone metadata |
| `deploy/models.json` | Model registry + input presets |

---

## Quick checklist

- [ ] Phone: USB debugging on, authorized
- [ ] `adb devices` → `device`
- [ ] `python scripts/deploy_smoke.py`
- [ ] `./deploy/convert_ncnn.sh`
- [ ] `./deploy/build_android_bench.sh`
- [ ] `python scripts/bench_mobile.py --preset deploy_720p`
- [ ] Sync JSON into `report/`
