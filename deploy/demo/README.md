# Class demo (Track D) — on-device PECSR via adb + NCNN

**Consensus:** `progress/track_d.md`. No webpage / no APK.

## Chrome (audience line)

Printed by `scripts/demo_mobile_sr.py` and burned into `side_by_side.png` title:

`PECSR · <phone model> · on device · NCNN Vulkan FP16 · LR 180×180 · XX.X ms`

Offline failsafe uses `pre-recorded` instead of `on device`.

## Runbook

```bash
cd Image-Super-Resolution
# once after C++ change:
./deploy/build_android_bench.sh

# crops (once):
/home/hyb/miniforge3/envs/cv_env/bin/python scripts/make_demo_crops.py

# live (phone USB + adb):
/home/hyb/miniforge3/envs/cv_env/bin/python scripts/demo_mobile_sr.py \
  --model pecsr --preset audit_180 \
  --lr deploy/demo/crops/crop01_texture_lr180.png \
  --save-failsafe --show

# vulkan fail → add --no-vulkan
# no phone → offline:
/home/hyb/miniforge3/envs/cv_env/bin/python scripts/demo_mobile_sr.py \
  --offline --failsafe-stem crop01_texture_lr180 --show
```

## Layout

| Path | Role |
|---|---|
| `crops/` | Curated LR 180² (+ `catalog.json`) |
| `out/` | Live pulls / side-by-side (local; gitignored) |
| `failsafe/` | Pre-recorded LR/HR + meta for offline |

NCNN weights stay under `deploy/artifacts/ncnn/` (not copied here).
