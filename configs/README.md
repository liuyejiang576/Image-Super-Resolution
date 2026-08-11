# configs/ — what to open

**Rule:** Locked training for claims / manifests uses **`exp/*.yaml` only**.  
Root `train_*.yaml` = templates (defaults for scripts / `generate_exp_configs.py`).  
**Do not** blind-run `scripts/generate_exp_configs.py` — it can overwrite `exp/` and omits Plus (`open_wont` / IMPLEMENTATION).

---

## Layout

```text
configs/
  data.yaml           Shared dataset paths
  train_*.yaml        Templates / script defaults (not headline recipes)
  exp/                Locked experiment recipes ← use these
  _inactive/          10k, 2k probes, old FSRCNN variants — do not re-run
```

---

## `exp/` map (active)

| File | Role | Manifest / control |
|---|---|---|
| `mobile_srnet_20k.yaml` | Lite / Base fair 20k | `fair_budget_manifest.json` |
| `mobile_srnet_plus_20k.yaml` | Plus fair 20k (headline) | `plus_20k_manifest.json` · `plus_20k.py` |
| `fsrcnn_fix_clean_20k.yaml` | FSRCNN fair 20k **bs=8** (historical table) | `fair_budget_manifest.json` |
| `fsrcnn_fix_clean_20k_bs24.yaml` | FSRCNN equal-batch **A1a** | `a1a_20k_manifest.json` · `a1a_20k.py` |
| `ecbsr_m10c16_20k.yaml` | ECBSR baseline **B3**（几何 M10C16；报告显示 **ECBSR**） | `ecbsr_20k_manifest.json` · `ecbsr_20k.py` |
| `sepres_v2_c16n8_20k.yaml` | B4 v2_a (gated) | `sepres_v2_20k_manifest.json` · `sepres_v2_20k.py --run-id` |
| `sepres_v2_c16n10_20k.yaml` | B4 v2_b **first train** ≡ PECSR | `sepres_v2_20k_manifest.json` · `sepres_v2_20k.py` |
| `sepres_v2_c20n6_20k.yaml` | B4 v2_c (gated) | `sepres_v2_20k_manifest.json` · `sepres_v2_20k.py --run-id` |
| `abl_plain3x3_c16n10_20k.yaml` | B5a P0 plain body | `b5_train_20k_manifest.json` · `b5_train_20k.py` |
| `sepres_v2_c16n10_20k_s123.yaml` / `_s2026.yaml` | B5 PECSR seeds | `b5_train_20k_manifest.json` |
| `ecbsr_m10c16_20k_s123.yaml` / `_s2026.yaml` | B5 ECBSR seeds | `b5_train_20k_manifest.json` |
| `pecsr_pixel_kd0_2k.yaml` / `pecsr_pixel_kd02_2k.yaml` | B5 PECSR Stage-B pixel KD λ=0/0.2 | `b5_train_20k_manifest.json` · **MSE 后** |
| `pecsr_vgg3_kd0_2k.yaml` / `pecsr_vgg3_kd01_2k.yaml` | B5 PECSR Stage-B VGG relu3 λ=0/0.01 | same · **method-local λ** |
| `mobile_srnet_kd0_20k.yaml` | KD λ=0 (Charbonnier) **historical Lite** | `fair_budget_manifest.json` |
| `mobile_srnet_kd02_20k.yaml` | KD λ=0.2 **historical Lite** | `fair_budget_manifest.json` |
| `fsrcnn_fix_clean_30k.yaml` | 20k→30k continuation | `arch_30k_manifest.json` |
| `mobile_srnet_30k.yaml` | same | `arch_30k_manifest.json` |
| `mobile_srnet_plus_30k.yaml` | same | `arch_30k_manifest.json` |

**Tags:** headline 20k · fair-debt (A1a/B3) · KD pair · cont_30k.  
After A1a replaces the table: consider moving bs8 `fsrcnn_fix_clean_20k.yaml` → `_inactive/exp/`.  
If 30k is dropped from narrative: move `*_30k.yaml` → `_inactive/exp/`.

---

## Root templates

| File | Used as default by |
|---|---|
| `data.yaml` | `sanity_check_data.py`, datasets |
| `train_mobile_srnet.yaml` | `train_mobile_srnet.py` |
| `train_mobile_srnet_kd.yaml` | `train_mobile_srnet_kd.py` |
| `train_fsrcnn.yaml` / `train_fsrcnn_fix_clean.yaml` | `train_fsrcnn.py` / generator |
| `train_ecbsr.yaml` | `train_ecbsr.py` |
| `train_sepres_v2.yaml` | `train_sepres_v2.py` (prefer `exp/sepres_v2_*`) |

Prefer launching via `python scripts/<job>_20k.py resume` so the **exp/** path from the manifest is used.

---

## `_inactive/`

| Area | Contents |
|---|---|
| `_inactive/exp/` | `*_10k`, Plus/VGG `*_2k`, KD λ-sweep 10k, superseded `pecsr_kd*_20k` |
| `_inactive/train_*.yaml` | Old FSRCNN fix / small / fast-resume |

See also: [`../IMPLEMENTATION.md`](../IMPLEMENTATION.md) §3 · [`../results/MANIFEST.md`](../results/MANIFEST.md) · report [`appendix_training_recipes.md`](../../report/appendix_training_recipes.md).
