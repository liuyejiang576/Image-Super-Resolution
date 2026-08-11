# Inactive scripts (do not run as current workflow)

| Path | Why |
|---|---|
| `stage_b/` | Plus 2k / VGG probes finished |
| `kd_probes/` | λ-sweep / KD parallel probes finished |
| `kd_diag/` | `gate_kd_methods`, `diagnose_kd`, `kd_per_image_analysis` — RQ3 evidence regen only |
| `lab_plot_dup/` | Lab copies superseded by **`../../report/plot/`** (canonical) |
| `profile_fsrcnn.py`, `compare_metrics.py` | Thin / early utilities |
| `quantize_benchmark.py` | CPU proxy; not deploy spine |
| KD shells / old postprocess / report builders | Superseded |

**Canonical figures & GPU latency:** `cd ../../report/plot && python …` (see `../../report/SYNC.md`).
