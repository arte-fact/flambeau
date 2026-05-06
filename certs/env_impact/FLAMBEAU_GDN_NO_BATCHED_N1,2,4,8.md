# Env-impact cert: `FLAMBEAU_GDN_NO_BATCHED`

- **Class:** B
- **Default state:** `batched`
- **Generated:** 2026-05-06T09:04:09+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** Force per-token GDN (disable batched-GDN inner loop)

**Preconditions:** `FLAMBEAU_INFLIGHT_SLOTS=$N`

## Measured cells

### qwen36-27b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 11559 | 2738 | 23.38 | — | — | 20.92 |  |
| `1` | 3 | 11475 | 2733 | 23.42 | +0.2% | match | 20.92 |  |

### qwen36-27b-q4_0 / pp2tp2 / N=2

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 17946 | 3360 | 32.00 | — | — | 21.37 |  |
| `1` | 3 | 17962 | 3414 | 31.34 | -2.1% | match | 21.37 |  |

### qwen36-27b-q4_0 / pp2tp2 / N=4

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 31088 | 4031 | 62.63 | — | — | 22.26 |  |
| `1` | 3 | 30989 | 4033 | 59.94 | -4.3% | match | 22.27 |  |

### qwen36-27b-q4_0 / pp2tp2 / N=8

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 56927 | 4014 | 120.90 | — | — | 24.13 |  |
| `1` | 3 | 57052 | 4036 | 116.29 | -3.8% | match | 24.03 |  |

### qwen36-35b-a3b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 3916 | 1430 | 44.77 | — | — | 23.05 |  |
| `1` | 3 | 3925 | 1446 | 44.26 | -1.1% | match | 23.05 |  |

### qwen36-35b-a3b-q4_0 / pp2tp2 / N=2

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 6271 | 1861 | 56.96 | — | — | 23.48 |  |
| `1` | 3 | 6247 | 1824 | 58.58 | +2.9% | match | 23.48 |  |

### qwen36-35b-a3b-q4_0 / pp2tp2 / N=4

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 11040 | 2198 | 112.49 | — | — | 24.34 |  |
| `1` | 3 | 11099 | 2204 | 112.50 | +0.0% | match | 24.34 |  |

### qwen36-35b-a3b-q4_0 / pp2tp2 / N=8

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 20430 | 2190 | 224.87 | — | — | 26.13 |  |
| `1` | 3 | 20493 | 2216 | 220.94 | -1.7% | DIVERGENT | 26.13 |  |

## Triage

- qwen36-27b-q4_0/pp2tp2 [N=1] `1` → **null** (+0.2%)
- qwen36-27b-q4_0/pp2tp2 [N=2] `1` → **loss** (-2.1%)
- qwen36-27b-q4_0/pp2tp2 [N=4] `1` → **loss** (-4.3%)
- qwen36-27b-q4_0/pp2tp2 [N=8] `1` → **loss** (-3.8%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=1] `1` → **null** (-1.1%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=2] `1` → **win** (+2.9%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=4] `1` → **null** (+0.0%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=8] `1` → **divergent** (-1.7%)

## Disposition

- **HALT-DIVERGENT** — 1/8 cells produce different output at greedy/fixed-seed; correctness bug, not a perf gate — file before migration

## Notes from spec

The N=2 cert flagged a divergent-output bug on 35B-A3B/pp2tp2 with =1
(per-token GDN). Full N sweep checks if the bug is N-dependent and
characterizes the perf delta vs default across the concurrency curve.
