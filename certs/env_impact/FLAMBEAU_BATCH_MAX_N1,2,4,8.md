# Env-impact cert: `FLAMBEAU_BATCH_MAX`

- **Class:** C
- **Default state:** `unbounded`
- **Generated:** 2026-05-06T12:26:21+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** Max batch size cap on scheduler dispatch

**Preconditions:** `FLAMBEAU_INFLIGHT_SLOTS=$N`

## Measured cells

### qwen36-27b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 11444 | 2741 | 23.35 | — | — | 20.92 |  |
| `2` | 3 | 11460 | 2728 | 23.46 | +0.5% | match | 20.92 |  |
| `4` | 3 | 11498 | 2732 | 23.43 | +0.3% | match | 20.92 |  |
| `8` | 3 | 11437 | 2746 | 23.31 | -0.2% | match | 20.92 |  |

### qwen36-27b-q4_0 / pp2tp2 / N=2

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 17937 | 3421 | 30.98 | — | — | 21.37 |  |
| `2` | 3 | 17957 | 3370 | 32.07 | +3.5% | match | 21.37 |  |
| `4` | 3 | 17944 | 3437 | 30.99 | +0.0% | match | 21.37 |  |
| `8` | 3 | 17969 | 3377 | 31.77 | +2.5% | match | 21.37 |  |

### qwen36-27b-q4_0 / pp2tp2 / N=4

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 31023 | 4032 | 61.25 | — | — | 22.26 |  |
| `2` | 3 | 31092 | 4019 | 59.50 | -2.9% | match | 22.26 |  |
| `4` | 3 | 30998 | 3971 | 62.12 | +1.4% | match | 22.26 |  |
| `8` | 3 | 31049 | 4019 | 62.05 | +1.3% | match | 22.28 |  |

### qwen36-27b-q4_0 / pp2tp2 / N=8

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 57138 | 4048 | 120.85 | — | — | 24.03 |  |
| `2` | 3 | 57115 | 4082 | 119.20 | -1.4% | match | 24.29 |  |
| `4` | 3 | 57072 | 4017 | 121.00 | +0.1% | match | 24.29 |  |
| `8` | 3 | 57146 | 4035 | 122.55 | +1.4% | match | 24.03 |  |

### qwen36-35b-a3b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 3927 | 1438 | 44.50 | — | — | 23.05 |  |
| `2` | 3 | 3913 | 1476 | 43.35 | -2.6% | match | 23.05 |  |
| `4` | 3 | 3909 | 1433 | 44.66 | +0.3% | match | 23.05 |  |
| `8` | 3 | 3918 | 1448 | 44.19 | -0.7% | match | 23.05 |  |

### qwen36-35b-a3b-q4_0 / pp2tp2 / N=2

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 6246 | 1811 | 58.49 | — | — | 23.48 |  |
| `2` | 3 | 6263 | 1825 | 57.66 | -1.4% | match | 23.48 |  |
| `4` | 3 | 6269 | 1852 | 56.81 | -2.9% | match | 23.48 |  |
| `8` | 3 | 6266 | 1823 | 58.45 | -0.1% | match | 23.48 |  |

### qwen36-35b-a3b-q4_0 / pp2tp2 / N=4

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 11019 | 2170 | 114.23 | — | — | 24.34 |  |
| `2` | 3 | 11034 | 2178 | 114.80 | +0.5% | match | 24.34 |  |
| `4` | 3 | 11001 | 2170 | 114.48 | +0.2% | match | 24.34 |  |
| `8` | 3 | 11052 | 2192 | 114.33 | +0.1% | match | 24.34 |  |

### qwen36-35b-a3b-q4_0 / pp2tp2 / N=8

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 20559 | 2225 | 225.07 | — | — | 26.13 |  |
| `2` | 3 | 20514 | 2194 | 224.55 | -0.2% | match | 26.13 |  |
| `4` | 3 | 20498 | 2214 | 227.25 | +1.0% | DIVERGENT | 26.13 |  |
| `8` | 3 | 20527 | 2201 | 222.78 | -1.0% | match | 26.13 |  |

## Triage

- qwen36-27b-q4_0/pp2tp2 [N=1] `2` → **null** (+0.5%)
- qwen36-27b-q4_0/pp2tp2 [N=1] `4` → **null** (+0.3%)
- qwen36-27b-q4_0/pp2tp2 [N=1] `8` → **null** (-0.2%)
- qwen36-27b-q4_0/pp2tp2 [N=2] `2` → **win** (+3.5%)
- qwen36-27b-q4_0/pp2tp2 [N=2] `4` → **null** (+0.0%)
- qwen36-27b-q4_0/pp2tp2 [N=2] `8` → **win** (+2.5%)
- qwen36-27b-q4_0/pp2tp2 [N=4] `2` → **loss** (-2.9%)
- qwen36-27b-q4_0/pp2tp2 [N=4] `4` → **null** (+1.4%)
- qwen36-27b-q4_0/pp2tp2 [N=4] `8` → **null** (+1.3%)
- qwen36-27b-q4_0/pp2tp2 [N=8] `2` → **null** (-1.4%)
- qwen36-27b-q4_0/pp2tp2 [N=8] `4` → **null** (+0.1%)
- qwen36-27b-q4_0/pp2tp2 [N=8] `8` → **null** (+1.4%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=1] `2` → **loss** (-2.6%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=1] `4` → **null** (+0.3%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=1] `8` → **null** (-0.7%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=2] `2` → **null** (-1.4%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=2] `4` → **loss** (-2.9%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=2] `8` → **null** (-0.1%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=4] `2` → **null** (+0.5%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=4] `4` → **null** (+0.2%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=4] `8` → **null** (+0.1%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=8] `2` → **null** (-0.2%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=8] `4` → **divergent** (+1.0%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=8] `8` → **null** (-1.0%)

## Disposition

- **HALT-DIVERGENT** — 1/24 cells produce different output at greedy/fixed-seed; correctness bug, not a perf gate — file before migration

## Notes from spec

At N=1 the cap never engages. At N=8/slots=8: cap=2 forces 4 separate
scheduler dispatches per round; cap=8 = one round. Identifies the
per-dispatch overhead inflection point (where bigger cap stops helping).
