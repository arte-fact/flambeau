# Env-impact cert: `FLAMBEAU_ASYNC_UBATCH`

- **Class:** B
- **Default state:** `off`
- **Generated:** 2026-05-06T14:07:43+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** Async ubatch in PP forward

**Preconditions:** `FLAMBEAU_INFLIGHT_SLOTS=$N`

## Measured cells

### qwen36-27b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 11513 | 2729 | 23.45 | — | — | 20.92 |  |
| `1` | 3 | 11446 | 2719 | 23.54 | +0.4% | match | 20.92 |  |

### qwen36-27b-q4_0 / pp2tp2 / N=2

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 17964 | 3466 | 30.54 | — | — | 21.37 |  |
| `1` | 3 | 17973 | 3437 | 31.00 | +1.5% | match | 21.37 |  |

### qwen36-27b-q4_0 / pp2tp2 / N=4

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 31054 | 4022 | 61.99 | — | — | 22.33 |  |
| `1` | 3 | 30994 | 4060 | 60.07 | -3.1% | match | 22.26 |  |

### qwen36-27b-q4_0 / pp2tp2 / N=8

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 57087 | 4059 | 120.34 | — | — | 24.17 |  |
| `1` | 3 | 57155 | 4111 | 120.18 | -0.1% | match | 24.03 |  |

### qwen36-35b-a3b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 3923 | 1428 | 44.82 | — | — | 23.05 |  |
| `1` | 3 | 3914 | 1449 | 44.18 | -1.4% | match | 23.05 |  |

### qwen36-35b-a3b-q4_0 / pp2tp2 / N=2

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 6271 | 1826 | 57.74 | — | — | 23.48 |  |
| `1` | 3 | 6272 | 1814 | 58.18 | +0.8% | DIVERGENT | 23.48 |  |

### qwen36-35b-a3b-q4_0 / pp2tp2 / N=4

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 11038 | 2220 | 113.99 | — | — | 24.34 |  |
| `1` | 3 | 11085 | 2211 | 112.13 | -1.6% | match | 24.34 |  |

### qwen36-35b-a3b-q4_0 / pp2tp2 / N=8

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 20496 | 2203 | 227.24 | — | — | 26.13 |  |
| `1` | 3 | 20580 | 2210 | 226.31 | -0.4% | DIVERGENT | 26.13 |  |

## Triage

- qwen36-27b-q4_0/pp2tp2 [N=1] `1` → **null** (+0.4%)
- qwen36-27b-q4_0/pp2tp2 [N=2] `1` → **null** (+1.5%)
- qwen36-27b-q4_0/pp2tp2 [N=4] `1` → **loss** (-3.1%)
- qwen36-27b-q4_0/pp2tp2 [N=8] `1` → **null** (-0.1%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=1] `1` → **null** (-1.4%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=2] `1` → **divergent** (+0.8%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=4] `1` → **null** (-1.6%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=8] `1` → **divergent** (-0.4%)

## Disposition

- **HALT-DIVERGENT** — 2/8 cells produce different output at greedy/fixed-seed; correctness bug, not a perf gate — file before migration

## Notes from spec

Async ubatch overlaps PP stages — win shows up under concurrent prefill load.
