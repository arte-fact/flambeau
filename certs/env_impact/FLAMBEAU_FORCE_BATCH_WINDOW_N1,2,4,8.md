# Env-impact cert: `FLAMBEAU_FORCE_BATCH_WINDOW`

- **Class:** C
- **Default state:** `off`
- **Generated:** 2026-05-06T10:45:29+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** Force batch-window timeout even when batch is empty

**Preconditions:** `FLAMBEAU_INFLIGHT_SLOTS=$N`, `FLAMBEAU_BATCH_WINDOW_US=1500`

## Measured cells

### qwen36-27b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 11535 | 2737 | 23.38 | — | — | 20.92 |  |
| `1` | 3 | 11426 | 2745 | 23.32 | -0.3% | match | 20.92 |  |

### qwen36-27b-q4_0 / pp2tp2 / N=2

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 17955 | 3370 | 31.96 | — | — | 21.37 |  |
| `1` | 3 | 17929 | 3386 | 31.79 | -0.5% | match | 21.37 |  |

### qwen36-27b-q4_0 / pp2tp2 / N=4

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 30993 | 4003 | 60.81 | — | — | 22.26 |  |
| `1` | 3 | 31026 | 3991 | 61.63 | +1.3% | match | 22.47 |  |

### qwen36-27b-q4_0 / pp2tp2 / N=8

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 57011 | 3991 | 122.66 | — | — | 24.10 |  |
| `1` | 3 | 57009 | 4013 | 123.53 | +0.7% | match | 24.28 |  |

### qwen36-35b-a3b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 3919 | 1448 | 44.20 | — | — | 23.05 |  |
| `1` | 3 | 3927 | 1438 | 44.51 | +0.7% | match | 23.05 |  |

### qwen36-35b-a3b-q4_0 / pp2tp2 / N=2

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 6276 | 1824 | 57.93 | — | — | 23.48 |  |
| `1` | 3 | 6273 | 1833 | 57.75 | -0.3% | match | 23.48 |  |

### qwen36-35b-a3b-q4_0 / pp2tp2 / N=4

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 11052 | 2200 | 112.76 | — | — | 24.34 |  |
| `1` | 3 | 11085 | 2186 | 113.51 | +0.7% | DIVERGENT | 24.34 |  |

### qwen36-35b-a3b-q4_0 / pp2tp2 / N=8

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 20534 | 2206 | 223.70 | — | — | 26.13 |  |
| `1` | 3 | 20531 | 2201 | 225.63 | +0.9% | match | 26.13 |  |

## Triage

- qwen36-27b-q4_0/pp2tp2 [N=1] `1` → **null** (-0.3%)
- qwen36-27b-q4_0/pp2tp2 [N=2] `1` → **null** (-0.5%)
- qwen36-27b-q4_0/pp2tp2 [N=4] `1` → **null** (+1.3%)
- qwen36-27b-q4_0/pp2tp2 [N=8] `1` → **null** (+0.7%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=1] `1` → **null** (+0.7%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=2] `1` → **null** (-0.3%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=4] `1` → **divergent** (+0.7%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=8] `1` → **null** (+0.9%)

## Disposition

- **HALT-DIVERGENT** — 1/8 cells produce different output at greedy/fixed-seed; correctness bug, not a perf gate — file before migration

## Notes from spec

At N=1 the empty-batch timeout never fires. At N=2/4/8 the window gates
how aggressively the scheduler coalesces — if forcing the timeout
helps tail latency or hurts throughput, the gate is real. Watch for
non-monotonic scaling across N (e.g. =1 might help at N=8 but hurt at N=2).
