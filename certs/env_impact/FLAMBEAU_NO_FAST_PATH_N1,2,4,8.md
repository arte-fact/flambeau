# Env-impact cert: `FLAMBEAU_NO_FAST_PATH`

- **Class:** D
- **Default state:** `fast-path-on`
- **Generated:** 2026-05-06T13:17:02+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** Disable scheduler fast path (force full coalescing logic)

**Preconditions:** `FLAMBEAU_INFLIGHT_SLOTS=$N`

## Measured cells

### qwen36-27b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 11443 | 2742 | 23.34 | — | — | 20.92 |  |
| `1` | 3 | 11484 | 2740 | 23.36 | +0.1% | match | 20.92 |  |

### qwen36-27b-q4_0 / pp2tp2 / N=2

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 17980 | 3482 | 30.36 | — | — | 21.37 |  |
| `1` | 3 | 17981 | 3438 | 30.91 | +1.8% | match | 21.37 |  |

### qwen36-27b-q4_0 / pp2tp2 / N=4

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 31144 | 4053 | 61.13 | — | — | 22.47 |  |
| `1` | 3 | 31056 | 4011 | 60.70 | -0.7% | match | 22.26 |  |

### qwen36-27b-q4_0 / pp2tp2 / N=8

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 57010 | 4006 | 118.96 | — | — | 24.32 |  |
| `1` | 3 | 57052 | 4025 | 124.05 | +4.3% | match | 24.08 |  |

### qwen36-35b-a3b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 3930 | 1449 | 44.17 | — | — | 23.05 |  |
| `1` | 3 | 3911 | 1474 | 43.42 | -1.7% | match | 23.05 |  |

### qwen36-35b-a3b-q4_0 / pp2tp2 / N=2

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 6250 | 1836 | 58.80 | — | — | 23.48 |  |
| `1` | 3 | 6286 | 1833 | 57.69 | -1.9% | match | 23.48 |  |

### qwen36-35b-a3b-q4_0 / pp2tp2 / N=4

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 11011 | 2188 | 113.26 | — | — | 24.34 |  |
| `1` | 3 | 11049 | 2212 | 114.14 | +0.8% | DIVERGENT | 24.34 |  |

### qwen36-35b-a3b-q4_0 / pp2tp2 / N=8

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 20504 | 2195 | 228.08 | — | — | 26.13 |  |
| `1` | 3 | 20507 | 2199 | 227.30 | -0.3% | DIVERGENT | 26.13 |  |

## Triage

- qwen36-27b-q4_0/pp2tp2 [N=1] `1` → **null** (+0.1%)
- qwen36-27b-q4_0/pp2tp2 [N=2] `1` → **null** (+1.8%)
- qwen36-27b-q4_0/pp2tp2 [N=4] `1` → **null** (-0.7%)
- qwen36-27b-q4_0/pp2tp2 [N=8] `1` → **win** (+4.3%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=1] `1` → **null** (-1.7%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=2] `1` → **null** (-1.9%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=4] `1` → **divergent** (+0.8%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=8] `1` → **divergent** (-0.3%)

## Disposition

- **HALT-DIVERGENT** — 2/8 cells produce different output at greedy/fixed-seed; correctness bug, not a perf gate — file before migration

## Notes from spec

At higher N the fast-path skip likely costs more (coalescing work
amplifies per dispatch). Verify the gate's slow path scales worse
under load — if so, dead path; if it actually helps at N=8, keep.
