# Env-impact cert: `FLAMBEAU_MBATCH`

- **Class:** B
- **Default state:** `off`
- **Generated:** 2026-05-06T09:54:52+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** MoE batched dispatch gate

**Preconditions:** `FLAMBEAU_INFLIGHT_SLOTS=$N`

## Measured cells

### qwen36-27b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 11455 | 2754 | 23.24 | — | — | 20.92 |  |
| `1` | 3 | 11413 | 2739 | 23.36 | +0.5% | match | 20.92 |  |

### qwen36-27b-q4_0 / pp2tp2 / N=2

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 17961 | 3405 | 31.48 | — | — | 21.61 |  |
| `1` | 3 | 17957 | 3403 | 31.49 | +0.0% | match | 21.37 |  |

### qwen36-27b-q4_0 / pp2tp2 / N=4

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 31029 | 3954 | 63.66 | — | — | 22.33 |  |
| `1` | 3 | 31071 | 4095 | 59.78 | -6.1% | match | 22.26 |  |

### qwen36-27b-q4_0 / pp2tp2 / N=8

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 57027 | 4065 | 120.82 | — | — | 24.03 |  |
| `1` | 3 | 57013 | 4027 | 121.26 | +0.4% | match | 24.03 |  |

### qwen36-35b-a3b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 3923 | 1440 | 44.43 | — | — | 23.05 |  |
| `1` | 3 | 3912 | 1430 | 44.74 | +0.7% | match | 23.05 |  |

### qwen36-35b-a3b-q4_0 / pp2tp2 / N=2

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 6275 | 1856 | 57.52 | — | — | 23.48 |  |
| `1` | 3 | 6269 | 1844 | 58.59 | +1.9% | match | 23.48 |  |

### qwen36-35b-a3b-q4_0 / pp2tp2 / N=4

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 10954 | 2158 | 113.75 | — | — | 24.34 |  |
| `1` | 3 | 11017 | 2149 | 110.88 | -2.5% | match | 24.34 |  |

### qwen36-35b-a3b-q4_0 / pp2tp2 / N=8

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 20527 | 2206 | 224.04 | — | — | 26.13 |  |
| `1` | 3 | 20458 | 2202 | 226.04 | +0.9% | match | 26.13 |  |

## Triage

- qwen36-27b-q4_0/pp2tp2 [N=1] `1` → **null** (+0.5%)
- qwen36-27b-q4_0/pp2tp2 [N=2] `1` → **null** (+0.0%)
- qwen36-27b-q4_0/pp2tp2 [N=4] `1` → **loss** (-6.1%)
- qwen36-27b-q4_0/pp2tp2 [N=8] `1` → **null** (+0.4%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=1] `1` → **null** (+0.7%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=2] `1` → **null** (+1.9%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=4] `1` → **loss** (-2.5%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=8] `1` → **null** (+0.9%)

## Disposition

- **CONTEXT-DEPENDENT** — 2/8 cells loss, rest null; consider dispatch-table row for the affected cells

## Notes from spec

Real MoE test deferred until 35B-A3B/pp2tp2 fits with INFLIGHT_SLOTS≥4.
