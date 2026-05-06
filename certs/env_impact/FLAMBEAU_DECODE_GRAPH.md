# Env-impact cert: `FLAMBEAU_DECODE_GRAPH`

- **Class:** B
- **Default state:** `off`
- **Generated:** 2026-05-05T22:49:33+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** Capture decode loop into HIP graph

## Measured cells

### qwen35-9b-q4_1 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 4235 | 1270 | 50.38 | — | — |  |
| `1` | 0 | 0 | 0 | 0.00 | -100.0% | n/a | no first chunk |

### qwen35-9b-q4_1 / pp4

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 6014 | 1457 | 43.93 | — | — |  |
| `1` | 3 | 6009 | 1490 | 42.96 | -2.2% | match |  |

### qwen35-9b-q4_1 / tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 4063 | 1208 | 52.98 | — | — |  |
| `1` | 3 | 4104 | 1211 | 52.85 | -0.3% | match |  |

### qwen36-27b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 11422 | 2741 | 23.35 | — | — |  |
| `1` | 0 | 0 | 0 | 0.00 | -100.0% | n/a | no first chunk |

### qwen36-27b-q4_0 / pp4

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 19038 | 3843 | 16.65 | — | — |  |
| `1` | 3 | 19027 | 3800 | 16.84 | +1.1% | match |  |

### qwen36-27b-q4_0 / tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 11366 | 2978 | 21.49 | — | — |  |
| `1` | 3 | 11456 | 3009 | 21.27 | -1.0% | match |  |

### qwen36-35b-a3b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 3903 | 1449 | 44.17 | — | — |  |
| `1` | 0 | 0 | 0 | 0.00 | -100.0% | n/a | no first chunk |

### qwen36-35b-a3b-q4_0 / pp4

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 5367 | 1570 | 40.77 | — | — |  |
| `1` | 3 | 5362 | 1545 | 41.41 | +1.6% | match |  |

### qwen36-35b-a3b-q4_0 / tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 3756 | 1261 | 50.76 | — | — |  |
| `1` | 3 | 3765 | 1260 | 50.78 | +0.0% | match |  |

## Triage

- qwen35-9b-q4_1/pp2tp2 `1` → **broken** (-100.0%)
- qwen35-9b-q4_1/pp4 `1` → **loss** (-2.2%)
- qwen35-9b-q4_1/tp2 `1` → **null** (-0.3%)
- qwen36-27b-q4_0/pp2tp2 `1` → **broken** (-100.0%)
- qwen36-27b-q4_0/pp4 `1` → **null** (+1.1%)
- qwen36-27b-q4_0/tp2 `1` → **null** (-1.0%)
- qwen36-35b-a3b-q4_0/pp2tp2 `1` → **broken** (-100.0%)
- qwen36-35b-a3b-q4_0/pp4 `1` → **null** (+1.6%)
- qwen36-35b-a3b-q4_0/tp2 `1` → **null** (+0.0%)

## Disposition

- **HALT-BROKEN** — 3/9 cells crashed; do not migrate before fix

## Notes from spec

Single confirmation cert across the live model/topology matrix, then
delete the var. Memory says don't propose this as a perf win on MI50.
