# Env-impact cert: `FLAMBEAU_ASYNC_GRAPH`

- **Class:** B
- **Default state:** `off`
- **Generated:** 2026-05-05T23:00:31+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** Async graph capture (V2.26.a-i5c experiment)

## Measured cells

### qwen36-27b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 11380 | 2729 | 23.45 | — | — |  |
| `1` | 3 | 11401 | 2726 | 23.48 | +0.1% | match |  |

### qwen36-27b-q4_0 / pp4

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 19006 | 3806 | 16.82 | — | — |  |
| `1` | 3 | 19009 | 3842 | 16.66 | -0.9% | match |  |

### qwen36-35b-a3b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 3900 | 1423 | 44.98 | — | — |  |
| `1` | 3 | 3904 | 1444 | 44.34 | -1.4% | match |  |

### qwen36-35b-a3b-q4_0 / pp4

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 5385 | 1534 | 41.71 | — | — |  |
| `1` | 3 | 5364 | 1571 | 40.73 | -2.4% | match |  |

## Triage

- qwen36-27b-q4_0/pp2tp2 `1` → **null** (+0.1%)
- qwen36-27b-q4_0/pp4 `1` → **null** (-0.9%)
- qwen36-35b-a3b-q4_0/pp2tp2 `1` → **null** (-1.4%)
- qwen36-35b-a3b-q4_0/pp4 `1` → **loss** (-2.4%)

## Disposition

- **CONTEXT-DEPENDENT** — 1/4 cells loss, rest null; consider dispatch-table row for the affected cells

## Notes from spec

Same expected outcome as DECODE_GRAPH. Confirm null and delete.
