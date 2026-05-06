# Env-impact cert: `FLAMBEAU_ASYNC_UBATCH`

- **Class:** B
- **Default state:** `off`
- **Generated:** 2026-05-05T23:07:02+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** Async ubatch in PP forward

## Measured cells

### qwen36-27b-q4_0 / pp4

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 19005 | 3842 | 16.66 | — | — |  |
| `1` | 3 | 19011 | 3823 | 16.74 | +0.5% | match |  |

### qwen36-35b-a3b-q4_0 / pp4

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 5348 | 1562 | 40.98 | — | — |  |
| `1` | 3 | 5343 | 1544 | 41.45 | +1.1% | match |  |

## Triage

- qwen36-27b-q4_0/pp4 `1` → **null** (+0.5%)
- qwen36-35b-a3b-q4_0/pp4 `1` → **null** (+1.1%)

## Disposition

- **CANDIDATE-DELETE** — no measured impact on any of 2 cells; bake default and remove gate

## Notes from spec

Read in models/pp.rs:1179.
