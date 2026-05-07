# Env-impact cert: `FLAMBEAU_PHASE3B_KV_Q8`

- **Class:** X
- **Default state:** `n/a`
- **Generated:** 2026-05-07T17:36:07+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** Phase 3b — Q8 KV arm

**Preconditions:** `FLAMBEAU_INFLIGHT_SLOTS=8`, `FLAMBEAU_KV=q8`

## Measured cells

### qwen36-27b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 0 | 0 | 0 | 0.00 | — | — | no first chunk |

### qwen36-27b-q4_0 / pp2tp2 / N=4

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 0 | 0 | 0 | 0.00 | — | — | no first chunk |

### qwen36-35b-a3b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 0 | 0 | 0 | 0.00 | — | — | no first chunk |

### qwen36-35b-a3b-q4_0 / pp2tp2 / N=4

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 0 | 0 | 0 | 0.00 | — | — | no first chunk |

## Triage

_No comparison rows produced._

## Disposition

- INSUFFICIENT-DATA

## Notes from spec

Q8 KV layout (~2× HBM saving on decode).
