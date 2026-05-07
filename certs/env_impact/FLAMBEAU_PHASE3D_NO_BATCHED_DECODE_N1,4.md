# Env-impact cert: `FLAMBEAU_PHASE3D_NO_BATCHED_DECODE`

- **Class:** X
- **Default state:** `n/a`
- **Generated:** 2026-05-07T18:12:59+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** Phase 3d — disable batched-decode scheduler

**Preconditions:** `FLAMBEAU_INFLIGHT_SLOTS=8`, `FLAMBEAU_NO_BATCHED_DECODE=1`

## Measured cells

### qwen36-27b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 11500 | 2737 | 23.38 | — | — | 24.03 |  |

### qwen36-27b-q4_0 / pp2tp2 / N=4

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 31079 | 4118 | 60.16 | — | — | 24.12 |  |

### qwen36-35b-a3b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 3907 | 1453 | 44.04 | — | — | 25.82 |  |

### qwen36-35b-a3b-q4_0 / pp2tp2 / N=4

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 11079 | 2180 | 112.85 | — | — | 26.13 |  |

## Triage

_No comparison rows produced._

## Disposition

- INSUFFICIENT-DATA

## Notes from spec

Validate the opt-out path still works; quantify regression.
