# Env-impact cert: `FLAMBEAU_PHASE3D_NO_GPU_SAMPLER`

- **Class:** X
- **Default state:** `n/a`
- **Generated:** 2026-05-07T18:03:35+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** Phase 3d — disable on-device GPU sampler

**Preconditions:** `FLAMBEAU_INFLIGHT_SLOTS=8`, `FLAMBEAU_NO_GPU_SAMPLER=1`

## Measured cells

### qwen36-27b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 11325 | 2756 | 23.23 | — | — | 24.03 |  |

### qwen36-27b-q4_0 / pp2tp2 / N=4

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 30874 | 4006 | 62.47 | — | — | 24.04 |  |

### qwen36-35b-a3b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 3904 | 1434 | 44.63 | — | — | 25.82 |  |

### qwen36-35b-a3b-q4_0 / pp2tp2 / N=4

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 11133 | 2185 | 114.49 | — | — | 26.13 |  |

## Triage

_No comparison rows produced._

## Disposition

- INSUFFICIENT-DATA

## Notes from spec

Validate the opt-out path still works; quantify regression.
