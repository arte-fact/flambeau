# Env-impact cert: `FLAMBEAU_PHASE3B_KV_F16`

- **Class:** X
- **Default state:** `n/a`
- **Generated:** 2026-05-07T17:34:25+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** Phase 3b — F16 KV baseline arm

**Preconditions:** `FLAMBEAU_INFLIGHT_SLOTS=8`, `FLAMBEAU_KV=f16`

## Measured cells

### qwen36-27b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 11451 | 2731 | 23.43 | — | — | 24.03 |  |

### qwen36-27b-q4_0 / pp2tp2 / N=4

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 31010 | 3972 | 62.06 | — | — | 24.26 |  |

### qwen36-35b-a3b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 3909 | 1447 | 44.22 | — | — | 25.82 |  |

### qwen36-35b-a3b-q4_0 / pp2tp2 / N=4

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 11081 | 2163 | 115.29 | — | — | 26.13 |  |

## Triage

_No comparison rows produced._

## Disposition

- INSUFFICIENT-DATA

## Notes from spec

F16 KV layout (canonical baseline).
