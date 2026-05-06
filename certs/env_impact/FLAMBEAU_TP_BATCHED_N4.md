# Env-impact cert: `FLAMBEAU_TP_BATCHED`

- **Class:** B
- **Default state:** `on`
- **Generated:** 2026-05-06T07:30:37+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** Batched TP forward (default ON; opt-out gate)

## Measured cells

### qwen36-27b-q4_0 / tp2 / N=4

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 33625 | 5100 | 49.13 | — | — |  |
| `0` | 2 | 394412 | 5509 | 40.59 | -17.4% | DIVERGENT |  |

## Triage

- qwen36-27b-q4_0/tp2 [N=4] `0` → **divergent** (-17.4%)

## Disposition

- **HALT-DIVERGENT** — 1/1 cells produce different output at greedy/fixed-seed; correctness bug, not a perf gate — file before migration

## Notes from spec

Re-test the opt-out at N=4 to see if the batching path's absence still
hurts under concurrent load (it should be even worse — batched-decode
needs the batched-TP forward to coalesce). Confirms whether the gate
can be deleted entirely vs kept for fall-back.
