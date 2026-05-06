# Env-impact cert: `FLAMBEAU_Q8_0_MMVQ_T128_VDR2`

- **Class:** B
- **Default state:** `off`
- **Generated:** 2026-05-05T21:35:46+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** Q8_0 MMVQ T=128 + VDR2 inner

## Measured cells

### qwen36-27b-q8_0 / pp4

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 36818 | 3952 | 16.20 | — | — |  |
| `off` | 3 | 36663 | 3986 | 16.06 | -0.9% | match |  |
| `on` | 3 | 36590 | 3968 | 16.13 | -0.4% | match |  |

### qwen36-27b-q8_0 / tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 19494 | 3100 | 20.64 | — | — |  |
| `off` | 3 | 19863 | 3266 | 19.59 | -5.1% | match |  |
| `on` | 3 | 20012 | 3154 | 20.29 | -1.7% | match |  |

## Triage

- qwen36-27b-q8_0/pp4 `off` → **null** (-0.9%)
- qwen36-27b-q8_0/pp4 `on` → **null** (-0.4%)
- qwen36-27b-q8_0/tp2 `off` → **loss** (-5.1%)
- qwen36-27b-q8_0/tp2 `on` → **null** (-1.7%)

## Disposition

- **CONTEXT-DEPENDENT** — 1/4 comparisons loss, rest null. Specifically:
  on qwen36-27b-q8_0/tp2, explicit `off` is -5.1% vs default `unset`,
  while explicit `on` is null (-1.7%). This means **`unset` and `off`
  take different code paths** — the `=off` branch has a hidden
  regression vs the implicit-default branch. Investigate the dispatch
  fork; likely a stale `else` branch. The 3-way `[unset, off, on]`
  test design caught this.

_(Disposition aggregate added post-hoc on 2026-05-05.)_

## Notes from spec

Read in ops/qmatmul.rs:548,995. Known partial cert (c9_followup).
