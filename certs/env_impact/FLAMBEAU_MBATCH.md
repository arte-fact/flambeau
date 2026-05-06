# Env-impact cert: `FLAMBEAU_MBATCH`

- **Class:** B
- **Default state:** `off`
- **Generated:** 2026-05-05T22:06:36+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** MoE batched dispatch gate

**Preconditions:** `FLAMBEAU_BATCHED_DECODE=1`, `FLAMBEAU_INFLIGHT_SLOTS=4`

## Measured cells

### qwen36-35b-a3b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 3909 | 1449 | 44.17 | — | — |  |
| `1` | 3 | 3912 | 1432 | 44.68 | +1.2% | match |  |

### qwen36-35b-a3b-q4_0 / tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 3768 | 1269 | 50.44 | — | — |  |
| `1` | 3 | 3761 | 1268 | 50.47 | +0.1% | match |  |

## Triage

- qwen36-35b-a3b-q4_0/pp2tp2 `1` → **null** (+1.2%)
- qwen36-35b-a3b-q4_0/tp2 `1` → **null** (+0.1%)

## Disposition

- **CANDIDATE-DELETE** — no measured impact on any of 2 cells; bake default and remove gate

## Notes from spec

Read in ops/moe.rs:1251. Standalone delta unmeasured post-#288.
