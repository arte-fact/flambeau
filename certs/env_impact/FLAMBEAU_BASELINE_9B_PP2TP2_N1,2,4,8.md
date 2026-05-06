# Env-impact cert: `FLAMBEAU_BASELINE_9B_PP2TP2`

- **Class:** X
- **Default state:** `n/a`
- **Generated:** 2026-05-06T08:13:31+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** Baseline characterization of Qwen3.5-9B on pp2tp2 across N=[1,2,4,8]

**Preconditions:** `FLAMBEAU_INFLIGHT_SLOTS=8`

## Measured cells

### qwen35-9b-q4_1 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 4418 | 1870 | 34.23 | — | — | 23.69 |  |

### qwen35-9b-q4_1 / pp2tp2 / N=2

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 7969 | 2219 | 51.43 | — | — | 23.69 |  |

### qwen35-9b-q4_1 / pp2tp2 / N=4

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 13876 | 2610 | 96.23 | — | — | 23.69 |  |

### qwen35-9b-q4_1 / pp2tp2 / N=8

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 26752 | 2754 | 177.55 | — | — | 23.69 |  |

## Triage

_No comparison rows produced._

## Disposition

- INSUFFICIENT-DATA

## Notes from spec

Measures aggregate throughput + TTFT p99 + peak VRAM at each
concurrency level on Qwen3.5-9B / pp2tp2 against the optimized
profile baseline. The environment variable name is a placeholder
(no actual var read in code); the harness still exercises every
booth/measurement path the way real env-var tests do, including
sample_vram=true.

Result curve answers:
  - Where does aggregate throughput saturate (does N=8 plateau or grow)?
  - How does TTFT p99 evolve with N (head-of-line blocking?)
  - VRAM scaling: linear in N? plateaus once KV cache + scratch are
    sized at INFLIGHT_SLOTS=8?
