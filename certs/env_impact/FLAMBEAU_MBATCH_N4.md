# Env-impact cert: `FLAMBEAU_MBATCH`

- **Class:** B
- **Default state:** `off`
- **Generated:** 2026-05-06T07:54:34+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** MoE batched dispatch gate

**Preconditions:** `FLAMBEAU_INFLIGHT_SLOTS=4`

## Measured cells

### qwen36-35b-a3b-q4_0 / pp2tp2 / N=4

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 11044 | 2176 | 115.89 | — | — |  |
| `1` | 3 | 11050 | 2176 | 113.88 | -1.7% | DIVERGENT |  |

### qwen36-35b-a3b-q4_0 / tp2 / N=4

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 11853 | 2637 | 92.93 | — | — |  |
| `1` | 3 | 11901 | 2687 | 93.30 | +0.4% | match |  |

## Triage

- qwen36-35b-a3b-q4_0/pp2tp2 [N=4] `1` → **divergent** (-1.7%)
- qwen36-35b-a3b-q4_0/tp2 [N=4] `1` → **null** (+0.4%)

## Disposition

- **HALT-DIVERGENT** — 1/2 cells produce different output at greedy/fixed-seed; correctness bug, not a perf gate — file before migration

## Notes from spec

Read in ops/moe.rs:1251. MoE expert batching only kicks in with concurrent slots.
