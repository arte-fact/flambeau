# Env-impact cert: `FLAMBEAU_DENSE_GATE_UP`

- **Class:** B
- **Default state:** `fused`
- **Generated:** 2026-05-05T20:09:53+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** Fused gate+up projection in dense FFN layers

## Measured cells

### qwen35-9b-q4_1 / pp4

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 6016 | 1483 | 43.16 | — | — |  |
| `unfused` | 3 | 6007 | 1486 | 43.07 | -0.2% | match |  |

### qwen35-9b-q4_1 / tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 4073 | 1213 | 52.74 | — | — |  |
| `unfused` | 3 | 4105 | 1209 | 52.95 | +0.4% | match |  |

### qwen36-27b-q4_0 / pp4

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 19023 | 3817 | 16.77 | — | — |  |
| `unfused` | 3 | 19014 | 3837 | 16.68 | -0.5% | match |  |

### qwen36-27b-q4_0 / tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 11355 | 2986 | 21.43 | — | — |  |
| `unfused` | 3 | 11485 | 3088 | 20.73 | -3.3% | DIVERGENT |  |

## Triage

- qwen35-9b-q4_1/pp4 `unfused` → **null** (-0.2%)
- qwen35-9b-q4_1/tp2 `unfused` → **null** (+0.4%)
- qwen36-27b-q4_0/pp4 `unfused` → **null** (-0.5%)
- qwen36-27b-q4_0/tp2 `unfused` → **divergent** (-3.3%, output text mismatch)

## Disposition

- **HALT-DIVERGENT** — 1/4 cells (qwen36-27b-q4_0/tp2) produces different
  output text at greedy/temp=0/fixed-seed when the unfused dense gate+up
  path is selected. This is a correctness bug in the unfused code path on
  TP2, NOT a perf gate. Do not migrate or delete the gate before fixing
  the unfused-on-TP2 numerical divergence — file a tracking issue first.
  Other 3 cells null (within noise) so this is specifically a TP2 bug.

_(Disposition aggregate added post-hoc on 2026-05-05; pre-patch logic
treated DIVERGENT as a tps loss. Triage row above also re-classified
from `loss` → `divergent` to reflect the correctness signal.)_

## Notes from spec

Read in models/dense_ffn.rs:165. C8 fusion cert exists; this re-confirms
default-ON is still winning post-recent commits. Skip MoE-only models.
