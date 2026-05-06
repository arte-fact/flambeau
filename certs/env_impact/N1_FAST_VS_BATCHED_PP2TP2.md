# N=1 fast-path vs batched-hybrid forced — pp2tp2

- **Generated:** 2026-05-06T15:52:08+00:00
- **Topology:** pp2tp2 (0,2,1,3)
- **Slots:** 1 | **N:** 1 | **ctx_cap:** 4096 | **tg_len:** 64
- **Profile:** `optimized.toml` (5 baseline keys)
- **Runs/cell:** 3 measured + 2 warmup, greedy / temp=0 / seed=0

**Question.** Can we delete the legacy `decode_logits` fast-path and force `forward_decode_batched_hybrid` for all decode (including N=1)? Decision rule:

- batched_forced loss vs legacy ≤ ~3% on prod models (27B / 35B) → **collapse paths**: delete the fast-path branch
- batched_forced loss > 3% on a prod model → keep the gate or build an N=1 fast path **inside** the batched function

Side-benefit if collapsed: eliminates the cross-slot prefill-decode overlap as a race surface (leading suspect for #16 35B-A3B/pp2tp2 multi-slot divergence — slot 1's prefill currently overlaps slot 0's legacy decode).

## Measured cells

| model | path | n | prefill ms | decode ms | tg t/s | Δ vs legacy | correct? | VRAM peak GB | err |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| qwen35-9b-q4_1 | legacy_fast_path | 3 | 4196 | 1255 | 50.99 | — | — | 10.56 |  |
| qwen35-9b-q4_1 | batched_forced | 3 | 4220 | 1283 | 49.90 | -2.2% | match | 10.56 |  |
| qwen36-27b-q4_0 | legacy_fast_path | 3 | 11372 | 2732 | 23.43 | — | — | 20.92 |  |
| qwen36-27b-q4_0 | batched_forced | 3 | 11366 | 2733 | 23.42 | -0.0% | match | 20.92 |  |
| qwen36-35b-a3b-q4_0 | legacy_fast_path | 3 | 3902 | 1432 | 44.70 | — | — | 23.05 |  |
| qwen36-35b-a3b-q4_0 | batched_forced | 3 | 3894 | 1442 | 44.40 | -0.7% | match | 23.05 |  |

## Triage

- **qwen35-9b-q4_1** → acceptable (-2.2% within ±3%)
- **qwen36-27b-q4_0** → acceptable (-0.0% within ±3%)
- **qwen36-35b-a3b-q4_0** → acceptable (-0.7% within ±3%)

## Disposition

- **COLLAPSE-PATHS** — batched_forced is within ±3% on all measured models. Plan: delete `decode_logits` for hybrid, delete the routes.rs:734 fast-path branch, delete `FLAMBEAU_NO_FAST_PATH`.
