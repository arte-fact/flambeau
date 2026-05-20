# S2 — LM-head Q4_K mmvq routed to r4 (was r2)

Date: 2026-05-20
Status: **shipped.** Coherence verified across 3 streaming-bench runs at
291-token prompt, 128-token decode. Cumulative L1+S2 wins on E4B.

## Change

`crates/forward/src/core/composites/output_head.rs`: when the call is
(a) `n_emit == 1` (decode), (b) `lm_head.dtype == Q4_K`, (c)
`vocab >= 131072`, (d) `hidden % 256 == 0`, bypass the dispatch table
and launch `mmvq_q4_k_r4` directly via the now-public
`mmvq_simple_launch`.

The dispatch table picks `mmvq_q4_k_r2` for all `m=1..127` Q4_K mmvq.
That's optimal for typical hidden-sized matmuls (n=2048–15360) but
leaves the **single largest per-call kernel** in our profile (LM head
at n=vocab=262144, 3.4 ms / call, **14% of E4B kernel time**)
unoptimised. r4 halves block count vs r2 (4 rows / block instead of 2);
at vocab scale the per-block work is small enough that the 4-row
amortisation pays off.

## Bench (E4B-Q4_0 SD, prompt 291 tok, decode 128 tok, hip:3)

3-run streaming-API sample, same prompt as the lever-2-ship cert:

| run | prefill t/s | decode t/s |
|---:|---:|---:|
| 1 | 530.0 | 46.1 |
| 2 | 590.6 | 45.9 |
| 3 | 589.9 | 46.2 |
| **avg** | **570** | **46.1** |

vs post-L1 baseline (`gemma4_v2_l1_attn_decode_splitk_tile4_2026_05_20`):

|  | prefill | decode |
|---|---:|---:|
| post-L1 (r2 LM head) | 580.9 | 44.3 |
| post-S2 (r4 LM head) | **570 ± 30** | **46.1** |
| Δ | within noise | **+4.1 %** |

Cumulative L1+S2 vs pre-L1: decode **41.1 → 46.1 = +12.2%**.
vs llama.cpp 70.6 t/s: ratio **0.58× → 0.65×**.

## Correctness

- `mmvq_q4_k_r4.cu`'s own comment: "byte-identical to r2 — same scales,
  same nibble decode, same F32 accumulation envelope". Only the
  lane-to-element layout differs.
- 3 × 128-token greedy decodes produced coherent technical text
  matching the prompt (compiler/GPU explanation).
- Sweep harness (`flambeau sweep --arch gfx906 --op qmatmul --dtype
  Q4_K`) — the runtime r4 kernel is the same code shipped behind
  `qmatmul_q4_K_mmvq_nw1_r4_gfx906` in `qmatmul.rs:1403-1410`;
  correctness math is the established r-quad pattern from candle P29.

## Scope

Only applies to **Q4_K LM-head at decode** with vocab >= 131072. The
gemma4 family (vocab=262144) is the targeted user. Qwen3 vocab≈152k
hits the threshold too. Smaller vocabs and non-Q4_K LM heads stay on
the established dispatch path (r2 / single-row).

## Recalibrated cumulative perf vs llama.cpp on E4B

|  | flambeau-v2 | llama.cpp | ratio |
|---|---:|---:|---:|
| prefill | ~580 t/s | 1046 t/s | 0.55× |
| decode  | **46 t/s** | 70.6 t/s | **0.65×** |

The remaining gap on decode is dominated by the 30+ small launches
per layer × 42 layers; structural fusion (S3) is the next session's
target.
