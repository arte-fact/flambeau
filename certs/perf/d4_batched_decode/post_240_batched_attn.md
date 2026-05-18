# #240 — v2 standard_attn batched-decode kernel (2026-05-18)

## What landed

The multi-slot / non-prefill branch of
`crates/forward/src/core/composites/standard_attn.rs` no longer
loops per-slot calling `kv_append_f16` + `attn_decode_f16` N times.
It now:

1. Builds host `[N] u64` K/V cache base-ptr arrays + `[N] i32`
   write-position + KV-length arrays.
2. Uploads them to new ScratchPool slots
   `attn_slot_{k_dst_ptrs,v_dst_ptrs,write_pos,n_kv}` (allocated when
   `max_slots > 1`).
3. Issues **one** `kv_append_f16_batched_slots` launch (one block per
   slot, all KV rows written simultaneously).
4. Issues **one** `attn_decode_f16_batched` launch
   (`grid=(n_heads_q, n_slots)`, all slots' attention output computed
   in one launch).

Companion ops added under `crates/model-ops/src/ops/attn_decode_batched.rs`:
`attn_decode_f16_batched` + `kv_append_f16_batched_slots`. Re-exported
from the model-ops root.

## Trace confirmation (Qwen3.6-35B-A3B-Q4_0 / PP2 / N=2 concurrent)

```
attention_decode_f16_batched        40 calls
kv_append_f16_batched_slots         40 calls
kv_append_f16                        0 calls    ← was N×40 before
```

The scalar per-slot kernels are gone; only the batched variants
remain.

## Throughput delta

Qwen3.5-9B-Q4_1 / PP2 / hip:0,2 / K=64 / runs=3 best-of:

|   N | pre-#240 agg | post-#240 agg | delta |
|----:|-------------:|--------------:|------:|
|   1 |  30.97 t/s   |  31.01 t/s    | +0.1% |
|   2 |  29.13 t/s   |  29.51 t/s    | +1.3% |
|   4 |  28.55 t/s   |  29.37 t/s    | +2.9% |

Qwen3.6-35B-A3B-Q4_0 / PP2 / hip:0,2 / K=24:

|   N | pre-#240 agg | post-#240 agg | delta |
|----:|-------------:|--------------:|------:|
|   1 |  19.80 t/s   |  19.83 t/s    |  0%   |
|   2 |  19.28 t/s   |  19.57 t/s    | +1.5% |
|   4 |  20.23 t/s   |  20.02 t/s    | -1%   |

## Why the lift is small

The batched-attn kernel correctly fires and saves N-1 launches per
layer, but attention is a small fraction of total per-layer GPU time
on these hybrid models. Profile post-#240 (Qwen3.6-35B-A3B, N=2):

| kernel                                              |  ms  |   %  |
|-----------------------------------------------------|-----:|-----:|
| `flambeau_indexed_moe_mmvq_q4_0_q8_1`               | 285  | 16%  |
| `flambeau_mmvq_q4_0_gate_up_dp4a_q8_1` (GDN gate+up)| 193  | 11%  |
| `flambeau_mmvq_q5_0_q8_1` (GDN)                     | 161  |  9%  |
| `flambeau_indexed_moe_mmvq_q4_0_gate_up`            | 147  |  8%  |
| **`flambeau_attention_decode_f16_batched`**         |  *   |  ~3% |
| `flambeau_gdn_state_step_alphabeta_f32_s128`        |  73  |  4%  |
| ... per-slot GDN MMVQ / state-step still dominates  |      |      |

The remaining N>1 ceiling is in:
- **GDN composite** (per-slot mmvq + state-step + pass-A loop) → #241.
- **MoE composite** at N>1: the `forward_decode_tp_f32` block is
  called N times sequentially via `moe_ffn_loop`. Dispatching N
  tokens at once via `forward_prefill_tp_f32` (where the
  `indexed_moe_mmq_*_tile8` kernels fire at n_pairs ≥ 8) → #242.

## Architectural value of #240

This is the right kernel for v2 to use regardless of perf delta. The
prior per-slot loop:
- Issued N×2 launches per attention layer at N>1 → 100+ extra
  launches per token on a 32-layer model.
- Re-uploaded `n_tokens_kv` host→device per slot (via the
  `kv_append_f16` scalar position path).
- Could not amortize the K/V HBM reads across the N Q rows (each
  per-slot call re-streamed the full slot K/V history).

The batched kernel collapses all three. The HBM-amortization
benefit is real for slots that share a context-length bucket and
becomes visible once attention is no longer eclipsed by GDN / MoE
per-slot loops (#241 + #242).

## Reproduce

```
scripts/profile/decode_step_trace_n2.sh v2 /tmp/v2_n2
python3 scripts/profile/summarize_kernel_trace.py \\
    /tmp/v2_n2/threadreaper/*kernel_trace.csv --top 15
```
