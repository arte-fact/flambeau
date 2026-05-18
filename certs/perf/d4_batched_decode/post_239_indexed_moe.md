# #239 — v2 moe_ffn adopts indexed_moe_* kernels (2026-05-18)

## Result

v2 N=1 throughput on Qwen3.6-35B-A3B-Q4_0 / PP2 / hip:0,2:

|              | legacy   | v2 pre-#239 | v2 post-#239 |
|--------------|---------:|------------:|-------------:|
| N=1 t/s      | 42.95    | 6.48        | **19.80**    |
| N=2 agg t/s  | 42.83    | 6.36        | 19.28        |
| N=4 agg t/s  | 43.84    | 6.50        | 20.23        |
| v2 ÷ legacy  | 1.00×    | 0.15×       | **0.46×**    |

**3.05× improvement at N=1**, closing about half the legacy gap from
6.6× to 2.2×. N>1 still hits the ~1× kernel ceiling (#240 / #241
remain the closing levers for batched-decode aggregate).

## Why

Before: v2 `moe_ffn_local` issued per-expert `mmvq_q4_0_q8_1` launches
in a host-side loop. For Qwen3.6 top-k=8 × 32 layers per token, that's
~256 launches/token plus per-expert swiglu / quantise / down ops.

After: the composite delegates to `flambeau_blocks::MoeExperts::
forward_decode_tp_f32`, which dispatches:
- `indexed_moe_mmvq_q4_0_gate_up_dp4a_q8_1` (1 call/layer, fused)
- `swiglu_f32_to_q8_1` (1 call/layer, fused activate+quantise)
- `indexed_moe_mmvq_q4_0_q8_1` (1 call/layer for down)
- `moe_combine_no_residual_f32` (1 call/layer)

## Profile delta (rocprofv3 --kernel-trace, K=4 + ~24-token prefill)

|                                      | legacy   | v2 pre   | v2 post  |
|--------------------------------------|---------:|---------:|---------:|
| Total GPU kernel time                |  382 ms  | 1719 ms  |  607 ms  |
| Total launches                       | 19 411   | 199 415  | 74 135   |
| `mmvq_q4_0_q8_1` calls               | 4 524    | 51 354   | 1 674    |
| `indexed_moe_*_q4_0_*` calls         | ~600     | 0        | **4 050**|

The dominant kernels in v2 post-#239 are:
- `flambeau_indexed_moe_mmvq_q4_0_q8_1`: 100 ms / 1890 calls (down)
- `flambeau_indexed_moe_mmvq_q4_0_gate_up_dp4a_q8_1`: 52 ms / 2160 calls
- `flambeau_mmvq_q4_0_gate_up_dp4a_q8_1`: 67 ms / 1620 calls (dense
  attention / GDN gate_up, not MoE)

Per-step CPU launch overhead also fell with the launch count (the
linear-N kernel-ceiling effect from D4 cert still applies at N>1).

## What landed

1. `ScratchConfig.max_experts_per_tok` field; flowed into ScratchPool
   per-slot allocations:
   - `moe_expert_ids: [top_k] I32`
   - `moe_expert_weights: [top_k] F32`
   - `moe_gate_out_f32 / moe_up_out_f32: [top_k * inter] F32`
   - `moe_activated_f16 / moe_activated_q8_1: [top_k * inter]`
   - `moe_down_f32: [top_k * hidden]`, `moe_down_f16` ditto.

2. `crates/forward/src/core/composites/moe_ffn.rs` decode path rewritten:
   - rmsnorm → quantise → router qmatmul (kept v2's path so quantised
     `ffn_gate_inp` weights still work — block's `route_decode`
     requires F16/F32 only).
   - Host top-k + softmax → upload `expert_ids` + `expert_weights` to
     device.
   - Build `MoeExperts` block from packed weights:
     `experts_gate[0].ptr` is the base of the contiguous
     stacked-experts tensor per `loader::upload_moe_experts_stacked`.
   - `block.forward_decode_tp_f32` → F32 partial in `pool.down_f32`.
   - `hooks.ar_sum_f32` (no-op SD/PP, real on TP/Hybrid).
   - Cast → `pool.delta`. Optional shared-expert add.

3. Both qwen35-v2 and gemma4-v2 dense arches set
   `max_experts_per_tok: 0` (no MoE).

## What didn't land yet

- **Prefill batched indexed-MoE** (`moe_ffn_loop` still iterates per
  token at decode shape). Per-token loop reuses the new path so it's
  no slower than before — but a fused
  `MoeExperts::forward_prefill_tp_f32` would deliver another
  multiplier on prefill MoE wall.
- **N>1 batched-decode aggregate** still at ~1× — #240 + #241 (batched
  attention + batched GDN) close that ceiling.
- **GPU-side topk** — v2 still does host top-k. Block's `route_decode`
  uses `topk_f32` device-side, but only for F16/F32 router weights.
  Closing this is a small additional win (~1µs saved per layer × 32
  layers = ~30µs per token).
- **Dense `flambeau_mmvq_q4_0_gate_up_dp4a_q8_1`** (67ms / 1620 calls
  in post-#239 trace) — this is attention/GDN, not MoE. #240 covers
  the batched-attention version.

## Reproduce

```
python3 scripts/bench/d4_batched_decode.py \\
    --model /artefact/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf \\
    --topology pp --devices 0,2 \\
    --n-max 4 --n-run 1,2,4 --tokens 24 --runs 2

scripts/profile/decode_step_trace.sh v2 /tmp/v2_trace
python3 scripts/profile/summarize_kernel_trace.py \\
    /tmp/v2_trace/threadreaper/*kernel_trace.csv --top 15
```
