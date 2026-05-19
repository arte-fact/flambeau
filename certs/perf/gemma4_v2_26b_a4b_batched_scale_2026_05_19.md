# gemma4-v2 26B-A4B MoE — batched `apply_per_expert_scale_f32` + bench vs llama.cpp

Date: 2026-05-19
Model: `gemma-4-26B-A4B-it-Q8_0.gguf`
Mesh: flambeau PP2 hip:0,1; llama.cpp PP2 hip:0,1 (`--split-mode layer`)
Prompt: 725-tok technical prompt (`PROMPT_BASE`)
Decode: 64 tokens, greedy temp=0
Dims: hidden=2816, head_count=16, head_kv=8, expert_count=128,
experts_per_tok=8, expert_ffn=704, 30 layers

## Change

`crates/kernels-hip/src/kernels/apply_per_expert_scale_f32.cu` now uses
`gridDim.x = n_tokens` with `expert_weights[t * top_k + k]` indexing.
Rust binding + Ops trait gained `n_tokens` parameter; both gemma4-v2
MoE batched prefill cascade and legacy gemma4 (`models/gemma4/src/moe.rs`,
`tp_moe_upload.rs`) call with `n_tokens=1` for decode and `n_tokens=N`
for the prefill cascade.

Before: prefill issued one launch per token per layer
(`for t in 0..n_tokens { apply_per_expert_scale_f32 }`) — ~725 × 26 ≈
18.8k launches per prefill of a 26-MoE-layer model.
After: one launch per layer — 26 launches.

## Results (2 runs)

| stack         | prefill t/s   | decode t/s    | notes                              |
|---------------|--------------:|--------------:|------------------------------------|
| flambeau-v2 (1) | 506.1        | 43.21         | run 1                              |
| flambeau-v2 (2) | 551.9        | 43.28         | run 2                              |
| **v2 mean**   | **~529**      | **~43.2**     | ±5 % run-to-run variance           |
| flambeau-legacy | ERR           | ERR           | "MoE prefill not supported"        |
| llama.cpp     | 490.6 / 490.1 | 66.30 / 65.87 | stable across runs                 |

Ratios (v2 mean / llama.cpp):

- prefill: **1.08×** (v2 ahead)
- decode:  **0.65×** (v2 behind by ~35 %)

## Reading

Prefill is competitive. v2 beats llama.cpp on this MoE prefill shape;
the per-token-loop → batched-kernel collapse removed ~18.8k tiny
launches per prefill, which matters at PP2 where each launch is the
critical path on its rank for that micro-step. Run-to-run noise is
~5 %, but both runs cleared llama.cpp.

**Decode is the open gap.** 0.65× is the same shape as gemma4-31B-Q4_0
on TP2 (0.87×) but worse — MoE decode hits 8-expert per-token MMVQ
dispatch which on v2 is the recently-shipped `indexed_moe_mmvq_*`
batched-decode path. Likely suspects, in priority order:

1. **MoE expert dispatch overhead.** v2 batches expert dispatch across
   the 8 selected experts but each is still a per-token-per-expert
   indexed MMVQ; llama.cpp's MoE decode kernel is a single fused launch
   per layer.
2. **Full attention every layer.** gemma4-26B-A4B is dense-attn + MoE,
   so the standard attn decode path runs every layer (28 layers); this
   is the same path as 31B-Q4_0 where we measured 0.87× — the gap is
   wider here likely because the MoE overhead piles on top.
3. **Per-expert scale fold.** The 1-launch-per-layer `apply_per_expert_scale`
   is now negligible at N=1 — single thread block, 8 threads. No
   further win here at decode.

## Next levers

- Profile decode-only with rocprofv3 hip-trace (running flambeau-v2
  decode-only on a hot context) to attribute the 35 % gap to specific
  kernels. Prior session noted SIGSEGV in `HipDevice::new` under
  interposer; may need attached-mode (`rocprofv3-attach`) to a
  pre-booted server.
- Lever-1 F16-direct mmvq (#251) is on for v2 decode; the MoE indexed
  variants may not be picking it up. Audit dispatch table for
  `indexed_moe_mmvq_q8_0_f16dst` parity.
- Lever-3 event-gated AR (#253) is on; size-gated dispatch should keep
  the small-n AR on the event path. Verify `n_elems` at decode is
  ≤ 65_536 for the 26B MoE (likely yes, hidden_dim=2304 × 1 tok).
