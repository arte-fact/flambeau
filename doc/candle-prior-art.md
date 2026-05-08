# Candle / llama.cpp / llamacpp-turbo — Prior-Art Map

Every V1 kernel below is a port, not a new implementation. This table tells an agent exactly which file to open when they claim a V1 step. The port target is **the fast known variant**, not a starting point to iterate on.

**Roots (sibling dirs under `/artefact/`):**

- `candle/` — `/artefact/candle/` — primary source. All candle citations are paths relative to this root.
- `turbo/` — `/artefact/llamacpp-turbo/llama-cpp-gfx906-turbo/` — llamacpp fork tuned for gfx906. Holds the 4-warp LDS-tiled MMQ + stream-K + L2-prefetch kernels.
- `llamacpp/` — `/artefact/llama.cpp/` — upstream. Correctness oracle + fallback reference for anything not in candle or turbo.

## V1.1 — GGUF loader + CPU dequant reference

| Need | Source | Notes |
|---|---|---|
| GGUF v3 file format | `candle/candle-core/src/quantized/gguf_file.rs` | Header parse, tensor enumeration, metadata kv-store. Port wholesale; candle's parser is production-quality. |
| GGUF block layouts | `candle/candle-core/src/quantized/k_quants.rs` + `candle/candle-core/src/quantized/*.rs` | Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K superblock structs; Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 legacy blocks. |
| CPU dequant reference | `llamacpp/ggml/src/ggml-cpu/ggml-cpu-quants.c` | Bit-exact target for round-trip. Compare byte-for-byte on random tensors. |
| Candle's dequant port | `candle/candle-core/src/quantized/k_quants.rs::Dequant::dequantize_row` | Rust port we can crib; verify against llamacpp. |

## V1.2 — HIP device + RCCL collectives

| Need | Source | Notes |
|---|---|---|
| `HipDevice`, `HipStream`, allocation | `candle/candle-core/src/hip_backend/device.rs` | Drop-order fix lives here (`Mutex<Option<RocBlas>>` for destructor). |
| Multi-device cluster scaffolding | `candle/candle-core/src/hip_backend/cluster.rs` | Rank driver, barrier, per-rank state. |
| RCCL collective bindings | `candle/candle-hip-kernels/src/` + search for `rccl` | Candle uses RCCL for all-reduce / all-gather / all-to-all. |
| Host-bounce fallback | `candle/candle-core/src/hip_backend/mod.rs` search `all_reduce_sum_f32_via_host` | Reference-only; cert harness uses this to validate RCCL. |

## V1.3 — First-class MMVQ (Q4_K / Q5_K / Q6_K / Q8_0)

| Need | Source | Notes |
|---|---|---|
| Q4_K / Q5_K / Q6_K MMVQ `*_nw1_r{2,4}` (multi-row DPP reduce) | `candle/candle-hip-kernels/src/quantized.cu` | Search `indexed_moe_forward_q4k_q8_1_nw1`, lines ~5700-6461. The MoE variant is what we port; the dense path uses the same body. Multi-row DPP is candle P29. |
| Q8_0 MMVQ | `candle/candle-hip-kernels/src/quantized.cu` | Search `mul_mat_vec_q8_0_q8_1_cuda{1..8}` — several template variants; cuda1 (256-thread) is default since P34. |
| Quantize activation to Q8_1 | `candle/candle-hip-kernels/src/quantize_kv_q8.cu` + `quantized.cu::quantize_row_q8_1` | Required partner to every K-quant MMVQ; take the fused variant. |
| DPP half/quarter-warp reduce | `candle/candle-hip-kernels/src/gfx906_primitives.cuh` | `gfx906_half_warp_reduce_sum_dpp`, `gfx906_quarter_warp_reduce_sum_dpp`. Lifts into `kernels-hip/arch_primitives/gfx906.cuh`. |

## V1.4 — First-class MMQ prefill (4-warp LDS-tiled)

| Need | Source | Notes |
|---|---|---|
| 4-warp LDS-tiled MMQ (Q4_K / Q6_K / Q8_0) | `turbo/ggml/src/ggml-cuda/mmq.cu` + `mmq.cuh` | **Primary port target.** Contains stream-K fixup + L2 prefetch + vectorised X/Y tile load. Much of this is in `mmq.cuh` template machinery. |
| Candle's port attempt (single-warp, incomplete) | `candle/candle-hip-kernels/src/mmq_turbo.cu` | Candle landed `mul_mat_q4_K_turbo_dense` + `q6_K_turbo_dense` (P37/P38) as single-warp — do NOT port this, it's the variant flambeau V1 explicitly skips. Read for the calling-convention + mmq_x tuning lessons only. |
| mmq_x selection heuristic | `candle/candle-core/src/quantized/hip.rs` near line 741-832 | K-quants want `mmq_x=8` at VGPR≈99. Port the heuristic. |
| llama.cpp upstream MMQ (reference) | `llamacpp/ggml/src/ggml-cuda/mmq.cu` | Source of turbo's port. Use for correctness comparison on edge shapes. |

## V1.5 — MoE: TopK router + indexed expert matmul + combine

| Need | Source | Notes |
|---|---|---|
| Indexed MoE MMVQ Q4_K/Q5_K/Q6_K with multi-row DPP | `candle/candle-hip-kernels/src/quantized.cu::indexed_moe_forward_q{4,5,6}k_q8_1_nw1_r{2,4}` | P29 pattern. Same body as V1.3 but with id/bounds indirection. |
| Fused gate+up MoE nw1_r2 | `candle/candle-hip-kernels/src/quantized.cu::indexed_moe_forward_*_gate_up_nw1_r2` | P30 — halves launch count on MoE decode. Q4_K/Q5_K only; Q8_0 stays r1. |
| Turbo MoE MMQ (Q4_K prefill) | `candle/candle-hip-kernels/src/mmq_turbo.cu::mul_mat_q4_K_turbo_moe_x*` | Port target: apply the 4-warp LDS-tiled treatment from V1.4 to this MoE version. |
| TopK on router logits | No direct candle analogue | New kernel. 128 F32 → top-8 per token. Start from a classical on-chip bitonic top-k pattern. |
| MoE combine (weighted sum + residual) | `candle/candle-hip-kernels/src/topk_moe.cu` | Search for `moe_combine` — candle C1 fused combine + residual. |

## V1.6 — Fused decode + attention + KV cache

| Need | Source | Notes |
|---|---|---|
| RMSNorm + Q8_1 fused | `candle/candle-hip-kernels/src/reduce.cu::rmsnorm_q8_fused` | Candle D1. One kernel replaces rmsnorm + quantize_q8_1 pair. |
| RMSNorm F16 (non-fused) | `candle/candle-hip-kernels/src/reduce.cu` search `rmsnorm_f16` | Fallback when the next op isn't a K-quant matmul. |
| RoPE (Q + K one kernel, interleaved pair) | `candle/candle-hip-kernels/src/` search `rope` | Gemma-4 uses the same layout Qwen3 does; rope_theta from GGUF metadata. |
| Masked softmax + scale fused | `candle/candle-hip-kernels/src/reduce.cu::masked_softmax_scale_fused` | Candle F5. Decomposed fallback when `n_rows < 24`. |
| SwiGLU (silu(gate) * up) | `candle/candle-hip-kernels/src/fused_pointwise.cu` | Fused silu + mul; fused with dispatch of up projection is V2. |
| Flash-attention v2 K-transposed (prefill) | `candle/candle-hip-kernels/src/flash_attn_v2.cu` | K pre-transposed at cache insert; avoids per-attention transpose cost. |
| GQA decode attention F16 KV | `candle/candle-hip-kernels/src/flash_attn_v2.cu::gqa_decode_mv_fast_d{128,256,512}` | Gemma-4 fused kernel; Qwen3.6 uses D=128. |
| GQA decode attention Q8 KV | `candle/candle-hip-kernels/src/attn_q8_kv.cu::gqa_attention_decode_q8` | Candle Q8 KV path. Pair with fused softmax for F3. |
| Fused-softmax Q8-attention decode | `candle/candle-hip-kernels/src/attn_q8_kv.cu` + `reduce.cu` | F3 — caller-controlled fusion. gfx906-shaped. |
| K-transposed KV cache | `candle/candle-core/src/hip_backend/kv_cache.rs` search `k_transposed` | Type-state flag in flambeau (`F16Transposed`). |
| Q8 KV cache storage | `candle/candle-core/src/hip_backend/q8_kv_cache.rs` | Full Q8 KV scheme. Our `Q8Contig` / `Q8Transposed` layouts port from here. |

## V1.7 — Model composition (Qwen3.6 MoE)

| Need | Source | Notes |
|---|---|---|
| Qwen3 MoE block list | `candle/candle-transformers/src/models/quantized_qwen35_moe.rs` | Reference implementation — take the block composition, drop the non-inference paths. |
| GGUF weight-name mapping | Same file, search for the weight dict | Qwen3.x names — e.g. `blk.N.attn_q.weight`, `blk.N.ffn_gate_exps.weight`. |
| Per-rank tensor-range loader (sharded load) | `candle/candle-transformers/src/models/quantized_blocks/gguf_loader.rs` search `load_ep` / `load_tp` | X5 pattern — each rank reads only its shard from mmap'd GGUF. |
| Mesh<N>-generic attention + MLP + MoE | `candle/candle-transformers/src/models/quantized_blocks/{attention,ffn,delta_net}.rs` | Candle X8 landing — reference for TP/EP wiring, **not** the shape of flambeau ops (we own the mesh-generic redesign). |

## V1.8 — Tokenizer, sampler, chat template, server

| Need | Source | Notes |
|---|---|---|
| Qwen3 chat template | Embedded in GGUF — `tokenizer.chat_template` metadata field | Apply via `minijinja`. Byte-for-byte match against `llamacpp/common/chat.cpp::common_chat_apply_template`. |
| Tokenizer | Embedded in GGUF — `tokenizer.ggml.*` fields | Feed to the `tokenizers` crate. |
| Sampler (temperature + top-p) | `candle/candle-examples/examples/quantized-qwen35/main.rs` search `LogitsProcessor` | CPU-side on the last logit row; not perf-critical. |
| OpenAI HTTP surface | `llamacpp/tools/server/server.cpp` | Reference for request/response JSON shapes, SSE framing, chat-completions vs completions distinction. Axum port. |

## Meta — when candle prior art is wrong

Some candle code paths are landing-steppingstones we explicitly **don't** port:

- **Single-warp MMQ (Q4_K/Q6_K turbo_dense)** — superseded by turbo's 4-warp LDS-tiled; V1 skips straight to 4-warp.
- **`CANDLE_*` env-flag gating** — replaced by `dispatch/<backend>/<arch>.toml`.
- **MMVQ `wc` (64-thread warp-coop) for K-quants** — superseded by cuda1 template (P34) at small M.
- **`alloc_zeros` in MMQ outputs** — start with `alloc`, write every output lane (P24 lesson).
- **Q5_K v3 opt-in flag** — null perf, delete on port.
- **dp4a fused Q8 attention (`gqa_decode_mv_fast_q8_dp4a_*`)** — P10 null; do not wire.

Refer to `/home/sandbox/.claude/projects/-artefact-candle/memory/MEMORY.md` for the full null-result log. A kernel described as "null" or "reverted" in candle's memory is a kernel **not** to port.
