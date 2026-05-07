# Q8 KV cache: why our decode kernel hit ≈ 1.0× F16, and how to get to 1.5×

**Date:** 2026-05-07
**Author:** Phase-3 follow-up investigation
**Audience:** anyone touching `attention_decode_q8_kv*.cu` or `attention_prefill_q8_kv.cu`

## TL;DR

Our Q8_0 KV cache decode kernel reads ~half the HBM bytes vs F16 but
runs at ~1× F16 wall on Qwen3.6-27B / pp2tp2 (post-batched-prefill,
post-split-K). Theory predicts 1.5–2×.

Root cause: we **dequantize K to FP16 in the score loop**, one element
per thread. That makes per-element work ~4 fp16 ALU cycles
(int8→int→fp32 cast + scale-multiply + Q-multiply + accumulate),
which on gfx906 (1 TB/s HBM, 256 fp16 ops/clock/SIMD) ties the HBM
cycle budget. The kernel is **compute-bound on dequant arithmetic**,
not bandwidth-bound. Halving the bytes saves nothing.

Every fast engine — llama.cpp, vLLM (FP8), TensorRT-LLM, SGLang,
INT-FlashAttention, SageAttention — solves this the same way:
**don't dequantize K to FP16 inside the score loop.** Keep the score
matmul in INT8 (`dp4a` / `IMMA.s8.s8.s32`), apply the scalar `K.d × Q.d`
once per 32-element block at the *end* of the dot product.

## What our current kernel does (the slow path)

`crates/kernels-hip/src/kernels/attention_decode_q8_kv*.cu` core loop:

```cuda
const flambeau_block_q8_0* k_block = k_cache + kv_row_blocks + block_idx;
const float k_d = (float) k_block->d;          // FP16 scale → FP32 cast
const int   k_q = (int)   k_block->qs[block_offset];  // int8 → int
my_partial = q_shared[tid] * (k_d * (float) k_q);     // 2 FP32 multiplies + 1 add
```

Per int8 byte read from K:
- 1 cast int8→int
- 1 cast int→fp32
- 1 fp16→fp32 cast on `k_d` (broadcast across 32 lanes)
- 2 fp32 multiplies, 1 fp32 add (the partial-sum)

That's ~5 ALU ops per byte. On gfx906 each SIMD has 64-wide wave at
~1.7 GHz fp32 issue rate, so 5 ops × 64 lanes ≈ 320 ops/clock
amortized. At HBM rate (1 TB/s ÷ 64 lanes / 1.7 GHz ≈ 9.2 bytes/clock),
the kernel can sustain ~9 byte-reads per clock — and needs ~5 cycles
of ALU per byte to consume them. Compute is the long pole.

## What llama.cpp / INT-FlashAttention do (the fast path)

llama.cpp's `fattn-common.cuh:264-289` (`vec_dot_fattn_vec_KQ_q8_0`):

```cuda
// 1. Q already quantized to Q8_1 in LDS once per FA tile
//    (block-amax reduce + roundf(v/d)) — see quantize_q8_1_to_shared.

// 2. Score-path inner loop: pure integer dp4a.
int sumi = 0;
#pragma unroll
for (int k = 0; k < QI8_0/4; ++k) {
    const int K4 = *((const int*)&K_q8_0[ib].qs[4*k]);     // 4 int8 K-bytes
    const int Q4 = *((const int*)&Q_shared.qs[4*k]);        // 4 int8 Q-bytes
    sumi = __dp4a(K4, Q4, sumi);                            // 4×int8 → int32 MAC
}

// 3. Scalar dequant ONCE at the end of the 32-element block:
const float dKQ = K_q8_0[ib].d * Q_d;
sum += dKQ * (float)sumi;
```

Per 32 int8 bytes of K read:
- 8× `__dp4a` calls (each consumes 4 int8 K + 4 int8 Q → int32)
- 1× FMA at block boundary (`K.d × Q.d × sumi`)

That's ~9 ALU ops per **32 bytes**, vs our 160 ops per 32 bytes
(5 ops × 32 elements). **~17× lower compute density.** The kernel
flips from compute-bound back to HBM-bound, where halving K bytes
finally pays off.

`__dp4a` on gfx906 maps to `v_dot4_i32_i8` — a single-cycle dot of
4×int8 producing int32. No tensor cores needed. INT-FlashAttention
(arxiv 2409.16997) generalizes this to all of QK and PV on Ampere
GPUs without FP8 hardware (closest analogue to MI50) and reports
**+72% over FP16 FlashAttention.**

## Engine-by-engine speedups (decode, single request)

| Engine | Format | Score-path | Decode speedup vs F16/BF16 | HW |
|---|---|---|---:|---|
| **flambeau (today)** | Q8_0 | FP16 dequant per element | ~1.00× | gfx906 |
| llama.cpp Q8_0 KV | Q8_0 (block) | INT8 dp4a, scalar K.d×Q.d at block end | 0.85–1.00× | CUDA + HIP |
| INT-FlashAttention | per-token INT8 | INT8 IMMA / dp4a end-to-end | **+72%** | Ampere |
| SageAttention | per-block INT8 | INT8 IMMA, INT8 PV | ~2× | Ada/Ampere |
| TensorRT-LLM INT8 KV | per-tensor/channel INT8 | INT8 IMMA + plugin | **1.45×** | Ampere/Hopper |
| vLLM FP8 KV (FA3) | per-tensor FP8 | FP8 IMMA on Hopper tensor cores | **1.85×** ITL | Hopper SM90 |
| SGLang/FlashInfer FP8 | per-tensor FP8 | FA3 FP8 path | ~2× microbench | Hopper |

llama.cpp's "neutral" number deserves a note: it stays neutral because
**Q8 KV's V path still dequants tile-wise to FP16** (only the K path
goes pure-int via dp4a). The K wins are real (~1.5×) but get
amortized against an unchanged V path. Production bias is *bytes
saved* (KV-cache fits in VRAM at 2× longer context) more than wall.

## Why FP8 wins more than INT8 on Hopper

Both vLLM and TensorRT-LLM measure FP8 > INT8 even at identical byte
count. Cited reason: *"lower dequantization overhead of FP8."*
Hopper's `IMMA.fp8.fp8.fp32` produces FP32 accumulators natively;
INT8 IMMA produces INT32 that needs cast + scale-multiply. On
gfx906 we have neither tensor-core path, so the comparison doesn't
apply — the dp4a route is what's available.

## Concrete fix for `attention_decode_q8_kv*.cu`

Two-phase rewrite. Phase 1 covers the K side (≥ 1.5× of attention
wall on decode); Phase 2 covers V (the +72% number).

### Phase 1 — INT8 score loop (port the llama.cpp Q8_0/Q8_1 + dp4a pattern)

1. **Quantize Q to Q8_1 in LDS once per kernel launch.**
   `Q[q_head, :]` → `int8 q_qs[head_dim] + (d, sum)` per QI8_1=32
   group. Cost: one block-amax reduction (`__shfl_xor` over the
   warp) + `roundf(val / d)`. Reference:
   `quantize_q8_1_to_shared` in
   `/artefact/llama.cpp/ggml/src/ggml-cuda/fattn-common.cuh:292`.
   Storage: `head_dim` int8 + `(head_dim/32) × (FP16 d, FP16 s)` =
   ~1.06 × head_dim bytes in LDS. Trivial.

2. **Replace the score loop with `__builtin_amdgcn_sdot4`.**
   gfx906 ISA equivalent of `__dp4a`. Each call:
   `sumi = sdot4(K_int32_packed, Q_int32_packed, sumi)` — consumes
   4 int8 K + 4 int8 Q, produces int32 partial sum. Loop unrolled
   `head_dim / 4` times per (q_head, t) pair. The `Q.s × K_sum_of_qs`
   term that Q8_1 needs falls out of the same loop (cached in `Q.s`).

3. **Apply `K.d × Q.d` once per QK8_0 block (32 elements), not per
   element.** One FMA at the end of the 4-call group: `score +=
   K_d * Q_d * (float)sumi + offset_correction`. Reference:
   `fattn-common.cuh:285`.

4. **Leave V on tile-dequant to FP16** for Phase 1. Matches
   llama.cpp. The post-softmax weight is FP16 already, so V dequant
   to FP16-tile in registers is free relative to the score loop's
   former cost.

**Expected outcome on gfx906:** score-loop compute drops ~17×,
making the score path HBM-bound for the first time. K-side bytes
halve → ~1.5–1.8× speedup on the K portion. V-side unchanged.
Weighted ~1.4× overall on decode-attention wall (K and V are
read in equal volume, so the average of 1.6× and 1.0× ≈ 1.3×).

### Phase 2 — INT8 V path (INT-FlashAttention extension)

After Phase 1 lands, the next +30-40% on attention wall comes from
also keeping the PV matmul in INT8. Quantize the post-softmax
attention weights `KQ_softmax_k` to Q8_1 per row (same per-row amax
trick), run `PV[head_dim] += dp4a(P_int8, V_int8, …)` block-wise,
multiply by `P.d × V.d` at the end of each V block. Cumulative
target: ~1.7× over F16 on the full attention kernel.

This is what INT-FlashAttention (arxiv 2409.16997) does end-to-end;
their reported **+72%** over FP16 FlashAttention on Ampere is the
2-matmuls-pure-INT8 number, scaled for MI50's lack of tensor cores
(MI50 dp4a has the same 4×int8 → int32 throughput as Ampere's
INT8 fast path on consumer cards).

## What NOT to do

- **FP8 E4M3/E5M2 on gfx906** — no hardware support. The vLLM/SGLang
  numbers are tensor-core FP8 IMMA on SM90; gfx906 would have to
  emulate it through fp16 anyway.
- **Per-tensor INT8** (one global scale instead of per-block) — every
  paper that compared finds this strictly worse on accuracy and
  identical on perf. Q8_0's per-32-element block is the right tradeoff.
- **Pre-dequant K to FP16 into LDS** — it's the same compute cost
  paid in shared memory instead of registers. No win.
- **Only port the K side and ship** — without Phase 2 the win is
  bounded by Amdahl. Plan multi-session.

## File pointers

- `crates/kernels-hip/src/kernels/attention_decode_q8_kv.cu` —
  current single-pass (will get the dp4a rewrite).
- `crates/kernels-hip/src/kernels/attention_decode_q8_kv_splitk.cu`
  — current split-K (commit ff21e2d). Same dp4a rewrite applies to
  the chunk pass; combine pass is layout-agnostic and untouched.
- `crates/kernels-hip/src/kernels/attention_prefill_q8_kv.cu` —
  current oracle prefill (commit 9960e44). Same rewrite applies.
- `crates/kernels-hip/src/arch_primitives/gfx906.cuh` — drop
  `dp4a_int8_int32` wrapper here; per architectural-rule-5 keep
  intrinsics out of `kernels-shared/`.
- `/artefact/llama.cpp/ggml/src/ggml-cuda/fattn-vec.cuh` — port
  reference. Line 87 dispatches `vec_dot_fattn_vec_KQ_q8_0`; line 170
  triggers the LDS-quantize.
- `/artefact/llama.cpp/ggml/src/ggml-cuda/fattn-common.cuh` — port
  reference. Lines 264–289 are the dp4a score loop; line 292 is
  `quantize_q8_1_to_shared`; line 548 is the V dequant (Phase 1
  leaves this as-is).

## Citations

- llama.cpp Q8_0 KV vec FA: `ggml-cuda/fattn-vec.cuh`,
  `ggml-cuda/fattn-common.cuh` (in-tree at `/artefact/llama.cpp/`).
- llama.cpp PSA on symmetric KV + AMD HIP fused-FA path:
  https://github.com/ggml-org/llama.cpp/discussions/22411
- DGX Spark Nemotron 30B Q8 vs F16 KV bench:
  https://forums.developer.nvidia.com/t/365138
- vLLM FP8 KV blog:
  https://vllm.ai/blog/fp8-kvcache
- TensorRT-LLM INT8 vs FP8 KV (squeezebits A/B):
  https://blog.squeezebits.com/vllm-vs-tensorrtllm-8-kv-cache-quantization-35079
- INT-FlashAttention paper (arxiv 2409.16997):
  https://arxiv.org/html/2409.16997v1
- SageAttention (arxiv 2410.02367):
  https://arxiv.org/html/2410.02367v9
- FlashInfer FP8 attention:
  https://flashinfer.ai/2024/02/02/introduce-flashinfer.html
