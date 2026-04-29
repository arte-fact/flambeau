# MTP-4-C — BF16-throughout MTP forward: precision lever NULL

**Status:** clean null. BF16 path produces **bit-identical argmax
sequence** to the F16/Q8_1 path for the canonical 16-step Qwen3.6-27B
acceptance harness. 56.2 % → 56.2 %.

## Setup

- Base: `Qwen3.6-27B-Q4_0.gguf` (pp4, 4× MI50)
- MTP head: `Qwen3.6-27B-mtp.gguf` (F16 linears, F32 norms — original
  converter output)
- Prompt: `[760, 6511, 314, 9338, 369]` ("The capital of France is")
- Persistent KV cache (`FLAMBEAU_MTP_KV_ACCUM=1`, default since
  session 6)
- 16 decode steps

## Result

| dtype | accepted | rate  | per-step predictions |
|-------|----------|-------|----------------------|
| F16/Q8_1 (default) | 9/16 | 56.2 % | 248046, 760, 271, 332, 271, 760, 369, 4252, 13, 561, 57590, 369, 57590, 279, 6511, 321 |
| BF16-throughout    | 9/16 | 56.2 % | 248046, 760, 271, 332, 271, 760, 369, 4252, 13, 561, 57590, 369, 57590, 279, 6511, 321 |

Switched via `FLAMBEAU_MTP_BF16=1`; load path picks
`load_mtp_head_bf16` (host-side F16→BF16 cast at upload), forward path
picks `forward_mtp_step_bf16` (composes the BF16 ops shipped under
MTP-4-C-1..C-5).

## What this rules out

The session-7a `mtp_4_session7a_f32_residuals_null.md` post-hoc
hypothesised the dominant precision loss in MTP forward was the
**per-mmvq Q8_1 activation quantize** (8 sites × ~0.78 % per-block-of-32
noise, ~2.2 % cumulative). The proposed fix was BF16 throughout.

The full BF16 path eliminates Q8_1 entirely — and produces identical
argmax decisions. Therefore Q8_1 activation noise is **not** the
bottleneck. Eight sequential 0.78 % perturbations per layer don't shift
top-1 logits at the LM head, at least on this prompt.

## What's left

- F16/BF16 mantissa precision difference is also ruled out (BF16's
  7-bit mantissa < F16's 10 — if anything BF16 is *less* precise on
  weights, and we still got the same answer).
- F32 residual additions ruled out by MTP-4-A.
- KV cache accumulation already provided +12.4 pp (session 6).

Plausible remaining sources:

1. **Structural difference in the MTP head** vs vLLM/sglang reference.
   The session-5 audit verified math equivalence on paper but didn't
   activation-bisect; session-3's GemmaRMSNorm `+1` bake bug + session-6's
   KV accumulation gap show structural bugs are still the main
   class of issue here.
2. **Greedy-argmax brittleness at the 56 % point.** The base model's
   actual next token sometimes ties with another top candidate; MTP's
   draft can land on a different but logits-equivalent token and be
   counted "wrong". Sampling-aware verify (top-K rejection) might lift
   the apparent rate without changing the underlying signal.
3. **Theoretical ceiling on this prompt.** "The capital of France is
   Paris..." is short and the next-token distribution is highly
   peaked; the MTP head trained for a *distribution* may simply not
   match the base model's argmax on the tail tokens.

## Code state

All BF16 kernels green per cert-check (61 → 63 → 64 rows over the
five sub-tasks). Kept as load-bearing infrastructure:

- `cast_{f16,f32,bf16}_{f16,f32,bf16}` — 4 BF16 cast kernels
- `mmvq_bf16_bf16` — BF16 weight × BF16 act MMVQ (256 threads/row)
- `rmsnorm_bf16` — BF16 in/out, F16 weight
- `attention_decode_bf16` — GQA online-softmax, BF16 throughout
- `split_q_gate_bf16` / `sigmoid_mul_bf16` / `swiglu_f32_to_bf16`
  / `rope_neox_partial_bf16` — pointwise BF16 ops

Forward composition:
- `load_mtp_head_bf16` — host-side F16→BF16 weight upload
- `forward_mtp_step_bf16` — full BF16 chain
- `forward_mtp_step_with_lm_head` — env-flag-routed (`FLAMBEAU_MTP_BF16`)

## Decision

Mark MTP-4-C umbrella null. The BF16 kernels stay shipped (useful
for future BF16-native models / CDNA2+ silicon where BF16 has native
FMA hardware). MTP forward defaults to the F16/Q8_1 path.

**Next options:**

- **MTP-5 pragmatic ship.** Accept 56.2 % and integrate the spec-decode
  driver with K=2 (1.32× throughput win) or K=3 (1.45×) in the
  `forward_one_token_pp` loop. The 75 % strict gate becomes a stretch
  goal, not a blocker.
- **Activation bisect vs vLLM/sglang.** Run vLLM's MTP step on the
  exact same `[h_t, e_token, position]` tuple; compare per-op outputs
  to bisect where flambeau diverges. Heavy lift (different stack,
  different shard layout) — only worth it if MTP-5 is gated on the
  strict 75 %.

## Sources

- Session 7a (F32 residuals null): `mtp_4_session7a_f32_residuals_null.md`
- Session 6 (KV accumulation +12.4 pp): `mtp_4_session6_kv_accumulation.md`
- Session 5 (vLLM/sglang structural audit): `mtp_4_session5_vllm_sglang_compare.md`
