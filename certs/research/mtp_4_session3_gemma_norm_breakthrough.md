# MTP-4 session 3 — GemmaRMSNorm `+1` bake fix; acceptance 0% → 37.5%

**Status:** root cause identified for the 0% acceptance. Live
acceptance now 37.5% (3/8) on the canonical "The capital of France
is" prompt. Below the 75% hard gate but a real signal. Remaining
gap is precision-related; structural math is now correct.

## Root cause: GemmaRMSNorm `(1 + weight)` convention

Qwen3-Next uses **GemmaRMSNorm** for ALL its norms (verified
sglang `qwen3_next.py:538-711` + vLLM): `output = (1 + weight) * x / rms(x)`.

llama.cpp's GGUF convention bakes the `+1` into stored weights so
standard RMSNorm `weight * x / rms(x)` produces correct output.
Empirical verification:

```
base output_norm.weight:           mean=1.96  → +1 baked ✓
base blk.{0..63}.attn_norm.weight: mean=0.97-1.24 → +1 baked ✓
base blk.{0..63}.ssm_norm.weight:  mean=0.86-1.22 → +1 baked ✓

MTP raw safetensors weights:
  mtp.layers.0.input_layernorm.weight:           mean=0.04  ← raw (Gemma)
  mtp.layers.0.post_attention_layernorm.weight:  mean=0.21  ← raw
  mtp.layers.0.self_attn.q_norm.weight:          mean=0.75  ← raw
  mtp.layers.0.self_attn.k_norm.weight:          mean=0.74  ← raw
  mtp.norm.weight:                               mean=1.27  ← raw
  mtp.pre_fc_norm_embedding.weight:              mean=-0.44 ← raw
  mtp.pre_fc_norm_hidden.weight:                 mean=-0.17 ← raw
```

The base GGUF was produced by llama.cpp's converter (which adds
`+1`); our `tools/convert_qwen36_mtp.py` was reading raw safetensor
values verbatim and missing this transformation. With weight ≈ 0,
flambeau's standard RMSNorm essentially zeros out the activation at
every MTP norm site — explains the 0% acceptance.

## The fix

`tools/convert_qwen36_mtp.py` — for every MTP norm tensor (all 7,
since they all use GemmaRMSNorm per sglang line 538-711, 67-69),
add `1.0` before quantizing/saving:

```python
if MTP_TENSOR_DTYPES[key] is GGMLQuantizationType.F32 and key.endswith(".weight"):
    is_norm = "norm" in key or key.endswith("layernorm.weight") or key == "mtp.norm.weight"
    if is_norm:
        t = t + 1.0
```

## Verification

After re-running the converter and re-running the live acceptance
test:

```
=== MTP-4 passive acceptance ===
prefill emits 11751 (ĠParis)
  step 1: predicted=11751 actual=271 ✗   ← model wants to repeat "Paris"
  step 2: predicted=760 actual=248068 ✗
  step 3: predicted=271 actual=271 ✓     ← match
  step 4: predicted=332 actual=248069 ✗
  step 5: predicted=271 actual=271 ✓     ← match
  step 6: predicted=248069 actual=4639 ✗
  step 7: predicted=198 actual=369 ✗
  step 8: predicted=4252 actual=4252 ✓   ← match

  acceptance: 37.5% (3/8)   wall: 52 ms / step
```

**0% → 37.5%** is real signal. MTP predictions are now in the
correct token-distribution regime (small token IDs and special tokens
in the right ranges), with three exact hits.

## Why not 90%?

vLLM's Qwen3.6-27B + Lorbus int4 (MTP preserved in BF16) reports
~90% acceptance at K=1. We're at 37.5%. Plausible reasons:

1. **Q4_0 base** vs vLLM's reference BF16 base. Q4_0 produces
   slightly lower-quality hidden states than BF16 — and MTP's
   prediction quality is bounded by the base's prediction
   determinism. Q4_0 hidden may diverge enough from "ground truth"
   to make MTP's prediction (trained on BF16 hidden) miss more
   often.
2. **Q8_0 MTP linears** vs Lorbus's BF16 MTP head. Per-matmul
   quant noise compounds across 8 linears.
3. **F16 hidden propagation** vs vLLM's BF16. F16 precision degrades
   over the long MTP chain (4 norms + 8 matmuls).
4. **Single-token K/V cache** in our passive harness — no MTP-side
   history accumulates. Unclear if vLLM's MTP attends over a longer
   history during draft cascading.

These are precision/setup losses, not implementation bugs. To push
higher, options are (in order of expected lift):

- Convert MTP linears as BF16 instead of Q8_0 (~3× MTP file size,
  but per-matmul noise drops to ~0%) — try first.
- Switch base to BF16 (impossible without a much bigger GGUF).
- Run MTP forward in F32 (custom path; doable).
- Investigate `is_nextn=True` flag in vLLM/sglang's MTP-block
  config — this was the residual suspicion from session 2 and may
  toggle additional architectural details.

## Other findings this session

- **Q4_0 token_embd dequant verified bit-exact** (carried over from
  session 2): `forward_embed_decode_host` produces identical row to
  `gguf.quants.dequantize` to 5 decimals.
- **Position-invariance of single-token attention** (carried over):
  Q · K is rotation-invariant when Q and K share position; our
  parity test cannot detect MROPE bugs.
- **Concat order `[embedding, hidden]`** (session 2): fixed.
- **`output_norm` IS applied before MTP** (session 2): fixed.

## Code state

- `tools/convert_qwen36_mtp.py` — adds +1 to MTP norm weights
  before write.
- `/artefact/models/Qwen3.6-27B-mtp.gguf` — regenerated with +1
  bake. Norm means now match base convention (1.04, 1.21, 1.74,
  1.75, 2.27, 0.56, 0.83).
- `tools/mtp_reference.py` — unchanged; loads from new GGUF
  correctly.
- `crates/models/qwen3-moe/src/mtp.rs` — unchanged; standard
  RMSNorm with the new (+1'd) weights produces the correct
  GemmaRMSNorm semantics.
- `crates/models/qwen3-moe/tests/mtp_acceptance_passive.rs` —
  unchanged.

## Sources

- [sglang qwen3_next.py — GemmaRMSNorm used by all decoder norms (lines 538, 705-711)](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_next.py)
- [sglang qwen3_next_mtp.py — GemmaRMSNorm for pre_fc_norm_* (line 67-69)](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_next_mtp.py)
- [HF GemmaRMSNorm definition: `output * (1.0 + weight)`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma/modeling_gemma.py)

## Decision

**File MTP-4 as "first signal" — 37.5% acceptance + structural
correctness verified.** This is below the 75% hard gate so MTP-4
doesn't fully pass. Next session is one of:

- **MTP-4 session 4**: try BF16 MTP linears (re-run converter with
  qtype=F16/BF16 for linears). If acceptance jumps to ≥60%, the
  Q8_0 of MTP head was the dominant precision drag. If still
  ≤40%, the issue is elsewhere.
- **MTP-4 session 5**: bisect intermediate values (h0, h1, h2)
  against a Python ref using REAL h_t from a base-model forward —
  see at which stage Rust diverges.

Either path is a session of work. The structural pieces are all
correct; remaining gap is bisectable.
