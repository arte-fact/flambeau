# MTP-1 — Qwen3.6-27B safetensors audit (MTP head)

**Verdict:** MTP head exists in `Qwen/Qwen3.6-27B` main release, single
layer, ~393M params, structurally simpler than a base layer. Math
is DeepSeek-V3 / EAGLE-2 style: `(h_t, e_{t+1}) → predicted h_{t+1}` →
reuses base LM head. Path B is real and tractable.

## Source

- HF repo: `Qwen/Qwen3.6-27B` main branch.
- Total weights: 55.6 GB BF16 across 15 safetensors shards.
- `model.safetensors.index.json` confirms 15 `mtp.*` tensors in
  shards 13 and 15. Earlier WebFetch hallucinated "no MTP keys" —
  direct grep on the index file is authoritative.

## Config keys (text_config)

```
"mtp_num_hidden_layers": 1,
"mtp_use_dedicated_embeddings": false,
"hidden_size": 5120,
"intermediate_size": 17408,
"head_dim": 256,
"num_attention_heads": 24,
"num_key_value_heads": 4,
"vocab_size": 248320,
"attn_output_gate": true,            // base model only — MTP layer doesn't have it
"layer_types": [linear_attention × 3, full_attention] × 16,  // 64 layers, 1:3 ratio
```

`mtp_use_dedicated_embeddings: false` confirms MTP **reuses
`model.language_model.embed_tokens.weight`** — no new embedding table.
`mtp.lm_head.weight` is absent from the index, so MTP **also reuses
`lm_head.weight`** (the base LM head). This is good — only the MTP
block + 4 norms + the FC projection need new loader paths.

## MTP tensors in the index

| Tensor | Role | Expected shape (BF16) |
|---|---|---|
| `mtp.pre_fc_norm_hidden.weight` | norm of base hidden h_t | `[5120]` |
| `mtp.pre_fc_norm_embedding.weight` | norm of next-token embedding e_{t+1} | `[5120]` |
| `mtp.fc.weight` | project `concat([norm_h, norm_e])` → hidden | `[5120, 10240]` (or transposed) |
| `mtp.layers.0.input_layernorm.weight` | pre-attn RMSNorm | `[5120]` |
| `mtp.layers.0.self_attn.q_proj.weight` | Q projection | `[6144, 5120]` (24 heads × 256 = 6144) |
| `mtp.layers.0.self_attn.k_proj.weight` | K projection | `[1024, 5120]` (4 heads × 256 = 1024) |
| `mtp.layers.0.self_attn.v_proj.weight` | V projection | `[1024, 5120]` |
| `mtp.layers.0.self_attn.o_proj.weight` | O projection | `[5120, 6144]` |
| `mtp.layers.0.self_attn.q_norm.weight` | per-head Q norm (Qwen3 convention) | `[256]` |
| `mtp.layers.0.self_attn.k_norm.weight` | per-head K norm | `[256]` |
| `mtp.layers.0.post_attention_layernorm.weight` | pre-MLP RMSNorm | `[5120]` |
| `mtp.layers.0.mlp.gate_proj.weight` | SwiGLU gate | `[17408, 5120]` |
| `mtp.layers.0.mlp.up_proj.weight` | SwiGLU up | `[17408, 5120]` |
| `mtp.layers.0.mlp.down_proj.weight` | SwiGLU down | `[5120, 17408]` |
| `mtp.norm.weight` | final RMSNorm before LM head | `[5120]` |

Total params: ~393M (≈786 MB BF16; ≈393 MB Q8_0).

**Critical structural notes:**

1. **MTP attention is NOT gated** (no `attn_gate` like base model has
   when `attn_output_gate=true`). MTP uses plain Qwen3-style attention
   with q_norm/k_norm.
2. **MTP attention is NOT GDN** — full-attention with KV cache. Much
   simpler than base model's GDN+full-attn hybrid.
3. **MTP MLP is dense SwiGLU** — `gate_proj`, `up_proj`, `down_proj`.
   No MoE.
4. **MTP reuses MROPE** (the base model's RoPE config:
   `dimension_count=64, sections=[11,11,10,0]`,
   `partial_rotary_factor=0.25`, `rope_theta=10M`). Need to apply
   MROPE inside MTP self-attn.

## Forward math (DeepSeek-V3 / EAGLE-2 style)

```text
Inputs:
    h_t   : F16 [5120]    — base hidden state at position t (after model.norm)
    t_{t+1}: u32         — next token id (sampled from h_t @ lm_head)
Output:
    logits_{t+2} : F32 [vocab_size]   — distribution over token t+2

Steps:
    e_{t+1} = embed_tokens(t_{t+1})                       // shared with base
    norm_h  = RMSNorm(h_t,    pre_fc_norm_hidden,    eps)
    norm_e  = RMSNorm(e_{t+1}, pre_fc_norm_embedding, eps)
    fc_in   = concat([norm_h, norm_e])  shape [10240]
    h0      = fc_in @ mtp.fc.weight^T   shape [5120]      // [10240]→[5120] dense

    // ─── one transformer block ───
    h0n     = RMSNorm(h0, layers.0.input_layernorm, eps)
    q       = h0n @ q_proj^T  → reshape [num_q_heads, head_dim]
    k       = h0n @ k_proj^T  → reshape [num_kv_heads, head_dim]
    v       = h0n @ v_proj^T  → reshape [num_kv_heads, head_dim]
    q       = RMSNorm-per-head(q, q_norm)
    k       = RMSNorm-per-head(k, k_norm)
    q, k    = MROPE(q, k, position=t)                     // shared positional encoding
    attn_out= scaled_dot_attn(q, k, v, mask=causal_with_mtp_kv)
    h1      = h0 + (attn_out @ o_proj^T)
    h1n     = RMSNorm(h1, layers.0.post_attention_layernorm, eps)
    mlp_out = (silu(h1n @ gate_proj^T) * (h1n @ up_proj^T)) @ down_proj^T
    h2      = h1 + mlp_out
    // ──────────────────────────────

    h_norm  = RMSNorm(h2, mtp.norm, eps)
    logits  = h_norm @ lm_head.weight^T                   // shared with base
```

For K-step lookahead (K > 1), cascade: feed (h2 from previous step,
e_{drafted next token}) back through the same MTP block. The single
MTP block weight is reused; per-step KV cache for MTP grows.

## Acceptance ceiling

vLLM Qwen3.6-27B recipe + Lorbus int4 (with MTP in BF16) reports:
- K=1: 1.9 per-pos accepted, ~+8% TPS
- K=2: 2.4 per-pos accepted, ~+14% TPS
- K=3: 3.4 per-pos accepted, ~+25% TPS (peak ROI)

These are real-world numbers on warmed-up hardware with vLLM's
plumbing. We can use them as the target band for our own MTP-4
gate (≥75% acceptance at K=1 leaves margin).

## Implementation lift for MTP-2..MTP-5

**MTP-2 (1 session) — converter:**
- Fork or fresh Python script: read 15 safetensors shards from
  `Qwen/Qwen3.6-27B`, isolate the `mtp.*` keys, write a separate
  `Qwen3.6-27B-mtp.gguf` containing ONLY those weights (Q8_0 quant
  for the linear layers, F32 for norms). Tag it with metadata
  pointing at the base model checksum so we can require matched
  pairs. ≈ 350 MB output.
- Why a separate file vs grafting onto the existing GGUF: simpler;
  user keeps their existing Unsloth GGUF for the base; we only
  download MTP-specific shards.

**MTP-3 (1 session) — loader + smoke:**
- New `MtpHeadWeights` struct in flambeau loader.
- Implement `forward_mtp_step` composed of existing ops:
  rmsnorm_f16, dense matmul (Q8_0 mmvq for projections), rope, attn,
  swiglu. **No new kernels needed** — every primitive exists today.
- Smoke vs Python (HuggingFace transformers) on a fixed input.
  Assert max-rel-err < 1e-3 (same tolerance bracket as our V1.7.4
  parity tests).

**MTP-4 (1-2 sessions) — speculative driver + GATE:**
- Reland `KvCache::rollback` (V2.33 had it; reverted with the spec
  branch).
- Allocate a draft KV cache (small — K-token history × num_layers=1).
- Greedy verify loop: target hidden h_t → MTP draft tokens
  t+1..t+K → target prefill on that K-token sequence → compare
  argmax at each position → accept matching prefix → rollback target
  KV beyond accept length.
- Measure on `enable_thinking=false` (per the upstream finding)
  with a real chat workload (the simplex-noise long generation
  prompt from V2.33).
- HARD GATE: ≥75% acceptance AND ≥1.4× combined throughput vs eager.

**MTP-5 (1-2 sessions) — TP/PP integration:**
- MTP head sits on the LM-head rank (last PP stage). Already where
  `output.weight` lives; no extra cross-rank copy.
- Smoke + bench at the chosen topology (likely pp4 since this
  qwen35-dense-hybrid model fits comfortably; pp2tp2 if VRAM pressure
  surfaces from carrying a draft KV).

## What flambeau does NOT need (good)

Compared to the DFlash audit:
- ✗ Sliding-window attention (MTP uses standard causal + KV cache).
- ✗ Cross-rank hidden-state taps (MTP only needs h_t from the FINAL
  layer, which is already on the head rank).
- ✗ Block-diffusion drafting (MTP is plain greedy/sampled AR).
- ✗ Custom new kernel families (MTP uses existing ops).
- ✗ Separate token embedder or LM head (shared with base).

## Recommendation

Proceed to MTP-2 with the Path B scoping confirmed. The lift is
clean — every kernel primitive already exists, the loader extension
is contained, and the only "new" infra is the converter (Python
script) + speculative driver (reuses the V2.33 scaffold pattern).

## Sources

- [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) — main BF16 release with MTP weights in shards 13, 15.
- [Qwen/Qwen3.6-27B `model.safetensors.index.json`](https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/model.safetensors.index.json) — authoritative tensor map; downloaded + grepped this session.
- [Lorbus/Qwen3.6-27B-int4-AutoRound](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) — confirms MTP head sourced from main release; preserves it in BF16; ~90% acceptance via vLLM.
- [vLLM Qwen3.6-27B recipe](https://recipes.vllm.ai/Qwen/Qwen3.6-27B) — canonical MTP config + acceptance numbers.
