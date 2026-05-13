# Gemma4 GGUF reconnaissance

**S1 deliverable.** Source: `/artefact/models/gemma-4-*.gguf`, audited
2026-05-13 via `cargo run -p flambeau-cli --release -- inspect-gguf`.
Cross-checked against `/artefact/llama.cpp/src/llama-model.cpp` lines
1629–1656 (hparams) and 4552–4700 (tensor loader) and
`src/models/gemma4-iswa.cpp` (forward graph).

Raw dumps: `/tmp/gemma4_{e4b,31b_q4,31b_q8,26b_q8,26b_ud}_inspect.txt`.

## File inventory

| File | Size | Variant | n_layer | n_head | n_kv_head | hidden | ff_len | swa_win | shared_kv | experts | per_layer_embd |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `gemma-4-E4B-it-Q4_0.gguf` | 4.84 GB | E4B (edge dense) | 42 | 8 | 2 | 2560 | 10240 | 512 | 18 | dense | 256 |
| `gemma-4-26B-A4B-it-Q8_0.gguf` | 26.86 GB | 26B-A4B (MoE) | 30 | 16 | 8 | 2816 | 2112 | 1024 | 0 | 128 / 8 used | 0 |
| `gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf` | 27.87 GB | 26B-A4B-UD-XL | 30 | 16 | 8 | 2816 | 2112 | 1024 | 0 | 128 / 8 used | 0 |
| `gemma-4-31B-it-Q4_0.gguf` | 17.34 GB | 31B (dense) | 60 | 32 | 16 | 5376 | 21504 | 1024 | 0 | dense | 0 |
| `gemma-4-31B-it-Q8_0.gguf` | 32.64 GB | 31B (dense) | 60 | 32 | 16 | 5376 | 21504 | 1024 | 0 | dense | 0 |

`n_layer → variant` matches the llama.cpp switch
(`30→26B-A4B, 35→E2B, 42→E4B, 60→31B`). No E2B file locally.

`head_count_kv` is stored as an **array** of length `n_layer`, even though
all entries are identical in every file we have. The config loader must
read it as `Vec<u32>` and broadcast / verify uniformity.

## Per-variant hparams (full set)

Shared across all 5 files: `key_length=512`, `key_length_swa=256`,
`value_length=512`, `value_length_swa=256`, `head_dim=256` (full),
`head_dim_swa=128`, `rope.freq_base=1000000`, `rope.freq_base_swa=10000`,
`rope.dimension_count=512`, `rope.dimension_count_swa=256`,
`layer_norm_rms_epsilon=1e-6`, `final_logit_softcapping=30`,
`context_length=131072 (E4B)` or `262144 (others)`.

MoE-specific (26B-A4B only): `expert_count=128`, `expert_used_count=8`,
`expert_feed_forward_length=704` (n_ff_exp).

Per-layer side-channel (E4B only): `embedding_length_per_layer_input=256`,
`attention.shared_kv_layers=18` (= last 18 of 42 layers reuse earlier
KV at inference time, per llama.cpp `n_layer_kv_from_start` mechanic).

### Gap — sliding_window_pattern is truncated by inspect-gguf

`inspect-gguf` prints `[true, true, true, true, …(N elems)]` for any
array > 4 entries. The actual per-layer SWA/full pattern is not
visible in the dump; can only be confirmed via the gemma4 config
loader reading the bool array directly (action: parse via the GGUF
reader at S4 time). `rope_freqs.weight` is **present** in every file,
which proves at least one full-attention layer exists per model (the
weight is `TENSOR_DUPLICATED` from the first full-attn layer in
llama.cpp).

## Tensor inventory (deduplicated, per family)

### Global tensors (all 5 files)

| Name | Required | E4B | 26B-A4B | 31B |
|---|---|---|---|---|
| `output_norm` | always | F32 [hidden] | F32 [hidden] | F32 [hidden] |
| `token_embd` | always | Q4_K [vocab,hidden] | Q8_0 / mixed | Q4_K / Q8_0 |
| `rope_freqs` | always (full-attn marker) | F32 [256] | F32 [256] | F32 [256] |
| `per_layer_token_embd` | **E4B only** | Q5_K [vocab, per_layer_embd × n_layer] | — | — |
| `per_layer_model_proj` | **E4B only** | BF16 [per_layer_embd × n_layer, hidden] | — | — |
| `per_layer_proj_norm` | **E4B only** | F32 [per_layer_embd] | — | — |

`output` (lm head) is absent in every file → falls back to tied
`token_embd` (matches llama.cpp `TENSOR_DUPLICATED` path at
`model.cpp:4567`).

### Per-layer tensors

**Common to every layer (all variants):**

| Name | E4B Q4_0 | 26B Q8_0 | 31B Q4_0 | Notes |
|---|---|---|---|---|
| `attn_q.weight` | Q4_0 [n_head·D, hidden] | Q8_0 | Q4_0 | column-parallel |
| `attn_k.weight` | Q4_0 [n_kv_head·D, hidden] | Q8_0 | Q4_0 | (optional for shared-KV tail) |
| `attn_v.weight` | Q4_0 [n_kv_head·D, hidden] | Q8_0 | Q4_0 | (optional for shared-KV tail) |
| `attn_output.weight` | Q4_0 [hidden, n_head·D] | Q8_0 | Q4_0 | row-parallel |
| `attn_norm.weight` | F32 [hidden] | F32 | F32 | pre-attn RMSNorm |
| `attn_q_norm.weight` | F32 [head_dim] | F32 [256] | F32 [256] | Q RMSNorm per-head |
| `attn_k_norm.weight` | F32 [head_dim] | F32 [256] | F32 [256] | K RMSNorm per-head |
| `post_attention_norm.weight` | F32 [hidden] | F32 | F32 | post-attn RMSNorm |
| `ffn_norm.weight` | F32 [hidden] | F32 | F32 | pre-FFN RMSNorm |
| `post_ffw_norm.weight` | F32 [hidden] | F32 | F32 | post-FFN RMSNorm |
| `layer_output_scale.weight` | F32 [1] | F32 [1] | F32 [1] | per-layer scalar |

**Dense FFN layers (E4B all + 31B all):**

| Name | Dtype example | Shape |
|---|---|---|
| `ffn_gate.weight` | Q4_0 | [ff_len, hidden] |
| `ffn_up.weight` | Q4_0 | [ff_len, hidden] |
| `ffn_down.weight` | Q4_1 (E4B/31B-Q4) | [hidden, ff_len] |

`ffn_down` is **Q4_1 not Q4_0** in the Q4_0 files (asymmetric quant
gives better dynamic range on the down-projection). 31B-Q8_0 has it as
Q8_0 uniformly.

**MoE layers (26B-A4B):** EVERY layer has BOTH a shared dense MLP **and** a routed-expert
branch run in parallel and summed.

| Name | Dtype | Shape | Notes |
|---|---|---|---|
| `ffn_gate.weight` | Q8_0 | [ff_len=2112, hidden=2816] | shared MLP |
| `ffn_up.weight` | Q8_0 | [2112, 2816] | shared MLP |
| `ffn_down.weight` | Q8_0 | [2816, 2112] | shared MLP |
| `ffn_gate_inp.weight` | F32 | [n_expert=128, hidden=2816] | router matmul |
| `ffn_gate_inp.scale` | F32 | [hidden=2816] | **router pre-scale tensor** (= llama.cpp `ffn_gate_inp_s`) |
| `ffn_gate_up_exps.weight` | Q8_0 | [128, **1408**, 2816] | **FUSED gate+up** (1408 = 2·n_ff_exp=2·704) |
| `ffn_down_exps.weight` | Q8_0 | [128, 2816, 704] | per-expert down |
| `ffn_down_exps.scale` | F32 | [128] | per-expert scale (used in MoE downproj math) |
| `pre_ffw_norm_2.weight` | F32 | [hidden] | pre-MoE RMSNorm (parallel to ffn_norm for shared) |
| `post_ffw_norm_1.weight` | F32 | [hidden] | post-shared-MLP RMSNorm |
| `post_ffw_norm_2.weight` | F32 | [hidden] | post-MoE RMSNorm |

`post_ffw_norm.weight` exists alongside `post_ffw_norm_{1,2}` — it's
the per-layer "final" norm applied to the summed FFN output (shared +
MoE), not the per-branch norms.

**E4B per-layer side-channel (E4B only):**

| Name | Dtype | Shape | Notes |
|---|---|---|---|
| `inp_gate.weight` | F32 | [per_layer_embd=256, hidden=2560] | gate matmul (`per_layer_inp_gate`) |
| `proj.weight` | F32 | [hidden=2560, per_layer_embd=256] | back-projection (`per_layer_proj`) |
| `post_norm.weight` | F32 | [hidden] | side-channel post-RMSNorm (`per_layer_post_norm`) |

## Name mapping (file → llama.cpp `LLM_TENSOR_*` → flambeau target)

| GGUF name | llama.cpp | Notes |
|---|---|---|
| `token_embd` | LLM_TENSOR_TOKEN_EMBD | |
| `output_norm` | LLM_TENSOR_OUTPUT_NORM | |
| `rope_freqs` | LLM_TENSOR_ROPE_FREQS | only loaded for full-attn layers; `TENSOR_DUPLICATED` after the first |
| `per_layer_token_embd` | LLM_TENSOR_PER_LAYER_TOKEN_EMBD | E4B-only |
| `per_layer_model_proj` | LLM_TENSOR_PER_LAYER_MODEL_PROJ | E4B-only |
| `per_layer_proj_norm` | LLM_TENSOR_PER_LAYER_PROJ_NORM | E4B-only |
| `blk.N.attn_{q,k,v,output}` | LLM_TENSOR_ATTN_{Q,K,V,OUT} | wk/wv `TENSOR_NOT_REQUIRED` when `has_kv(il)=false` |
| `blk.N.attn_{q,k}_norm` | LLM_TENSOR_ATTN_{Q,K}_NORM | attn_k_norm also `TENSOR_NOT_REQUIRED` for shared-KV tail |
| `blk.N.attn_norm` | LLM_TENSOR_ATTN_NORM | |
| `blk.N.post_attention_norm` | LLM_TENSOR_ATTN_POST_NORM | |
| `blk.N.ffn_{gate,up,down}` | LLM_TENSOR_FFN_{GATE,UP,DOWN} | shared MLP (every layer in MoE; only FFN in dense) |
| `blk.N.ffn_norm` | LLM_TENSOR_FFN_NORM | pre-shared-MLP norm |
| `blk.N.post_ffw_norm` | LLM_TENSOR_FFN_POST_NORM | post-FFN-output norm |
| `blk.N.pre_ffw_norm_2` | LLM_TENSOR_FFN_PRE_NORM_2 | pre-MoE norm (MoE only) |
| `blk.N.post_ffw_norm_1` | LLM_TENSOR_FFN_POST_NORM_1 | post-shared-MLP norm (MoE only) |
| `blk.N.post_ffw_norm_2` | LLM_TENSOR_FFN_POST_NORM_2 | post-MoE norm (MoE only) |
| `blk.N.ffn_gate_inp.weight` | LLM_TENSOR_FFN_GATE_INP | router matmul |
| `blk.N.ffn_gate_inp.scale` | LLM_TENSOR_FFN_GATE_INP_S | **router pre-scale** (note: GGUF name uses `.scale` suffix, not a separate tensor name) |
| `blk.N.ffn_gate_up_exps.weight` | LLM_TENSOR_FFN_GATE_UP_EXPS | fused gate+up per expert |
| `blk.N.ffn_down_exps.weight` | LLM_TENSOR_FFN_DOWN_EXPS | per-expert down |
| `blk.N.ffn_down_exps.scale` | (companion scale tensor) | indexed per-expert F32 scale |
| `blk.N.layer_output_scale` | LLM_TENSOR_LAYER_OUT_SCALE | per-layer scalar |
| `blk.N.inp_gate` | LLM_TENSOR_PER_LAYER_INP_GATE | E4B-only |
| `blk.N.proj` | LLM_TENSOR_PER_LAYER_PROJ | E4B-only |
| `blk.N.post_norm` | LLM_TENSOR_PER_LAYER_POST_NORM | E4B-only |

GGUF convention: tensors with a companion `.scale` (e.g.
`ffn_gate_inp.scale`, `ffn_down_exps.scale`) are stored as separate
entries under the same base name. The loader needs both.

## Dtype coverage audit

| File | Dtypes used | Native HIP MMVQ/MMQ? |
|---|---|---|
| E4B-Q4_0 | F32, BF16, Q4_0, Q4_1, Q4_K, Q5_K | Q4_0/Q4_1/Q4_K/Q5_K ✓; BF16 ✓ (per QDtype); **only used for per_layer_model_proj — a matmul** |
| 31B-Q4_0 | F32, Q4_0, Q4_1, Q4_K | ✓ |
| 31B-Q8_0 | F32, Q8_0 | ✓ |
| 26B-A4B-Q8_0 | F32, Q8_0 | ✓ |
| 26B-A4B-UD-Q8_K_XL | F32, BF16, Q8_0, Q8_K | Q8_K is in QDtype (Phase 4); **BF16 verify needed** |

**Verify-on-load action for S4:** BF16 matmul dispatch on gfx906 — used
for `per_layer_model_proj` in E4B (BF16 [10752, 2560]) and somewhere
in 26B-UD. Check existing BF16 mmvq/MMQ row in `dispatch/hip/gfx906.toml`.

No MXFP4 leaves in any of these 5 files. No IQ family. No dtype 23.
The UD-Q8_K_XL files do not trip the historical gap; the
[[project-quant-coverage-post-phase4]] memory entry's prediction holds.

## Tokenizer findings

All 5 files: `tokenizer.ggml.model = "gemma4"` (new family),
`vocab_size = 262144`, `bos = 2`, `eos = 106`, `pad = 0`, `unk = 3`,
`mask = 4`, `add_space_prefix = false`.

| File | `add_bos_token` (GGUF) | Notes |
|---|---|---|
| E4B-Q4_0 | **false** | needs override at load (llama.cpp PR #21500) |
| 26B-A4B-Q8_0 | **false** | needs override |
| 26B-A4B-UD-Q8_K_XL | **true** | already correct |
| 31B-Q4_0 | **false** | needs override |
| 31B-Q8_0 | **false** | needs override |

**Action for S8 tokenizer task:** ignore the GGUF
`add_bos_token` flag for gemma4-arch files; always force `add_bos =
true`. The chat templates in these files also expect a BOS, and 4/5
files have the wrong flag set at conversion time.

The base tokenizer uses 514906 BPE merges. Per llama.cpp commits
referenced in the plan:
- #21488 — byte-token handling in BPE detokenize.
- #21406 — custom newline split pretokenize rule.
- #21492 — strip `</s>` from EOG set (EOS=106 here is `<end_of_turn>`, not `</s>`).
- #21534, #21343 — additional tokenizer edge cases.

## Per-variant size estimates (decode at 4k ctx)

| Model | Weights | KV (F16, full ctx=4k) | Total | 16 GB MI50 fit? |
|---|---:|---:|---:|---|
| E4B-Q4_0 | 4.84 GB | ~84 MB (42L × 2 KV × 256D × 4096 × 2B × 2) | ~5 GB | single ✓ |
| 31B-Q4_0 | 17.34 GB | ~480 MB (60L × 16 × 256 × 4096 × 2 × 2) | ~18 GB | **PP2 needed** at full ctx; PP1 at small ctx |
| 31B-Q8_0 | 32.64 GB | ~480 MB | ~33 GB | PP2 minimum, PP4 comfortable |
| 26B-A4B-Q8_0 | 26.86 GB | ~480 MB (30L × 8 × 256 × 4096 × 2 × 2 × 2 K+V) | ~27 GB | PP2 minimum, PP4 comfortable |
| 26B-A4B-UD-Q8_K_XL | 27.87 GB | ~480 MB | ~28 GB | PP2 minimum |

For shared-KV tail (E4B), the effective KV count is 24 layers × 4 KV
heads = ~48 MB at 4k ctx — about half the naïve count, since the last
18 layers reuse earlier KV.

## Composition gates for S5–S7

Distilled requirements that downstream tasks consume:

1. **Per-layer LayerSpec** holds `(is_swa, has_kv, kv_share_src, is_moe, n_head, n_kv_head, head_dim_q, head_dim_kv, rope_base, ff_len_or_n_ff_exp)`. Most fields broadcast from scalar metadata; `n_kv_head` reads the array form.
2. **SWA pattern** must be loaded as `Vec<bool>[n_layer]` — `attention.sliding_window_pattern` array. Inspect-gguf truncates display but the array is present in metadata.
3. **Shared-KV mapping** (E4B only): construct from `attention.shared_kv_layers=18` (last 18 layers reuse). The kv_share_src layer mapping is determined by llama.cpp's `inp_attn_kv_iswa` — needs a read of `build_inp_attn_kv_iswa` to confirm pairing rule (likely "last full-attn / last SWA layer ≤ n_layer_kv_from_start of the same type").
4. **Router pre-scale**: every MoE layer has `ffn_gate_inp.scale` F32 [hidden]. Multiply `attn_out` by this BEFORE the router matmul.
5. **Fused expert gate+up**: split `ffn_gate_up_exps` `[E, 2·F, H]` into `gate_exps[E, F, H]` + `up_exps[E, F, H]` at load — no new MMVQ kernel needed.
6. **No `output` tensor** in any file — tied to `token_embd`.
7. **`add_bos_token` override** in tokenizer config — always true for gemma4 arch.
8. **BF16 matmul dispatch confirmation** — exercised by E4B `per_layer_model_proj` and UD-XL files.

## Risks resolved / remaining

| Risk | Status |
|---|---|
| UD-Q8_K_XL fails load (historical dtype-23 / IQ gap) | **resolved** — only F32/BF16/Q8_0/Q8_K used; all in QDtype |
| MXFP4 dispatch needed | **not present** — none of the 5 files use MXFP4 |
| Per-layer head_dim differs across layers | **no in our files** — head_count_kv stored as array but uniform; head_dim is scalar; need to keep code path generic for future model variants |
| sliding_window_pattern not visible in inspect | **flagged** — extend inspect-gguf or read array directly at S4 |
| `output` lm-head tied to embd | **expected** — matches llama.cpp behavior, no special action |
| BF16 matmul on gfx906 unverified | **action** — confirm `dispatch/hip/gfx906.toml` has BF16 MMVQ row before S5 |
| `ffn_gate_inp.scale` semantics | **clarified** — pre-router scalar broadcast multiply on the attn_out activation; not a per-expert correction |

## Bottom line

5 files cover 3 architectures (E4B edge dense with per-layer-embd +
shared-KV; 26B-A4B MoE with fused experts; 31B large dense). All weights
load natively in current QDtype. Two arch-specific features
(per-layer-embd and shared-KV tail) are confined to E4B and can be
implemented as edge-only fast paths. The MoE + softcap + SWA path
covers 26B-A4B + 31B + UD-XL and is the shorter critical path to a
useful gemma4 first cut. S4 can start.
