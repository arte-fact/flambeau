# gemma4-v2 E4B / E2B (per-layer-embd) — partial implementation status

Date: 2026-05-20
Status: **scaffolding lands; coherence-blocker localised to shared-KV
routing — distinct from the per-layer-embd path**

E4B now boots through the full forward path including the per-layer
side-channel build + apply, but generates incoherent tokens on real
prompts. The infrastructure is in place. The output-coherence bug has
been localised (see "Root cause" below); it is a *different* missing
feature from the per-layer-embd, so the next session is two separate
pieces of work, not a debug diff.

## What landed

1. **`flambeau-blocks::per_layer_embd::build_inp_per_layer_table`** —
   host-side helper that produces the `[n_layer × pe]` F32 table once
   per token from the GGUF raw bytes:
   - dequant `per_layer_token_embd[token]` (Q5_K) × `sqrt(pe)`
   - `per_layer_model_proj @ inp_batch_f16` (BF16 × F16 → F32) ×
     `1 / sqrt(hidden)`
   - rmsnorm with `per_layer_proj_norm` per layer slice
   - `(table + proj) × (1 / sqrt(2))`
2. **gemma4-v2 config**: removed the `PerLayerEmbdUnsupported` bail.
   Parses `gemma4.embedding_length_per_layer_input` into
   `PerLayerEmbdDims { pe }`.
3. **gemma4-v2 loader**: when `config.per_layer_embd.is_some()`, loads
   - 3 globals: `per_layer_token_embd` (raw mmap copy + dtype + row
     byte stride), `per_layer_model_proj` (raw mmap copy + dtype),
     `per_layer_proj_norm` (raw mmap copy). Allocates a device buffer
     for the per-token table.
   - 3 per-layer tensors: `blk.N.inp_gate` (F32 device), `blk.N.proj`
     (F32 device), `blk.N.post_norm` (cast F32 → F16 on upload).
4. **forward scratch pool**: 6 new F32/F16 scratch slots for the
   apply block, gated on `ScratchConfig::per_layer_embd > 0`.
5. **`ForwardCtx`**: two new trait methods + engine impls:
   - `per_layer_embd_build_table` — DtoH read the current token's
     embedding, run the host build, HtoD upload to `table_dev`.
   - `per_layer_embd_apply` — wires the existing
     `flambeau_blocks::PerLayerEmbedBlock::forward_decode` into the
     engine with pool-resident scratch.
6. **gemma4-v2 model.rs**: per-token forward integration. Builds the
   table before the layer loop; calls the apply between the FFN
   residual add and the `layer_output_scale`. For prefill (n > 1) it
   loops the entire forward token-by-token (per-token build + apply
   require n=1; batched table build is a follow-up).

## Current behaviour

Boot succeeds. A short chat call (`"The capital of France is"`,
24-token decode) returns 24 random unicode-soup tokens. A diagnostic
flag (`FLAMBEAU_E4B_NO_PLE=1` — bypassed the apply block during
debug, then removed) showed the same garbage shape *without* the
per-layer side-channel apply running. So the bug is upstream of the
per-layer-embd path.

## Root cause — KV sharing for the tail 18 layers

`gemma4.attention.shared_kv_layers = 18` on the E4B GGUF means the
tail 18 layers (indices 24..41) **do not own their KV — they reuse
K/V written by earlier layers' caches**.

llama.cpp's `src/llama-hparams.cpp:231` computes
`n_layer_kv_from_start = n_layer - n_kv_shared_layers = 42 - 18 = 24`.
Then `src/models/gemma4-iswa.cpp:79`:

```cpp
if (hparams.has_kv(il)) {                // il < 24
    // compute K, V; KV-write; attention read
} else {                                 // il in [24, 42)
    // compute Q only; read K/V from a previous layer's cache
    cur = build_attn(inp_attn, wo, ..., Qcur, nullptr, nullptr, ...);
}
```

v2's `flambeau_blocks::StandardAttention` has no KV-share routing.
It runs the full Q + K + V projection + KV-write + attention read
for every layer. For layers 24..41 this:

1. Wastes the K, V projection work
2. Writes K, V to a slot that is never read
3. Reads zeros (or stale state) from the layer's own — never written —
   KV cache for the actual attention computation
4. Produces a near-zero or garbage attention delta
5. Pollutes the residual stream with bad attention output

That alone is enough to make output incoherent. The per-layer-embd
math added on top is correct in shape but reading from a residual
that's already corrupt.

The GGUF still lists `blk.N.attn_v.weight` for `il >= 24` (legacy
loader artifact from the original Google export) but llama.cpp
silently ignores those tensors when `has_kv(il) == false`.

## What to do next session (split into two parts)

**Part A — KV-share routing (unblocks coherent E4B output)**:

1. Add per-layer `has_kv: bool` to gemma4-v2's config / layout.
   Set `has_kv = il < n_layer - shared_kv_layers`. (And, separately:
   load `gemma4.attention.shared_kv_layers` as an optional u32 in
   `Gemma4V2Config::from_gguf`.)
2. In gemma4-v2's loader: when `has_kv == false`, skip loading
   `attn_k`, `attn_v`, `attn_k_norm`, `attn_v_unit_norm_w` for that
   layer.
3. In `flambeau_blocks::StandardAttention` (or a small `StandardAttention
   ::with_shared_kv` builder) plus `standard_attn_local`: when the
   layer has `has_kv == false`, skip the K/V projection + KV write +
   the `kv_append` kernel; the attention-read should use the
   share-src layer's KV cache slot.
4. Loader needs a `kv_share_src` resolver (legacy gemma4's
   `ModelLayout::resolve_kv_sharing` is the existing reference).
   The simplest mapping for E4B: `kv_share_src[il] = il_of_last_full_layer_le(il)`
   where the "last full layer" tracks the most recent has_kv layer in
   the SWA/full alternation.

**Part B — validate the per-layer-embd math once attention is right**:

1. Re-add a `FLAMBEAU_E4B_NO_PLE` diagnostic gate in the model.rs
   per_layer_embd_apply call site.
2. Boot E4B with KV-share landed. With `FLAMBEAU_E4B_NO_PLE=1` the
   model should be runnable (probably degraded coherence — the
   per-layer side-channel is load-bearing — but not garbage soup).
3. Drop the env flag and confirm coherence vs llama.cpp.

## Files changed (commit boundary)

- `crates/blocks/src/per_layer_embd.rs` (host build helper)
- `crates/blocks/Cargo.toml` (no change — quant dep already present)
- `crates/models/gemma4-v2/Cargo.toml` (added `flambeau-blocks` dep)
- `crates/models/gemma4-v2/src/{config,loader,arch,model}.rs`
- `crates/forward/src/{ctx,engine}.rs` (trait + impl)
- `crates/forward/src/core/scratch.rs` (config field + scratch slots)
- `crates/models/qwen35-v2/src/arch.rs`,
  `crates/models/qwen35moe-v2/src/arch.rs` (ScratchConfig default
  field bump)

No new kernels needed — every op the apply uses
(`dense_gemv_f32_f16`, `gelu_mul_f32`, `cast_f32_to_f16`,
`rmsnorm_f16`, `add_f16`) was already there.

## Honest call

This was scoped as a multi-session feature port and that estimate
held. ~360 LOC of plumbing landed cleanly; the math doesn't validate
yet. Next session is the diff-against-llama.cpp debug pass. The
infrastructure is solid foundation — none of it needs to be redone.
