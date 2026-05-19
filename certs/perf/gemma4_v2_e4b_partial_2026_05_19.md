# gemma4-v2 E4B / E2B (per-layer-embd) — partial implementation status

Date: 2026-05-19
Status: **scaffolding lands, output not coherent yet**

E4B now boots through the full forward path including the per-layer
side-channel build + apply, but generates incoherent tokens on real
prompts. The infrastructure is in place; the math doesn't pass
coherence and needs a side-by-side diff against llama.cpp to localise.

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
24-token decode) returns 24 random unicode-soup tokens. Disabling the
apply block (test env `FLAMBEAU_E4B_NO_PLE=1` during the debug, now
removed) also produces garbage — different garbage. That means the
output isn't right *with or without* the per-layer side-channel:

- The base gemma4 forward path on E4B itself isn't producing
  coherent state. Suspects to chase next session:
  - **GGUF reports `gemma4.attention.shared_kv_layers = 18` but every
    blk.N has its own `attn_v.weight`** — investigate whether this is
    metadata-only or whether some layers must read KV from a peer.
    Legacy gemma4 has `kv_share_src` resolver logic; v2 does not.
  - **Per-layer attention head_dim alternates** (256 for SWA, 512 for
    full-attn) — verify the config-driven dispatch path lines up.
  - **Q/K-norm per-head dim** (256 vs 512 per layer) — verify the
    norm weights are loaded with matching dims.
- The per-layer side-channel math itself may also be off; not yet
  validated independently of the rest.

## What to validate next session

1. Boot E4B with `FLAMBEAU_E4B_NO_PLE=1` (re-add the diagnostic flag
   in model.rs at the per_layer_embd_apply call site). If output is
   *still* garbage, the per-layer-embd path is downstream of the bug.
2. Run llama.cpp on the same prompt with `--log-disable` off and
   verbose mode. Capture per-layer intermediate tensors (probably via
   GGML_LOG=) and compare against flambeau's same-shape state. The
   first divergent layer points at the bug.
3. Confirm the head_dim per-layer path: at layer 17 (first full-attn
   layer, head_dim=512), the SWA-layer head_dim_swa=256 attn_q_norm
   shape changes to 512. Make sure the loader picks the right size
   per layer.

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
