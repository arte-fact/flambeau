# gemma4-v2 E4B / E2B (per-layer-embd) — handoff for #256

Date: 2026-05-19
Status: not started. Config still bails at boot with
`PerLayerEmbdUnsupported`.

## Why this is its own session

E4B / E2B (gemma 4n) adds a token-derived F32 side-channel embedding
that gets mixed into the residual stream after every layer's FFN
post-norm. This is a separate architectural pathway from the main
residual stream and adds:

- 3 new global tensors (`per_layer_token_embd` Q5_K [vocab, pe*n_layer],
  `per_layer_model_proj` BF16 [pe*n_layer, hidden],
  `per_layer_proj_norm` F32 [pe])
- 3 new per-layer tensors (`blk.N.inp_gate` F32 [pe, hidden],
  `blk.N.proj` F32 [hidden, pe], `blk.N.post_norm` F32 [hidden])
- A host-side per-token setup pass (Q5_K dequant + BF16 matmul +
  rmsnorm + add) producing `inp_per_layer[pe × n_layer]` uploaded to
  device once per token
- A per-layer apply block after the FFN residual add: `inp_gate
  projection → GELU → elemwise mul with inp_per_layer[layer] → proj
  back → rmsnorm → residual add`

Boot-then-skip-the-block is not a valid shortcut: the per-layer
contribution is load-bearing for output coherence — without it the
model produces garbage tokens.

## Existing infrastructure to reuse

Most pieces already exist in the legacy gemma4 + blocks crates:

- `crates/blocks/src/per_layer_embd.rs`:
  - `PerLayerEmbedLayerWeights { inp_gate, proj, post_norm_f16 }`
  - `PerLayerEmbedDecodeScratch { gate_out_f32, activated_f32,
    activated_f16, proj_out_f32, proj_out_f16, normed_f16 }`
  - `PerLayerEmbedBlock::forward_decode` — the per-layer apply
    (`dense_gemv_f32_f16 → gelu_mul_f32 → cast_f32_to_f16 →
    dense_gemv_f32_f16 → cast_f32_to_f16 → rmsnorm_f16 → add_f16`)
  - `table_slice_ptr(table_base, il, pe)` helper
- `crates/models/gemma4/src/per_layer_embd.rs`:
  - `build_inp_per_layer_table` — host-side Q5_K dequant + BF16 matmul
    + rmsnorm + add, emits the `[n_layer × pe]` F32 table
  - `upload_inp_per_layer_table` — single HtoD copy of the table

Neither side-channel apply nor the host build needs new kernel work.
The legacy code path stalls because `Gemma4PpDriver::upload` bails
when `cfg.per_layer_embed.is_some()` (pp.rs:816) — i.e. legacy never
wired the upload either. v2 has the same gap.

## Scope for #256 (concrete LOC estimates)

1. **Config** (~20 LOC, `gemma4-v2/src/config.rs`):
   - Add `PerLayerEmbdDims { per_layer: usize }` struct + field on
     `Gemma4V2Config`.
   - Parse `gemma4.embedding_length_per_layer_input`; replace the
     `PerLayerEmbdUnsupported` bail with `Some(…)` / `None`.

2. **Model struct fields** (~30 LOC, `gemma4-v2/src/loader.rs`):
   - `per_layer_weights: Vec<Option<PerLayerEmbedLayerWeights>>`
   - `per_layer_globals: Option<PerLayerEmbedGlobals>` storing
     `per_layer_token_embd_raw: &[u8]` (mmap slice — careful with
     `advise_drop_tensor`), `per_layer_model_proj_raw: &[u8]`,
     `per_layer_proj_norm_raw: &[u8]`, plus the on-device
     `per_layer_table_dev: DevicePtr` of size `pe * n_layer * 4` for
     the uploaded table.

3. **Loader per-layer branch** (~120 LOC, `gemma4-v2/src/loader.rs`):
   - When `config.per_layer_embd.is_some()`, after the FFN load, load
     three additional tensors per layer:
     - `blk.N.inp_gate.weight` F32 [pe, hidden] → device F32
     - `blk.N.proj.weight` F32 [hidden, pe] → device F32
     - `blk.N.post_norm.weight` F32 [hidden] → cast to F16 on device
   - Allocate `per_layer_table_dev` (one buffer, sized for all layers).
   - Cache the three global tensor raw mmap slices in
     `per_layer_globals` (don't `advise_drop_tensor` them — they're
     touched on every token).

4. **Scratch pool extension** (~30 LOC, `forward/src/core/scratch.rs`):
   - Six new fields for the per-layer-embd decode scratch buffers
     sized as `max(pe, hidden) * f32 / f16`.
   - Conditional allocation only when the model exposes per_layer dims
     (gate by `config.per_layer > 0`).

5. **ForwardCtx method** (~80 LOC across `forward/src/ctx.rs`,
   `forward/src/engine.rs`, `forward/src/testing.rs`):
   - `fn per_layer_embd_apply(&mut self, pe_in: &Tensor<F16>,
     weights: &PerLayerEmbedLayerWeights, table_slice_offset: usize,
     hidden: usize, pe: usize, layer_idx: usize) -> Result<()>`
   - Engine impl: delegate to `PerLayerEmbedBlock::forward_decode`,
     passing the per-layer scratch from the pool and the `table_base
     + table_slice_offset` device pointer.
   - SD / PP impls behave the same (no AR). TP/Hybrid: AR over the
     `proj` output before the residual add (n_elems = hidden) — the
     existing `bar_ar_residual_rmsnorm` shape is the right tool.

6. **Per-token forward setup** (~60 LOC, new helper in `gemma4-v2/`):
   - At the start of `forward()` (decode path, n_tokens = 1): host-side
     dequantise the main-stream token embedding to F16, call
     `build_inp_per_layer_table` from legacy gemma4 (move the helper
     to a shared crate, e.g. `flambeau-blocks::per_layer_embd::build`,
     since both legacy and v2 will use it), then
     `upload_inp_per_layer_table` to the pool's `per_layer_table_dev`.
   - Track per-slot table buffers if `INFLIGHT_SLOTS > 1`.

7. **Model.rs integration** (~15 LOC, `gemma4-v2/src/model.rs`):
   - After the FFN residual add and before the `layer_output_scale`
     multiply, call `ctx.per_layer_embd_apply` when
     `model.per_layer_weights[li].is_some()`.

## Validation plan

- Real-prompt coherence check: boot E4B, send "The capital of France
  is" — must return tokens including "Paris" within the first 8
  decode tokens.
- Bench shape: `python3 scripts/bench/gemma4_v2_vs_legacy_vs_llamacpp.py
  --only gemma4-E4B-Q4_0 --tg 64`. Target ≥ 0.7× of llama.cpp's
  71.45 t/s (so ≥ 50 t/s decode). Prefill llama.cpp baseline is
  1055 t/s; v2 small-model prefill historically beats llama.cpp by
  1.5–2× (see 31B-Q4_0 TP2 in
  `gemma4_v2_post_levers_status_2026_05_19.md`).

## Out (deferred further)

- E4B prefill batching (n_tokens > 1) — the host build helper
  currently assumes n_tokens = 1. Generalising it means batching the
  Q5_K dequant + BF16 matmul on the CPU per token, or porting both
  steps to GPU. For first-ship, fall back to per-token sequential
  build during prefill (slow but correct).
- The shared-KV layers (E4B has `gemma4.attention.shared_kv_layers
  = 18`) — the tail-N layers share KV with another layer. Already
  noted as `has_attn_v` probe in the loader; the forward needs to
  route the KV writes to the source layer's cache. Likely a small
  extra change once the per-layer-embd path is in.

## Recap

This is ~360 LOC of model-glue work, no new kernels. The blocking
piece is moving `build_inp_per_layer_table` from legacy gemma4 to a
shared module so v2 can call it, plus integrating an existing block
(`PerLayerEmbedBlock`) into the gemma4-v2 forward path. Roughly
2–3 sessions of focused work depending on debugging the host build
side (Q5_K dequant in particular can be finicky to verify against
llama.cpp's reference).
