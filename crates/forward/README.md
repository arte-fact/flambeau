# flambeau-forward

Topology executor for model forward passes. Lets a model write **one**
forward function — generic over a `ForwardCtx` trait — that runs
unchanged under PP / TP / Hybrid. The topology dimension collapses
into the executor; arch code stops repeating itself.

## What this fixes

Today, each arch crate writes its forward path three times — once for
PP, once for TP, once for Hybrid — each open-coding `rmsnorm → attn →
AR → norm → ffn → AR → ...` with topology-specific AllReduce
placement, peer-copy, and handoff. That's the bulk of the 10–22 kLOC
per arch (~35–40× llama.cpp's arch files).

With this crate + `flambeau-model-ops`, an arch becomes:

```rust
// crates/models/qwen35-v2/src/forward.rs   (~ 200-400 LOC total)
pub fn forward_one_token<C: ForwardCtx>(
    model: &Model,
    ctx: &mut C,
    token: u32,
    position: usize,
) -> Result<u32> {
    let mut x = ctx.embed(&model.token_embd, token)?;
    for layer_idx in ctx.layer_range(&model.layout) {
        let layer = &model.layers[layer_idx];
        let an = ctx.rmsnorm(&x, &layer.attn_norm)?;
        let a  = ctx.standard_attn(&an, &layer.attn, layer_idx, position)?;
        x = ctx.residual_add(x, a)?;
        let fn_ = ctx.rmsnorm(&x, &layer.ffn_norm)?;
        let f  = ctx.dense_ffn(&fn_, &layer.ffn)?;
        x = ctx.residual_add(x, f)?;
    }
    let out = ctx.rmsnorm(&x, &model.output_norm)?;
    ctx.output_head(&out, &model.lm_head)
}
```

That same function runs under any of `PpForwardCtx`, `TpForwardCtx`,
`HybridForwardCtx`. Topology-specific machinery — AR after row-
parallel ops, peer-copy between PP stages, intra-stage TP composition
inside hybrid — lives inside the corresponding `impl ForwardCtx for
*ForwardCtx`, never in the model.

## Layout

```
crates/forward/
├── Cargo.toml
├── README.md
├── CLAUDE.md
└── src/
    ├── lib.rs         // re-exports the public surface
    ├── ctx.rs         // ForwardCtx trait — composite-op contract
    ├── pp.rs          // PpForwardCtx — pipeline-parallel impl
    ├── tp.rs          // TpForwardCtx — tensor-parallel impl
    ├── hybrid.rs      // HybridForwardCtx — PP-of-TP impl
    ├── layer_range.rs // shared layer-iteration helpers
    └── testing.rs     // RecordingCtx — mock for unit-testing model
                       // forward functions without a device
```

## ForwardCtx — the trait

`ForwardCtx` is the contract that every topology implements. It carries
no "is this PP?" predicates and no `as_*` downcasts (rule 12 of the
project CLAUDE.md). Methods are the **composite ops** the model needs
— each one knows how to do its job under the topology that owns the
context.

Composite ops (the trait's surface):

- `embed(token_embd, token_id) -> Tensor<F16>`
- `rmsnorm(input, weight) -> Tensor<F16>`
- `residual_add(a, b) -> Tensor<F16>`
- `standard_attn(input, weights, layer_idx, position) -> Tensor<F16>`
- `dense_ffn(input, weights) -> Tensor<F16>`
- `moe_ffn(input, weights) -> Tensor<F16>`
- `output_head(input, lm_head) -> ()` (writes logits to ctx's slot)
- `layer_range(layout) -> impl Iterator<Item = usize>` (per-rank layer indices)

Each composite calls into `flambeau-model-ops` leaf primitives. The
topology insertions (`tp_sum` after `standard_attn`'s output proj,
`peer_copy` at the end of `layer_range`'s last item on a stage
boundary, etc.) happen inside the composite implementation.

## Adding a new composite

If a model needs a primitive that isn't on `ForwardCtx` yet:

1. Add the method to the `ForwardCtx` trait in `ctx.rs`.
2. Implement it on **all three** topology contexts. No "TP only"
   methods on the trait — that's the closed-enum-disguised-as-a-trait
   anti-pattern (rule 12).
3. If the new composite needs leaf ops that don't exist yet, those go
   in `flambeau-model-ops` first, with their own tests.
4. Add a topology-parity test (run the same model forward against
   real device under each of the three ctxs and confirm bit-equal
   outputs on a tiny synthetic model).

## Testing

Two layers of tests live here.

### Unit (RecordingCtx, no device)

`testing.rs` exposes a `RecordingCtx` that implements `ForwardCtx` by
appending each call to a `Vec<OpCall>`. Used to assert "this model
forward function calls embed → rmsnorm → standard_attn → ... in the
right order with the right layer indices" — fast, no device required.

```rust
#[test]
fn qwen35_forward_op_sequence() {
    let model = synthetic_qwen35_model();
    let mut ctx = RecordingCtx::pp(&model.layout, /*rank=*/0, /*n_ranks=*/2);
    qwen35_v2::forward_one_token(&model, &mut ctx, /*token=*/42, /*pos=*/0).unwrap();
    assert_eq!(ctx.ops_called.first().unwrap(), &OpCall::Embed { token: 42 });
    // ... etc
}
```

### Topology parity (real device)

Per topology, a tiny synthetic model is forwarded through `PpForwardCtx`,
`TpForwardCtx`, `HybridForwardCtx` and the outputs compared bit-for-bit
(or within F16 tolerance). The same model forward function is called
unchanged; the only difference is which ctx is constructed.

## Relationship to other crates

- **flambeau-model-ops** — leaf op vocabulary. Every composite in this
  crate eventually bottoms out in a model-ops free function. We
  depend on it; we do NOT bypass it to call kernels directly.
- **flambeau-ops** — kernel-launch wrappers. model-ops calls into
  this; flambeau-forward never depends on it directly.
- **flambeau-model-ops** — the OLD topology orchestration layer. Lives in
  parallel; deleted when models-v2 cuts over.
- **crates/models/*-v2** — model crates that use this. Each ~300–500
  LOC. Generic over `<C: ForwardCtx>`.

## Open design issue — to decide before composites land

**Scratch lifetime.** When a composite returns `Tensor<F16>`, the
underlying device memory is owned by ctx (per-request scratch pool).
Three viable shapes:

1. Ctx owns a per-layer scratch pool; returned tensors are valid until
   end of layer iteration. Lifetime annotated as `Tensor<'a, F16>`.
2. Returned `Tensor` is a token/handle into ctx's pool; later ops that
   take it as arg resolve through ctx.
3. Caller passes the output buffer in (`ctx.rmsnorm(input, weight,
   output)`) — no ctx scratch alloc.

Option 1 is what the README example shows. Option 3 makes the model
code more verbose but eliminates lifetime gymnastics. Decide during
the first composite impl pass; do not pre-empt.
