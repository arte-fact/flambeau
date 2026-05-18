# v2 forward unification — TopoHooks / StageHooks / ForwardEngine

Tracking: task #226 (umbrella) + #227–#233 (slices).

## Goal

Collapse the four ctx structs (`SingleDeviceForwardCtx`, `TpForwardCtx`,
`PpForwardCtx`, `HybridForwardCtx`) into one generic engine, and unify
decode + prefill at the composite level. The model writes ONE forward
function per arch; it handles all four topologies and N≥1 tokens.

Non-goal: changing kernels. The model-ops layer already has the N-token
primitives (`attn_prefill_f16`, batched `rmsnorm_quant_q8_1`, batched
`kv_append_f16`, m-aware `qmatmul` dispatch). This refactor wires them.

## Current shape (decode-only, N=1)

```
                  ┌─────────────────────────────────────────────┐
model::forward<C> │ ctx.embed → ctx.rmsnorm → ctx.standard_attn │
                  │ → ctx.residual_add → ... → ctx.output_head  │
                  └────────────────┬────────────────────────────┘
                                   │  C: ForwardCtx (trait, 8 methods)
       ┌───────────────────────────┼───────────────────────────┐
       │                           │                           │
┌──────▼──────┐  ┌──────▼──────┐  ┌▼──────────┐  ┌──────────▼──────┐
│SingleDevice │  │  TpForward  │  │ PpForward │  │  HybridForward  │
│ ForwardCtx  │  │     Ctx     │  │    Ctx    │  │      Ctx        │
└──────┬──────┘  └──────┬──────┘  └─────┬─────┘  └─────────┬───────┘
       │                │                │                  │
       └────────────────┴────────┬───────┴──────────────────┘
                                  │
                         ┌────────▼────────┐
                         │ core/composites │ (one _local fn per op,
                         └────────┬────────┘  takes CoreState + Hooks)
                                  │
                         ┌────────▼────────┐
                         │ flambeau-model- │ (leaf kernels)
                         │     ops         │
                         └─────────────────┘
```

Each ctx struct has 8 trait methods, ~5 LOC each, dispatching to the
same composite. SD/TP differ only in `TopologyHooks`. PP/Hybrid add
peer-copy in `embed` and `output_head`. ~700 LOC of pure dispatch.

## Target shape (unified)

```
                  ┌─────────────────────────────────────────────┐
model::forward<C> │ ctx.embed(tokens) → ... → ctx.output_head(n)│ ONE function
                  └────────────────┬────────────────────────────┘ N=1 or N>>1
                                   │  C: ForwardCtx
                                   │
                         ┌─────────▼─────────┐
                         │ ForwardEngine<H,S>│  generic + monomorphised
                         │   H: TopoHooks    │  per (topology, arch)
                         │   S: StageHooks   │
                         └─────────┬─────────┘
                                   │
                         ┌─────────▼─────────┐
                         │  core/composites  │  ONE fn per op,
                         │  (n_tokens-aware) │  branches kernel choice
                         └─────────┬─────────┘  on n_tokens at leaf
                                   │
                         ┌─────────▼─────────┐
                         │ flambeau-model-   │
                         │       ops         │
                         └───────────────────┘

           Typedefs (no new structs):
             type SingleDeviceEngine<'a> = ForwardEngine<'a, NoopHooks, SoloStage>;
             type TpEngine<'a>           = ForwardEngine<'a, TpHooks,   SoloStage>;
             type PpEngine<'a>           = ForwardEngine<'a, NoopHooks, PpStage<'a>>;
             type HybridEngine<'a>       = ForwardEngine<'a, HybridHooks, HybStage>;
```

## The two hook traits

### `TopologyHooks` (already exists — keep)

Intra-stage callback. Differs by topology:

```rust
pub trait TopologyHooks {
    fn ar_sum_f32(&mut self, buf, n_elems, device, stream) -> Result<()>;
}
impl TopologyHooks for NoopHooks    { /* SD + PP — no AR */ }
impl TopologyHooks for TpHooks      { /* AR across n_ranks */ }
impl TopologyHooks for HybridHooks  { /* AR across tp_size of this stage */ }
```

After a row-parallel matmul (output_proj, ffn_down), the composite calls
`hooks.ar_sum_f32(ptr, n_elems, ...)`. The hook reduces across the
ranks owned by the topology — TP across all ranks, Hybrid across the
stage's TP ring, SD/PP no-op. **The hook doesn't care if `n_elems` is
`hidden` (decode) or `n_tokens * hidden` (prefill).**

### `StageHooks` (new)

Inter-stage callback. Differs by stage role:

```rust
pub trait StageHooks {
    fn is_first(&self) -> bool;                      // SD/TP: always
    fn is_last(&self)  -> bool;                      // SD/TP: always
    fn layer_range(&self, layout) -> Range<usize>;   // SD/TP: 0..n_layers
    fn peer_recv(&mut self, core, n_tokens)          // PP rank>0, Hybrid stage>0
        -> Result<Tensor<F16>>;
    fn peer_send(&mut self, core, input, n_tokens)   // PP rank<last, Hybrid stage<last
        -> Result<()>;
}

pub struct SoloStage;                       // SD + TP — never peers
pub struct PpStage<'a> {                    // PP — rank-bounded layer slice + Vec<f16>
    rank, n_ranks, layer_start, layer_end,
    peer_buffer: &'a mut Vec<f16>,
}
pub struct HybStage {                       // Hybrid — stage-bounded slice + Arc<Mutex>
    stage_idx, n_stages, rank_in_stage,
    layer_start, layer_end,
    peer_buffer: Arc<Mutex<Vec<f16>>>,
    handoff_barrier: Arc<Barrier>,
}
```

`peer_recv`/`peer_send` sizes the transfer at `n_tokens * hidden * 2`
bytes and lazy-resizes `peer_buffer` if needed. Same code path for
decode (N=1) and prefill (N>1). Caller bounds N at `prefill_ubatch`.

## The engine

```rust
pub struct ForwardEngine<'a, H: TopologyHooks, S: StageHooks> {
    pub core:  CoreState<'a>,
    pub hooks: H,
    pub stage: S,
}

impl<H, S> ForwardCtx for ForwardEngine<'_, H, S>
where H: TopologyHooks, S: StageHooks {
    fn embed(&mut self, w, tokens: &[u32]) -> Result<Tensor<F16>> {
        if self.stage.is_first() {
            composites::embed_local(&mut self.core, &mut self.hooks, w, tokens)
        } else {
            self.stage.peer_recv(&mut self.core, tokens.len())
        }
    }
    fn output_head(&mut self, x, lm, n_tokens) -> Result<()> {
        if self.stage.is_last() {
            composites::output_head_local(&mut self.core, &mut self.hooks, x, lm, n_tokens)
        } else {
            self.stage.peer_send(&mut self.core, x, n_tokens)
        }
    }
    fn standard_attn(&mut self, x, w, li, pos_start, n) -> Result<Tensor<F16>> {
        composites::standard_attn_local(&mut self.core, &mut self.hooks, x, w, li, pos_start, n)
    }
    /* ...gdn_layer, dense_ffn, moe_ffn, rmsnorm, residual_add: all one-liners */
    fn layer_range<'b>(&'b mut self, layout) -> Box<dyn Iterator<Item=usize>+'b> {
        Box::new(self.stage.layer_range(layout))
    }
    fn logits(&self) -> &[f32] { &self.core.logits_host }
}
```

ONE impl block. Compiler monomorphises per (H, S) pair the workers
instantiate. Zero new ctx structs.

## The ForwardCtx trait surface

Methods become n_tokens-aware. Decode = N=1; prefill = N=tokens.len().

```rust
pub trait ForwardCtx {
    fn embed(&mut self, w: &EmbeddingWeights, tokens: &[u32]) -> Result<Tensor<F16>>;
    fn rmsnorm(&mut self, x, weight, eps, n_tokens) -> Result<Tensor<F16>>;
    fn residual_add(&mut self, a, b, n_tokens) -> Result<Tensor<F16>>;
    fn standard_attn(&mut self, x, w, layer_idx, start_position, n_tokens)
        -> Result<Tensor<F16>>;
    fn gdn_layer(&mut self, x, w, layer_idx, n_tokens) -> Result<Tensor<F16>>;
    fn dense_ffn(&mut self, x, w, n_tokens) -> Result<Tensor<F16>>;
    fn moe_ffn(&mut self, x, w, n_tokens) -> Result<Tensor<F16>>;
    fn output_head(&mut self, x, lm_head, n_tokens) -> Result<()>;
    fn layer_range<'a>(&'a mut self, layout) -> Box<dyn Iterator<Item=usize>+'a>;
    fn logits(&self) -> &[f32];           // last token's logits
}
```

## Composites — kernel branch on n_tokens at the leaf

`standard_attn_local(state, hooks, x, weights, layer_idx, start_pos, n_tokens)`:
- rmsnorm_quant_q8_1 with `n_tokens, hidden`
- qmatmul Q/K/V/output_proj with `m = n_tokens` — dispatch table picks
  MMVQ at m=1 or MMQ at m≥32 (already wired in flambeau-ops).
- cast_f32_to_f16 over `n_tokens * width`
- kv_append_f16 with `n_tokens` rows from `start_pos`
- attention kernel: **branch** — `attn_decode_f16` at N=1 (MMVQ-shaped),
  `attn_prefill_f16` at N>1 (tile-MMQ-shaped). The branch is one if
  inside the composite; trait surface stays unified.
- output_proj qmatmul `m = n_tokens` → AR via hook → cast.

Same pattern for the other composites. `output_head_local` extracts the
LAST token's logits when N>1 (slice `x[(n-1)*hidden..n*hidden]`).

## Model code — same body for decode + prefill

```rust
pub fn forward<C: ForwardCtx>(
    model: &Qwen35V2Model,
    ctx: &mut C,
    tokens: &[u32],
    start_position: usize,
) -> Result<()> {
    let n = tokens.len();
    let mut x = ctx.embed(&model.embedding, tokens)?;
    for li in ctx.layer_range(&model.layout) {
        let kind = model.layer_kinds[li];
        let delta = match kind {
            LayerKind::FullAttn => ctx.standard_attn(
                &x, model.full_attn[li].as_ref().unwrap(), li, start_position, n,
            )?,
            LayerKind::Gdn => ctx.gdn_layer(
                &x, model.gdn[li].as_ref().unwrap(), li, n,
            )?,
        };
        x = ctx.residual_add(x, delta, n)?;
        let delta = ctx.dense_ffn(&x, model.ffn[li].as_ref().unwrap(), n)?;
        x = ctx.residual_add(x, delta, n)?;
    }
    ctx.output_head(&x, &model.lm_head, n)?;
    Ok(())
}
```

`Arch::forward(model, ctx, tokens: &[u32], start_position)`. ONE
method on the trait. `tokens.len() == 1` is decode; longer is prefill.

## Worker / Session

```rust
enum Command {
    Forward { tokens: Vec<u32>, start_position: usize, reply },
    ResetKv { reply },
    Shutdown,
}

impl<A: Arch> Session<A> {
    pub fn forward(&mut self, tokens: &[u32], start_position: usize)
        -> Result<()> { /* dispatch Command::Forward */ }
    pub fn forward_one_token(&mut self, token: u32, position: usize)
        -> Result<()> { self.forward(&[token], position) }
    pub fn forward_prefill_logits(...) -> Result<()> { self.forward(prompt, 0) }
}
```

`orchestrate::run_forward` becomes `run_forward(topology, handles, tokens, start_position)`.
PP/TP/Hybrid dispatch logic unchanged — they already handle Vec<f32>
replies of any size.

## Migration mapping (U2)

| Old                         | New                                      |
|-----------------------------|------------------------------------------|
| `SingleDeviceForwardCtx`    | `SingleDeviceEngine = ForwardEngine<NoopHooks, SoloStage>` |
| `TpForwardCtx`              | `TpEngine          = ForwardEngine<TpHooks,   SoloStage>` |
| `PpForwardCtx`              | `PpEngine          = ForwardEngine<NoopHooks, PpStage>`   |
| `HybridForwardCtx`          | `HybridEngine      = ForwardEngine<HybridHooks, HybStage>`|
| `PpHooks` (empty)           | dropped — `PpEngine` uses `NoopHooks`    |
| `PpForwardCtx::embed`       | `ForwardEngine::embed` + `PpStage::peer_recv` |
| `HybridForwardCtx::output_head` | `ForwardEngine::output_head` + `HybStage::peer_send` |

## Slice plan

| Slice | Task | What lands                                                   |
|-------|------|--------------------------------------------------------------|
| U1    | #227 | `StageHooks` trait + `SoloStage`/`PpStage`/`HybStage` impls + `ForwardEngine` struct + typedefs. Composites unchanged. Old ctxs unchanged. New types compile but unused. |
| U2    | #228 | Replace old ctx structs with typedefs. Worker constructors swap. All decode tests stay green. Net LOC: ~−400. |
| U3    | #229 | Composites take `n_tokens`. Branch on N==1 at leaf-kernel choice. Co-located parity test (N=1 vs N=K bit-identical at composite output). |
| U4    | #230 | `Arch::forward(tokens: &[u32], start_position)`. Per-arch forward functions take a slice. Old `forward_one_token` becomes a thin wrapper or is dropped. |
| U5    | #231 | `Command::Forward { tokens, start_position }`. Session `forward`/`forward_one_token`/`forward_prefill_logits` collapse. |
| U6    | #232 | SD prefill bench — target ≥ legacy 503 tok/s on Qwen3.5-9B-Q4_1 (currently 40.5). |
| U7    | #233 | TP/PP/Hybrid prefill — expect zero new ctx code; PP peer_buffer lazy-resize sufficient. Matrix smoke + long-prompt smoke. |

Each slice is independently committable + testable. Slice boundary
invariants:
- After U1: existing tests still green (new types unused).
- After U2: existing decode tests still green; net LOC down.
- After U3: composite parity tests N=1 ≡ N=K bit-equal.
- After U4: existing decode tests still green; SD long-prompt prefill correct.
- After U5: existing decode + prefill end-to-end via unified path.
- After U6: SD prefill ≥ legacy.
- After U7: TP/PP/Hybrid prefill ≥ legacy or documented gap.
