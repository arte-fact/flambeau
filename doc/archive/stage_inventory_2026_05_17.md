# Stage struct inventory — qwen3-moe × {PP,TP,Hybrid} + gemma4 × {PP,TP,Hybrid} (B1)

Side-by-side audit of every per-rank stage struct across both arches.
Drives the shape of `flambeau_blocks::ModelStageCommon<W, K>` (B2).

## The six structs

| arch | topology | struct | file | concern split |
|---|---|---|---|---|
| qwen3-moe | PP | `Qwen3MoERankShard` | `sharded.rs:60` | **weights only** (KV lives on `Qwen3MoEShardedSession`) |
| qwen3-moe | TP | `Qwen3MoETpRankShard` | `tp_sharded.rs:75` | **weights only** (KV on `Qwen3MoETpSession`) |
| qwen3-moe | Hybrid | `Qwen3MoEHybridStage` | `hybrid.rs:112` | **wraps `Qwen3MoETpModel`** (weights only); KV on `Qwen3MoEHybridSession` |
| gemma4 | PP | `Gemma4PpStage` | `pp.rs:142` | **weights + KV + scratch** bundled |
| gemma4 | TP | `Gemma4TpStage` | `tp.rs:59` | **weights + KV + scratch** bundled |
| gemma4 | Hybrid | `HybridRankState` | `hybrid.rs:53` | **weights + KV + scratch** bundled |

Immediate observation: **qwen3-moe already split weights/session.**
The Stage structs that the splits produced are WEIGHTS-ONLY. KV+scratch
live elsewhere (`Session` + scratch fields on the `Inflight` wrappers in
server-side `model.rs`).

Gemma4 hasn't done that split yet. Its three Stage structs carry
**everything** in one bundle.

## Field-by-field inventory

### Bookkeeping fields (present in 6 of 6)

| field | qwen-PP | qwen-TP | qwen-Hyb | gem-PP | gem-TP | gem-Hyb |
|---|---|---|---|---|---|---|
| rank identifier | `rank: RankId` | `rank: RankId` | (via tp_model) | `rank: usize` | `rank: usize` | `rank_in_stage: usize` |
| device id | `device_id: i32` | `device_id: i32` | (via sub_cluster) | implicit | implicit | implicit |
| `total_bytes` budget | ✓ | ✓ | (in tp_model) | – | – | – |
| `disposed: bool` | ✓ | ✓ | (in tp_model) | ✓ | ✓ | ✓ |
| `RawAllocTracker` | – | – | – | ✓ | ✓ | ✓ |

### Weights fields (present in 6 of 6, shape varies)

| field | qwen-PP | qwen-TP | qwen-Hyb | gem-PP | gem-TP | gem-Hyb |
|---|---|---|---|---|---|---|
| per-layer weights | `Vec<LayerWeights>` | `Vec<Vec<TpLayerTensor>>` | (in tp_model) | `Vec<Gemma4LayerWeights>` | `Vec<Gemma4LayerWeights>` | `Vec<Gemma4LayerWeights>` |
| globals: token_embd | `Option<DeviceTensor>` | `DeviceTensor` | (in tp_model) | `Option<DeviceTensor>` | `DeviceTensor` | `Option<DeviceTensor>` |
| globals: output_norm | `Option<DeviceTensor>` | `DeviceTensor` | (in tp_model) | `Option<DeviceTensor>` | `DeviceTensor` | `Option<DeviceTensor>` |
| globals: lm_head | `Option<DeviceTensor>` | `Option<DeviceTensor>` | (in tp_model) | `Option<DeviceTensor>` | `Option<DeviceTensor>` | `Option<DeviceTensor>` |
| OpsRegistry | `ops: OpsRegistry` | (on outer `Qwen3MoETpModel.ops: Vec`) | (in tp_model) | (on outer `Gemma4PpDriver.regs`) | (on outer `Gemma4TpDriver.regs`) | (on outer `Gemma4HybridStage.regs`) |

### KV / session fields (present 0 of qwen-3, 3 of gemma4-3)

This is the split that doesn't exist yet for gemma4.

| field | qwen-PP | qwen-TP | qwen-Hyb | gem-PP | gem-TP | gem-Hyb |
|---|---|---|---|---|---|---|
| `kv_caches: Vec<Option<KvCache<F16Contig>>>` | – (Session) | – (Session) | – (Session) | ✓ | ✓ | ✓ |
| Scratch ptrs (`q_f16`, `k_f16`, `v_f16`, `mmvq_f32`, `x_q8_1`, `splitk_partials_*`, `positions`, …) | – (ScratchTopology) | – | – | `LayerScratchPtrs` | `TpScratchPtrs` | `HybridScratchPtrs` |
| `hidden_a` / `hidden` ping-pong | – | – | – | `hidden_a` + `hidden_b` (PP only) | `hidden` (single) | `hidden` |
| `partial_attn{,_f32}`, `partial_ffn` (TP-style) | – | – | – | – | ✓ | ✓ |
| `attn_normed_f32_tmp` | – | – | – | – | ✓ | ✓ |
| `output_head_scratch: Option<OutputHeadScratch>` | – | – | – | ✓ (last rank) | ✓ (head rank) | ✓ (head rank) |
| `tp_moe_scratch: Option<Gemma4TpMoeScratch>` | – | – | – | (PP-MoE: separate `moe_scratch`) | ✓ | ✓ |
| `core: TpRankCore` (TP sync events) | – | (Session) | (Session) | – | ✓ | ✓ |
| `positions_host: Vec<i32>` | – | – | – | ✓ | ✓ | ✓ |
| `max_tokens` (KV cap) | – | – | – | ✓ | ✓ | ✓ |

## What the shared shape is

Reading the table from top to bottom, the **truly shared 6-of-6 fields**
boil down to:

1. **A rank identifier** (`RankId` or `usize` — gemma4 uses raw usize,
   qwen uses the typed `RankId` newtype).
2. **A device id** (i32) — gemma4 stores it implicitly through cluster
   lookups; qwen stores it explicitly.
3. **Per-layer weights** (`Vec<W>` where `W` is the arch's
   `LayerWeights` type).
4. **Three optional globals**: token_embd, output_norm, lm_head, each
   `Option<DeviceTensor>` (TP variants are non-`Option<>` because every
   rank has the full replica — that's a layout choice, not a shape
   difference).
5. **A `disposed: bool` + Drop guard** + dispose impl that frees the
   layer weights and globals.
6. **A `RawAllocTracker`** OR a fixed bookkeeping pattern (qwen
   freed-via-`DeviceTensor.bytes`; gemma4 freed-via-tracker).

Everything else is split: KV+scratch only on gemma4 today (will move
to Session after B4), TP-specific fields (`core`, `partial_attn`,
`tp_moe_scratch`) on TP/Hybrid only.

## The actual abstraction (StageCommon)

The cleanest shared shape — composed-into, not inherited-from — is:

```rust
// flambeau-blocks::model_stage
pub struct StageCommon<W> {
    pub rank: RankId,
    pub device_id: i32,
    pub layer_weights: Vec<W>,
    pub token_embd: Option<DeviceTensor>,
    pub output_norm: Option<DeviceTensor>,
    pub lm_head: Option<DeviceTensor>,
    pub total_bytes: usize,
    pub raw_alloc: RawAllocTracker,
    disposed: bool,
}

impl<W: HasDeviceTensors> StageCommon<W> {
    pub fn dispose(&mut self, device: &HipDevice) -> Result<()> { ... }
    pub fn warn_on_leak(&self, tag: &'static str) { ... }
}
```

Where `HasDeviceTensors` is a small trait — `fn for_each_tensor(&mut self, f: impl FnMut(&mut DeviceTensor))` — that each arch's `LayerWeights` implements once (gemma4 has 6-12 tensors per layer, qwen3-moe has 8-20 depending on dense/MoE).

That single trait method replaces the ~150 LOC of `dispose` bodies in
each of the 6 structs.

## What stays out of StageCommon (per-topology / per-arch extension)

Each outer `Stage` struct gains an embedded `pub common: StageCommon<W>`
and keeps its own extension fields:

- **Gemma4PpStage** extension: hidden_a/b, kv_caches, LayerScratchPtrs,
  output_head_scratch, moe_scratch, prefill scratch, positions_host,
  max_tokens, local_kv_share_src, global_layer_indices
- **Gemma4TpStage** extension: hidden, partial_attn, partial_attn_f32,
  attn_normed_f32_tmp, partial_ffn, TpScratchPtrs, output_head_scratch,
  tp_moe_scratch, core, kv_caches, positions_host, post_attention_norm_f32
- **HybridRankState** extension: same as TP minus a few + rank_in_stage
- **Qwen3MoERankShard** extension: ops (will move to outer model once
  qwen Phase C also refactors per-arch dispatch)
- **Qwen3MoETpRankShard** extension: layers stored as `Vec<Vec<TpLayerTensor>>`
- **Qwen3MoEHybridStage** extension: stage_idx, layer_range,
  sub_cluster, tp_model

## The split that Phase B is doing

For gemma4, the per-rank state needs to be split into two pieces:

```
Gemma4<*>Stage  (today)         →   Gemma4<*>Model         (weights — Arc-shared)
                                   + Gemma4<*>Session      (KV + scratch — per-request)
```

The Model side gets the StageCommon. The Session side gets all the
gemma4-specific scratch fields + KV caches.

```rust
// gemma4/src/pp.rs (after B4)
pub struct Gemma4PpModel {
    pub common: StageCommon<Gemma4LayerWeights>,
    pub cfg: Gemma4Config,
    pub layout: ModelLayout,
    // ... per-topology globals
}

pub struct Gemma4PpSession {
    pub kv_caches: Vec<Option<KvCache<F16Contig, HipDevice>>>,
    pub scratch: LayerScratchPtrs,
    pub hidden_a: DevicePtr,
    pub hidden_b: DevicePtr,
    pub output_head_scratch: Option<OutputHeadScratch>,
    pub moe_scratch: Option<Gemma4MoeScratch>,
    pub prefill: PrefillScratchPtrs,
    pub max_tokens: usize,
    pub positions_host: Vec<i32>,
    pub raw_alloc: RawAllocTracker,
    disposed: bool,
}
```

Qwen3-moe Shards become:

```rust
// qwen3-moe/src/sharded.rs (after B3)
pub struct Qwen3MoERankShard {
    pub common: StageCommon<LayerWeights>,
    pub ops: OpsRegistry, // stays per-shard for now
}
```

The dispose body — currently ~150 LOC per Stage — collapses to:

```rust
impl Qwen3MoERankShard {
    pub fn dispose(self, device: &HipDevice) -> Result<()> {
        self.common.dispose(device)
    }
}
```

## Phase B2 next steps

1. Define `HasDeviceTensors` trait in `flambeau_blocks`.
2. Define `StageCommon<W: HasDeviceTensors>` + `dispose` + `warn_on_leak`.
3. Add a thin `flambeau-blocks::ModelStageCommon` re-export so model
   crates can `use flambeau_blocks::StageCommon`.

`LayerWeights` impls of `HasDeviceTensors` follow in B3 (qwen3-moe
validation) and B4-B6 (gemma4 adoption).

## Anti-claims (CLAUDE.md rule 14 check)

This abstraction is justified by **6 existing sites duplicating ~80%
of dispose body + bookkeeping fields**, with a 7th site (mistral) on
the roadmap. The shape is composed into each Stage, not inherited
from. Stage-specific fields stay concrete. The trait surface is one
method (`for_each_tensor`). No arch-leaky type parameters.

If `HasDeviceTensors` impls in B3 reveal awkward fits (e.g. some
`LayerWeights` have nested `Option<...>` chains that don't tour
cleanly), the abstraction gets revised BEFORE gemma4 adopts it in B4.
