# Topology axis audit — PP / TP / Hybrid duplication (2026-05-17)

Follows `doc/arch_state_2026_05_17.md`. The previous audit identified
arch-axis duplication (qwen3-moe ↔ gemma4) and proposed a 4-phase
mutualisation pass. Phase B was originally three sequential tasks
("split gemma4 PP weights/session", "split gemma4 TP …", "split
gemma4 Hybrid …") — but those three tasks would replicate the same
weights-vs-session split three times in three driver files, which is
itself the same duplication anti-pattern. This audit looks at what's
actually shared / could be shared across the PP/TP/Hybrid axis BEFORE
committing to a 3× split.

## Inventory

### LOC per topology

| file | LOC |
|---|---:|
| `models/gemma4/src/pp.rs` | 1 496 |
| `models/gemma4/src/tp.rs` | 1 645 |
| `models/gemma4/src/hybrid.rs` | 1 583 |
| **gemma4 total** | **4 724** |
| `models/qwen3-moe/src/sharded.rs` (PP) | 1 661 |
| `models/qwen3-moe/src/tp_sharded.rs` (TP) | 1 528 |
| `models/qwen3-moe/src/hybrid.rs` | 732 |
| `models/qwen3-moe/src/forward/pp.rs` | 1 892 |
| `models/qwen3-moe/src/forward/tp.rs` | 3 042 |
| `models/qwen3-moe/src/forward/hybrid.rs` | 1 686 |
| **qwen3-moe total** | **10 541** |

15 265 LOC across 9 files for `qwen3-moe × {pp,tp,hyb}` and
`gemma4 × {pp,tp,hyb}`.

### What's already mutualised (`flambeau-blocks::topology`)

- `pub trait PpDecodeDriver` + `pub fn forward_one_token_pp<D>`
- `pub trait TpDecodeDriver` + `pub fn forward_one_token_tp<D>`
- `pub trait HybridDecodeDriver` + `pub fn forward_one_token_hybrid<D>`
- Three sibling prefill orchestrators + traits.

Both arches implement these driver traits. The **per-token rank loop
+ peer-copy + output-head + argmax orchestration is already shared**.

### What's duplicated per topology within each arch

| concern | PP | TP | Hybrid | shared? |
|---|---|---|---|---|
| Cluster handle type | `HipCluster` | `TpCluster` (wraps cluster + `BarP2pAllReduce`) | `HybridCluster` (sub-clusters + global) | already in `blocks::driver_base` |
| Stage struct (per-rank weights + KV + scratch) | `Gemma4PpStage` (148 LOC) | `Gemma4TpStage` (62 LOC) | `HybridRankState` + `Gemma4HybridStage` (55 LOC) | shape ~identical, **fields ~identical**, no shared type |
| Scratch struct fields | `PpScratchPtrs` | `TpScratchPtrs` | `HybridScratchPtrs` | ~95% identical (q_f16, k_f16, v_f16, mmvq_f32, x_q8_1, …); **could be shared** |
| `dispose(device)` | 30 LOC | 30 LOC | 30 LOC | ~identical — `RawAllocTracker::dispose` + free per-layer KVs |
| `Drop` warn-on-leak | identical | identical | identical | trivially shared |
| Weight upload per layer | `upload_layer_pp` | `upload_layer_tp` (already `pub(crate)`) | reuses `upload_layer_tp` | partly shared via typed `WeightRole` already |
| Globals upload (token_embd, output_norm, lm_head) | inline in each | inline | inline | identical shape, **could be shared helper** |
| Cluster construction in `upload` | plain `HipCluster::new` | wraps in `Arc` + `TpCluster::from_arc` | sub-clusters before global per `hybrid_cluster_order` memory | already in `blocks::driver_base { TpCluster, HybridCluster }` |

### What's irreducibly topology-specific

- `forward_layer_decode` body — PP/TP/Hybrid forward kernel choices,
  AllReduce/peer-copy placement, and intra-layer composition are
  genuinely different. **This is the bulk of the LOC and CAN'T be
  merged.** Roughly:
  - PP: serial single-rank execution; peer-copy between stages
  - TP: every rank runs every layer; `tp_allreduce_sum*` after attn
    and FFN
  - Hybrid: per-stage TP composition + inter-stage peer-copy

- Weight upload `WeightLayout` per role: `Replicated` (PP, head
  rank) / `ColParallel` (TP-sharded weights, dim 0) / `RowParallel`
  (TP-sharded output projections, dim 1) / `Local` (KV caches,
  per-rank-local). These ARE already abstracted via
  `flambeau-blocks::weight_roles::WeightRole`, but the call sites
  duplicate the per-topology helper closures.

## What this means for Phase B

Phase B was 3 tasks: split each topology's `Gemma4*Driver` into
`Gemma4*Model` (weights) + `Gemma4*Session` (KV + scratch). The
"triple job" concern: each split file is structurally identical
(80% of the LOC is dispose / Stage shape / scratch alloc) and 20%
topology-specific.

### Option 1 — Do the 3× split anyway (the originally-planned shape)

- ~3 × 500 LOC of new code (one Gemma4*Model + Gemma4*Session per topo)
- ~3 × 200 LOC of dispose / Drop boilerplate (identical)
- Unblocks multi-slot inflight pool and proper `reset_for_next_request`
- Doesn't introduce abstractions; consistent with the qwen3-moe shape
  which is already this way (and works)

### Option 2 — Add a generic `<W,K,S>::Stage` skeleton in `flambeau-blocks` first

- Define `pub struct ModelStage<W, K, S, Topo>` with shared fields
  (rank id, device id, weights, kv_caches, scratch, RawAllocTracker)
  and a shared `dispose(device: &HipDevice)` impl
- Gemma4 (and qwen3-moe later) parameterise on `(W=Gemma4LayerWeights,
  K=KvCache<F16Contig>, S=Gemma4{Pp,Tp,Hybrid}Scratch)`
- Each topology still owns its forward body, but the shared
  scaffolding stops repeating

Tradeoff: introduces generics that span both arches, risks the
"premature abstraction" anti-pattern CLAUDE.md warns about ("Three
similar lines is better than a premature abstraction"). The actual
duplication isn't quite three identical pieces — each Stage has
topology-specific knobs (Hybrid has stage_idx, TP has rank within
sub_cluster, etc.) so the generic ends up with conditional fields.

### Option 3 — Mutualise just the scratch + dispose (no full Stage generic)

Push the **shared scratch struct shape** + the **dispose pattern** into
`flambeau-blocks`. Today the 9 Driver files each carry a near-
identical scratch struct (`q_f16, k_f16, v_f16, mmvq_f32, x_q8_1,
splitk_partials_m/s/o, ...`). The differences are:
- TP/Hybrid: locally-sized (`q_width / n_ranks`, etc.)
- Hybrid: also `tp_moe_scratch`, `partial_attn_f32`
- PP: no AR scratch

A single `StandardAttnDecodeScratchPtrs` + topology-tailored
extensions (a struct that *contains* the shared part) gives 80% of
the dedup at minimal abstraction cost.

## Recommendation — REVISED (rule-of-three applied)

The original recommendation said "do 3× split, mutualise post-hoc"
citing the "three similar lines is better than premature abstraction"
maxim. **That was wrong here**: the maxim applies to small repetitions
(three string formats, three trivial match arms), not to 100-LOC
Stage structs duplicated across 6 sites. The classic rule of three
says **two is acceptable, three is the refactor signal**. At 6
existing sites (qwen × 3 + gemma4 × 3) and 9 incoming with the next
arch (mistral / qwen3-coder-next), copy-paste is past the threshold.

CLAUDE.md rule 14 now codifies this.

### Revised plan

1. **B1** Side-by-side inventory of all 6 existing Stage structs.
   Identify the actual shared shape (~80% common: rank id, device id,
   `Vec<W>` weights, `Vec<Option<K>>` KV caches, `RawAllocTracker`,
   `disposed` flag, dispose body, Drop warn-on-leak).
2. **B2** Extract `flambeau_blocks::ModelStageCommon<W, K>` (or
   similar) with the shared fields + `dispose(device)` impl.
   Composition over inheritance — each topology's `Stage` *contains*
   a `StageCommon` rather than re-declaring its fields.
3. **B3** Refactor qwen3-moe's existing 3 Stage structs onto
   `StageCommon`. qwen3-moe is the most-tested existing arch; if the
   abstraction doesn't fit cleanly here, revise BEFORE gemma4
   adopts it.
4. **B4** Gemma4 PP weights/session split using `StageCommon`.
   Single source of truth from day one.
5. **B5** Gemma4 TP using `StageCommon`.
6. **B6** Gemma4 Hybrid using `StageCommon`.
7. **B7** Inflight pool ≥ 1 + working `reset_for_next_request`.

### Why composition over a `Stage<W, K, S, Topo>` generic

A wrapping generic with a `Topo` type parameter would force every
shared method to know its topology specifics (cluster type, AllReduce
helper, peer-copy shape). That's the "closed enum disguised as a
trait" anti-pattern CLAUDE.md rule 12 warns about.

Composition keeps the topology specifics in the outer struct
(per-topology fields like `stage_idx`, `core: TpRankCore`,
`tp_moe_scratch: Option<Gemma4TpMoeScratch>`, etc.) and pulls only
the shared 80% into `StageCommon`. Adding a new topology adds a new
outer struct that re-uses `StageCommon`; no trait/generic surface
changes.

### Risk mitigation

- B3 validates the abstraction on **existing working code with the
  best test coverage** before locking gemma4 in. If `StageCommon`
  needs revision, it happens there, not in the middle of a gemma4
  splitting session.
- Composition (not inheritance / not generics) keeps the coupling
  low. Each topology's outer struct remains a concrete type with
  topology-specific methods.

## What this audit does NOT propose

- ❌ A generic `Driver<Topo>` with topology trait — the per-topology
  forward bodies are too different; would need a 20-method trait.
- ❌ Merging PP / TP / Hybrid forward files — the AllReduce
  placement is genuinely different per topology.
- ❌ Sharing scratch structs across qwen3-moe AND gemma4 — those
  have different per-layer composition (GDN vs SWA, top-K MoE
  shape) and the scratches reflect that.
- ❌ Moving topology files into `blocks/` — the topology
  orchestrators are already there (`PpDecodeDriver` etc.); the
  arch-specific driver bodies belong in the model crates per
  CLAUDE.md rule 4.

The "topology mutualisation" win is bounded. Most of the 15k LOC is
arch×topology-specific by nature. The 5-10% that's reducible
(dispose / Drop / common scratch fields) is best harvested AFTER
the concrete splits land, with the duplication visible side-by-side.
