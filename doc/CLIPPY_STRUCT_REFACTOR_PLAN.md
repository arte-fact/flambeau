# `too_many_arguments` → struct-aggregate refactor

**Goal.** Eliminate every `clippy::too_many_arguments` warning in the workspace
without using a single `#![allow(...)]` or `#[allow(...)]`. Every flagged
kernel-launch wrapper, trait method, mid-layer driver, and helper drops to
≤ 6 args by grouping its parameters into small `Copy + Clone + Debug`
POD structs that carry the existing semantic blocks (launch ctx, buffers,
shape, knobs).

**Constraint.** No lint suppression anywhere — pure Rust refactor only.

**Starting state.** 315 `too_many_arguments` sites surfaced by the strip-allow
pass (`6ba5b89`). Workspace at 478 warnings post-dead-code cleanup
(`23863e6`).

## Foundation type vocabulary

All in `crates/ops/src/sig.rs`, re-exported at crate root. Each is
`#[derive(Copy, Clone, Debug)]` POD, all fields `pub`, no methods.

| Type | Fields | Used by |
|---|---|---|
| `OpCtx<'a>` | `reg: &'a OpsRegistry, stream: &'a HipStream` | every `hip/*` free-fn wrapper |
| `MmvqBuffers` | `weights, act_q8_1, dst: DevicePtr` | single-weight MMVQ family |
| `MmvqGateUpBuffers` | `gate_w, up_w, act_q8_1, gate_out, up_out: DevicePtr` | fused gate+up MMVQ family |
| `MatmulShape` | `m, k, n: usize` | `qmatmul` / `mmq_*` |
| `MmvqShape` | `n_rows, k: usize` | non-batched MMVQ |
| `MmvqBatchShape` | `n_rows, k, n_slots: usize` | row-tile / batched MMVQ |
| `MmvqGateUpShape` | `n_rows_gate, n_rows_up, k: usize` | gate+up MMVQ |
| `AttnBuffers` | `q, k, v, out: DevicePtr` | every attention variant |
| `AttnDecodeShape` | `n_heads_q, n_heads_kv, head_dim, n_tokens_kv: usize` | `attention_decode_*` |
| `AttnPrefillShape` | `n_q_tokens, n_heads_q, n_heads_kv, head_dim, n_k_tokens, q_offset: usize` | `attention_prefill_*` |
| `AttnKnobs` | `scale: f32, window_size: i32` | every attention variant |
| `NormBuffers` | `input, weight, output: DevicePtr` | `rmsnorm_*` |
| `NormResidual` | `residual_in, residual_out: DevicePtr, residual_scale: f32` | `rmsnorm_*_add_residual` |

Additional structs grow per family as they're migrated (`AttnPrefillSlots`,
`AttnDecodeSlots`, etc.).

## Phase plan

Each phase touches **trait method(s) + impl block(s) + free-fn wrapper(s) +
every caller** atomically. The right slice is 2-3 sibling functions —
small enough to land green in one commit, large enough that the per-commit
overhead doesn't dominate. A first attempt at "do all 12 attention
functions in one phase" overshot scope and was reverted; the slicing
below is the corrected one.

| # | Phase | Status | Sites cleared | Files |
|---|---|---|---|---|
| P0 | Foundation — `sig.rs` + re-exports | ☑ done (`8a0aXXX`) | 0 | `crates/ops/src/sig.rs` (new), `crates/ops/src/lib.rs` |
| **P1 — Attention family (12 fns, 6 sub-phases)** | | | | |
| P1a | `attention_decode_f16` + `_slots` (graph-capture sibling) | ☑ done | 4 | trait, impl, wrapper, decode callers (model-ops + tests) |
| P1b | `attention_decode_f16_batched` + `_paged` | ☑ done | 4 | trait, impl, wrapper, batched callers (model-ops + 2 tests) |
| P1c | `attention_decode_f16_splitk` + `_splitk_h2` | ☑ done | 4 | trait, impl, wrapper, splitk callers (model-ops + swa_softcap_parity) |
| P1d | `attention_decode_q8_kv` + `_splitk` (Q8 decode pair) | ☐ pending | 2 | trait, impl, wrapper, q8 decode callers |
| P1e | `attention_prefill_f16` + `_slots` | ☐ pending | 2 | trait, impl, wrapper, prefill callers, tests |
| P1f | `attention_prefill_q8_kv` + `_f16_paged` | ☐ pending | 2 | trait, impl, wrapper, remaining prefill callers |
| **P2 — Matmul family (~100 sites, sub-phased by sibling group)** | | | | |
| P2a | mmvq single-weight (Q4_0/Q5_K/Q8_0 t128 + warpcoop64) | ☐ pending | ~15 | trait, impl, wrapper, qmatmul callers |
| P2b | mmvq gate-up fused (Q4_0/Q4_1/Q5_K/Q8_0/Q4_0_t128) | ☐ pending | ~15 | same |
| P2c | mmvq row-tile batched (Q4_0/Q5_K/Q8_0 row_tile_batched + gate_up variants) | ☐ pending | ~20 | same |
| P2d | mmvq KV-out (`mmvq_q4_0_kv_f16dst`) + `mmvq` generic + `mmvq_f16_direct` | ☐ pending | ~10 | same |
| P2e | `qmatmul` composite + `mmq` per-dtype | ☐ pending | ~10 | same |
| P2f | indexed_moe_mmvq + indexed_moe_mmq (MoE family — touches hip/moe.rs) | ☐ pending | ~30 | trait, impl, hip/moe.rs, moe_experts.rs |
| **P3 — Norm fused** | | | | |
| P3a | `rmsnorm_*_add_residual` family (3-4 fns) | ☐ pending | ~5 | `ops_trait.rs`, `hip/norm.rs` + callers |
| P3b | `rmsnorm_rope_neox_partial_f16` + `rope_neox_partial_f16` | ☐ pending | ~5 | `ops_trait.rs`, `hip/norm.rs`, `hip/pe.rs` + callers |
| **P4 — KV-append (5 fns)** | | | | |
| P4a | `kv_append_f16_paged_*` (prefill + slots) | ☐ pending | 2 | `ops_trait.rs`, `hip/attention.rs` (where these live) + callers |
| P4b | `kv_append_f16_batched_slots` + `kv_append_v_unit_norm_f16` | ☐ pending | 2 | same |
| **P5 — MoE compose** | | | | |
| P5a | `moe_sort_by_expert_*` (3 variants) | ☐ pending | 3 | `ops_trait.rs`, `hip/moe.rs`, `hip/router.rs` |
| P5b | `moe_combine_*` + `moe_router_*` | ☐ pending | 3 | same |
| **P6 — Collective** | | | | |
| P6 | `bar_ar_*` (4 sibling fns share an 11-arg shape) + `flambeau_p2p_allreduce_sum_tp*` | ☐ pending | ~7 | `backend-hip/src/bar_p2p.rs`, `forward/src/runtime/ar.rs` |
| **P7 — Mid-layer drivers** | | | | |
| P7a | `delta_net.rs` (6 sites) | ☐ pending | ~6 | model-ops/src/delta_net.rs |
| P7b | `moe_experts.rs` (2 sites) + `forward/src/loader/shard.rs::upload_*_sharded_quant` (2 sites) | ☐ pending | ~4 | model-ops + forward/loader |
| P7c | `forward/src/loader/moe.rs` + `forward/src/loader/gdn_shard.rs` | ☐ pending | ~6 | forward/loader |
| **P8 — Tail** | | | | |
| P8 | `build.rs`, `tp_slice.rs`, `quantize_k.rs`, `ctx.rs`, `workers.rs::init_rank` | ☐ pending | ~12 | misc |

## Per-phase commit policy

- Each phase = one commit.
- Each commit must keep `cargo build --release --features hip_serve` green
  AND `cargo clippy --release --features hip_serve --workspace --all-targets`
  errorless (warnings down monotonically, never up).
- Update the **Status** column in the table above to `☑ done <sha>` after
  each commit lands. Update the **Sites cleared** column with the actual
  delta clippy reports.
- Update the **Live warning count** row below after each phase.

## Live state

| Metric | At start | After P0 | After P1a | After P1f | After P2f | After P3 | After P4 | After P5 | After P6 | After P7 | After P8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| total warnings | 478 | 478 | 475 | 471 | 467 | — | — | — | — | — | — |
| `too_many_arguments` | 315 | 315 | 311 | 307 | 303 | — | — | — | — | — | — |
| errors | 0 | 0 | 0 | 0 | 0 | — | — | — | — | — | — |
| cumulative LOC delta | 0 | +232 | +276 | +313 | tbd | — | — | — | — | — | — |

LOC deltas per commit (insertions − deletions, from `git show --stat`):

| Phase | Commit | + | − | net | warnings cleared |
|---|---|---|---|---|---|
| P0 | `8a0aXXX` | 232 | 0 | +232 | 0 (setup) |
| P1a | `62f177e` | 284 | 240 | +44 | 4 |
| P1b | `cf2f429` | 151 | 150 | +1 | 4 |
| P1c | (pending) | tbd | tbd | tbd | 4 |

## Non-goals

- Builder pattern — verbose for mandatory-field POD; struct literals beat it.
- `impl Into<T>` from tuples — tuples have no field names and break > 12 elems.
- Const generics on `head_dim` — runtime dispatch table is the design choice
  (rule 6, root CLAUDE.md).
- Adding methods to the new types beyond `#[derive(...)]` — they are pure
  parameter aggregates, not behaviour carriers.

## Rules invariants

- The new types live in `crates/ops/src/sig.rs` (not `model-ops`, which has
  rule 7 "no new struct types without asking" — the asking happened, but
  the types belong at the wrapper layer, not the leaf-op layer).
- Each type is `Copy + Clone + Debug`. No `PartialEq`/`Eq` until a test
  needs it.
- Buffer aggregates carry `DevicePtr` by value (it's already `Copy`).
- Shape aggregates use `usize`. Knob aggregates use the native scalar
  type for the kernel arg (`f32`, `i32`).
- No lifetimes on the aggregates except `OpCtx<'a>`. Everything else is
  `'static`-borrowing-free.
