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
| P1d | `attention_decode_q8_kv` + `_splitk` (Q8 decode pair) | ☑ done | 4 | trait, impl, wrapper, q8 decode callers |
| P1e | `attention_prefill_f16` + `_slots` | ☑ done | 4 | trait, impl, wrapper, prefill callers (model-ops + 2 tests) |
| P1f | `attention_prefill_q8_kv` + `_f16_paged` | ☑ done | 4 | trait, impl, wrapper, remaining prefill callers (2 model-ops files) |
| **P2 — Matmul family (~100 sites, sub-phased by sibling group)** | | | | |
| P2a | All 9 batched single-weight fns (`mmvq_q4_0_batched`, `q4_k_batched`, `q6_k_batched`, `q8_0_batched`, `q4_1_batched`, `q5_k_r2_batched`, `q5_k_row_tile_batched`, `q4_0_row_tile_batched`, `q8_0_row_tile_batched`) | ☑ done | 9 | trait (2 of 9), impl (2 of 9), hip/qmatmul.rs (9 free fns + 6 in-file dispatcher calls). Zero external callers found. |
| P2b | mmvq gate-up fused (Q4_0/Q4_1/Q5_K/Q8_0/Q4_0_t128) | ☑ done | 10 | trait (5 methods), impl (5 methods), hip/qmatmul.rs (5 free fns), 4 model-ops callers |
| P2c | `mmvq_q4_0_gate_up_{batched,row_tile_batched}` (siblings, share GateUpBatchShape) | ☑ done | 3 | trait (1), impl (1), hip/qmatmul.rs (2 free fns), 1 model-ops caller (delta_net) |
| P2d | remaining single-weight non-batched (`mmvq_q4_0_t128`, `_warpcoop64`, `_kv_f16dst`) | ☐ pending | ~5 | same |
| P2e | mmvq KV-out + `mmvq` generic + `mmvq_f16_direct` + `qmatmul` + `mmq` | ☐ pending | ~10 | same |
| P2f1 | Single-weight `indexed_moe_mmvq_*` (19 free fns + 16 trait + 16 impl + 21 callers) | ☑ done | ~35 | hip/moe.rs, ops_trait.rs, ops_impl.rs, model-ops/moe_experts.rs (incl. 2 macros) — driven by 3 awk scripts under scripts/migrate_moe_*.awk |
| P2f2 | indexed_moe_mmvq gate-up (2 fns: `q4_0_gate_up`, `q8_0_gate_up`) | ☑ done | 4 | trait (2 methods), impl (2 methods), hip/moe.rs (2 free fns), 3 caller sites in moe_experts.rs |
| P2f3 | `indexed_moe_mmvq_q4_k_gate_up` + `_gate_up_sorted` + `_r2_sorted` | ☑ done | 6 | trait (3), impl (3), hip/moe.rs (3 free fns + new `MoeMmvqGateUpSortedBuffers` aggregate in sig.rs), 2 callers in moe_experts.rs |
| P2f4 | MMQ tile8 gate-up — actually 18 free fns + 15 trait + 15 impl (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q3_K/Q5_K and 9 IQ variants), all share `MoeMmqTile8GateUpBuffers` + existing `MoeShape` | ☑ done | 33 | sig.rs (new `MoeMmqTile8GateUpBuffers` + `MoeMmqTile8DownBuffers`), trait, impl, hip/moe.rs, 15 caller sites in moe_experts.rs |
| P2f5 | MMQ tile8 down — 18 free fns + 16 trait + 16 impl (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q3_K/Q5_K/Q6_K + 9 IQ variants), all share `MoeMmqTile8DownBuffers` + `MoeShape` | ☑ done | 34 | trait, impl, hip/moe.rs, 16 caller sites in moe_experts.rs |

### P2f attempt-and-revert notes (for next session)

1st attempt at P2f1 (19 single-weight MMVQ free fns + 16 trait + 16 impls)
landed cleanly in `hip/moe.rs` / `ops_trait.rs` / `ops_impl.rs` via a
Python bulk-script (4-arg buffers + 4-arg shape via `MoeMmvqBuffers` +
`MoeMmvqShape` aggregates added to `sig.rs`). Caller migration in
`model-ops/src/moe_experts.rs` (~30 sites) couldn't complete because
the bash tool started backgrounding Python invocations and the
substitutions never landed — build was broken, reverted to P2c state.

Next-session restart plan:
- Re-extend `sig.rs` with `MoeMmvqBuffers { weights, act, expert_ids,
  dst }` + `MoeMmvqShape { n_rows, n_tokens, top_k, n_sb_per_row }`
  + `MoeMmvqGateUpBuffers { gate_w, up_w, act, expert_ids, gate_out,
  up_out }` + `MoeMmvqSortedBuffers` (for `_q4_k_r2_sorted`).
- Decompose P2f into sub-slices:
  - P2f1 (single-weight MMVQ, 19 fns) — bulk via awk or `Edit replace_all`
    where possible
  - P2f2 (gate-up MMVQ, 2 fns: `q4_0_gate_up`, `q8_0_gate_up`)
  - P2f3 (sorted MMVQ, 1 fn: `q4_k_r2_sorted`)
  - P2f4 (MMQ tile8 gate-up, 11 fns) — already uses `MoeShape` aggregate
  - P2f5 (MMQ tile8 down, 11 fns) — same
- Each sub-slice = 1 commit; verify build+clippy green per commit.
- Beware double-application of `reg -> ctx.reg` substitutions when
  bulk-scripting bodies — the body-fix step must idempotently check
  for existing `ctx.` prefix before substituting (the 1st attempt
  produced `ctx.ctx.reg` on the 6 `_0`-quant variants because the
  second Python pass re-ran on already-migrated bodies).
| **P3 — Norm fused** | | | | |
| P3a | 7 rmsnorm wrappers + 4 caller sites (rmsnorm.rs leaf + delta_net + 3 forward composites) | ☑ done | 10 | `ops_trait.rs`, `hip/norm.rs`, `hip/ops_impl.rs`, sig.rs (+`NormResidualBuffers`/`NormFusedAddBuffers`/`NormShape`), 5 callers. Also added `HipOps::ctx()` helper. |
| P3b | 3 rope wrappers (`rope_f16`, `rope_neox_partial_f16`, `rmsnorm_rope_neox_partial_f16`) + 6 caller sites | ☑ done | 5 | sig.rs (+`RopeBuffers`/`RopeFusedBuffers`/`RopeShape`/`RopePartialShape`), trait, impl, hip/pe.rs, 6 callers. |
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

| Metric | At start | After P0 | After P1f | After P2a | After P2b | After P2c | After P2f1 | After P2f2 | After P2f3 | After P2f4 | After P2f5 | After P3a | After P3b |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| total warnings | 478 | 478 | 456 | 447 | 437 | 434 | 413 | 414 | 411 | 378 | 344 | 329 | 324 |
| `too_many_arguments` | 315 | 315 | 291 | 282 | 272 | 269 | 234 | 230 | 224 | 191 | 157 | 147 | 142 |
| errors | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cumulative LOC delta | 0 | +232 | +10 | +12 | −136 | −162 | −46 | −93 | −157 | −509 | −774 | −815 | −825 |

LOC deltas per commit (insertions − deletions, from `git show --stat`):

| Phase | Commit | + | − | net | warnings cleared |
|---|---|---|---|---|---|
| P0 | `8a0aXXX` | 232 | 0 | +232 | 0 (setup) |
| P1a | `62f177e` | 284 | 240 | +44 | 4 |
| P1b | `cf2f429` | 151 | 150 | +1 | 4 |
| P1c | `60e2a81` | 121 | 192 | −71 | 4 |
| P1d | `1cc57f8` | 112 | 151 | −39 | 4 |
| P1e | `56af554` | 116 | 263 | −147 | 4 |
| P1f | `f7b1229` | 114 | 160 | −46 | 4 |
| P2a | `760d53e` | 150 | 148 | +2 | 9 |
| P2b | `46cd52e` | 128 | 276 | −148 | 10 |
| P2c | `8c09856` | 45 | 71 | −26 | 3 |
| P2f1 | `8c0aa99` | 941 | 825 | +116 | 35 |
| P2f2 | `b46878f` | 82 | 129 | −47 | 4 |
| P2f3 | `bd95e29` | 101 | 165 | −64 | 6 |
| P2f4 | `9c269ff` | 382 | 734 | −352 | 33 |
| P2f5 | `5e99a56` | 339 | 604 | −265 | 34 |
| P3a | `f4eca1b` | 285 | 326 | −41 | 10 |
| P3b | `8681de0` | 183 | 193 | −10 | 5 |

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
