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

| # | Phase | Status | Sites cleared | Files |
|---|---|---|---|---|
| P0 | Foundation — `sig.rs` + re-exports | ☑ done | 0 | `crates/ops/src/sig.rs` (new), `crates/ops/src/lib.rs` |
| P1 | Attention family | ☐ pending | 17 | `ops_trait.rs`, `hip/attention.rs`, `model-ops/src/ops/attn_*.rs`, attention tests |
| P2a | mmvq single-weight | ☐ pending | ~30 | `ops_trait.rs`, `hip/qmatmul.rs`, `model-ops/src/ops/qmatmul.rs` |
| P2b | mmvq gate-up fused | ☐ pending | ~25 | same as P2a |
| P2c | mmvq batched / row-tile | ☐ pending | ~25 | same as P2a |
| P2d | qmatmul + mmq | ☐ pending | ~20 | same as P2a |
| P3 | Norm fused | ☐ pending | ~10 | `ops_trait.rs`, `hip/norm.rs`, `hip/pe.rs` |
| P4 | KV-append | ☐ pending | ~9 | `ops_trait.rs`, `hip/kv_append.rs` |
| P5 | MoE compose | ☐ pending | ~10 | `ops_trait.rs`, `hip/moe.rs`, `hip/router.rs` |
| P6 | Collective (`bar_ar_*`) | ☐ pending | ~7 | `backend-hip/src/bar_p2p.rs`, `forward/src/runtime/ar.rs` |
| P7 | Mid-layer drivers | ☐ pending | ~30 | per-file (`delta_net.rs`, `moe_experts.rs`, `loader/moe.rs`, `loader/gdn_shard.rs`, `loader/shard.rs`) |
| P8 | Tail | ☐ pending | ~12 | `build.rs`, `tp_slice.rs`, `quantize_k.rs`, `ctx.rs`, `workers.rs::init_rank` |

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

| Metric | At start | After P0 | After P1 | After P2 | After P3 | After P4 | After P5 | After P6 | After P7 | After P8 |
|---|---|---|---|---|---|---|---|---|---|---|
| total warnings | 478 | 478 | — | — | — | — | — | — | — | — |
| `too_many_arguments` | 315 | 315 | — | — | — | — | — | — | — | — |
| errors | 0 | 0 | — | — | — | — | — | — | — | — |

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
