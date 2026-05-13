//! Tensor-parallel decode + prefill for the MoE FFN and shared-expert
//! paths. The decode entry points are thin shims over
//! [`flambeau_blocks::MoeExperts::forward_decode_tp`] and
//! [`flambeau_blocks::SharedExpert::forward_decode`] with per-rank
//! sliced intermediate. The prefill paths still inline the indexed-MoE
//! tile8 dispatch — moving them into the block requires a `tp_world`
//! aware threshold (the TP path disables decode-tile8 to avoid the AR
//! serialisation against the tile8 launch); deferred to S7b-2.
//!
//! Per-rank shape contract:
//! - `ffn_gate_exps` / `ffn_up_exps` (col-parallel) →
//!   `[n_experts, local_inter, hidden]`.
//! - `ffn_down_exps` (row-parallel) → `[n_experts, hidden, local_inter]`.
//! - `partial_ffn_out` is `Σ_k weights[k] * down_local[k, :]` (no
//!   residual). The layer driver folds this and the residual via
//!   `BarP2pAllReduce` after this returns.
//! - Bit-exactness:
//!   `Σ_r Σ_k w[k] * down_local_r[k] = Σ_k w[k] * Σ_r down_local_r[k]`
//!   = `Σ_k w[k] * down_full[k]` modulo FP32 reduction order.

#![cfg(feature = "hip")]

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "forward-path composition; same rationale as super::moe — every \
              unsafe is a kernel launch over session-scoped buffers."
)]

use anyhow::{bail, Context, Result};
use flambeau_core::DevicePtr;
use flambeau_ops::hip::{HipStream, OpsRegistry};

use super::common::validate_moe_dtypes;
use super::moe::{MoePrefillScratch, MoeScratch, SharedExpertPrefillScratch, SharedExpertScratch};
use crate::config::Qwen3MoEConfig;
use crate::weights::DeviceTensor;

/// Per-rank decode for one MoE FFN layer.
/// `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps` are TP-sliced per
/// the layout table. `expert_ids` and `expert_weights` come from
/// the (replicated) router output — same on every rank since the
/// router runs replicated.
/// # Errors
/// - `tp_world == 0` or `moe_intermediate_size % tp_world != 0`.
/// - Dtype validation failures (propagated from `validate_moe_dtypes`).
/// - Underlying op-dispatch / kernel-launch failures.
#[expect(
    clippy::too_many_arguments,
    reason = "matches the PP forward_moe_ffn_decode signature shape"
)]
pub fn forward_moe_ffn_decode_tp(
    ops: &OpsRegistry,
    stream: &HipStream,
    cfg: &Qwen3MoEConfig,
    ffn_gate_exps: &DeviceTensor,
    ffn_up_exps: &DeviceTensor,
    ffn_down_exps: &DeviceTensor,
    scratch: &mut MoeScratch,
    x_norm: DevicePtr,
    partial_ffn_out: DevicePtr,
    tp_world: u32,
) -> Result<()> {
    if tp_world == 0 {
        bail!("tp_world must be >= 1");
    }
    let world = tp_world as usize;
    let hidden = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size;
    if inter % world != 0 {
        bail!("moe_intermediate_size {inter} not divisible by tp_world {tp_world}");
    }
    let local_inter = inter / world;
    validate_moe_dtypes(
        "indexed-MoE (TP)",
        ffn_gate_exps.dtype,
        ffn_up_exps.dtype,
        ffn_down_exps.dtype,
        hidden,
        local_inter,
    )?;
    let block = build_moe_experts_block_tp(cfg, ffn_gate_exps, ffn_up_exps, ffn_down_exps, local_inter)
        .context("build MoeExperts (TP decode)")?;
    let hipops = flambeau_ops::HipOps::new(ops, stream);
    block.forward_decode_tp(&hipops, x_norm, partial_ffn_out, scratch.view())
}

/// Build a per-rank TP-sliced [`flambeau_blocks::MoeExperts`] block.
/// `local_inter = moe_intermediate_size / tp_world`. Used by both
/// the TP decode shim above and the prefill path's per-call instance.
fn build_moe_experts_block_tp(
    cfg: &Qwen3MoEConfig,
    ffn_gate_exps: &DeviceTensor,
    ffn_up_exps: &DeviceTensor,
    ffn_down_exps: &DeviceTensor,
    local_inter: usize,
) -> Result<flambeau_blocks::MoeExperts> {
    use super::common::qdtype_of;
    let hidden = cfg.hidden_size;
    let gate_dt = qdtype_of(ffn_gate_exps.dtype)?;
    let up_dt = qdtype_of(ffn_up_exps.dtype)?;
    let down_dt = qdtype_of(ffn_down_exps.dtype)?;
    // The router weight isn't consumed on the TP decode_tp path (the
    // router runs replicated on every rank via the caller's
    // `route_decode` call upstream); a dummy WeightHandle satisfies
    // the constructor's shape check.
    let router_handle = flambeau_blocks::WeightHandle {
        ptr: DevicePtr(0),
        dtype: qdtype_of(flambeau_quant::GgmlDType::F32)?,
        dims: [cfg.num_experts, hidden],
    };
    flambeau_blocks::MoeExperts::new(
        router_handle,
        flambeau_blocks::WeightHandle {
            ptr: ffn_gate_exps.ptr,
            dtype: gate_dt,
            dims: [cfg.num_experts * local_inter, hidden],
        },
        flambeau_blocks::WeightHandle {
            ptr: ffn_up_exps.ptr,
            dtype: up_dt,
            dims: [cfg.num_experts * local_inter, hidden],
        },
        flambeau_blocks::WeightHandle {
            ptr: ffn_down_exps.ptr,
            dtype: down_dt,
            dims: [cfg.num_experts * hidden, local_inter],
        },
        hidden,
        local_inter,
        cfg.num_experts,
        cfg.num_experts_per_tok,
    )
}

/// per-rank decode for the shared expert.
/// Structurally identical to [`super::dense_ffn_tp::forward_dense_ffn_decode_tp`]
/// except `shared_expert_intermediate_size` replaces `moe_intermediate_size`,
/// and the output goes to `shared_delta_out` (the layer driver folds it into
/// `partial_ffn_out` before the AR via `moe_combine_f16` with `residual =
/// shared_delta`).
#[expect(
    clippy::too_many_arguments,
    reason = "matches dense FFN TP signature shape"
)]
pub fn forward_shared_expert_decode_tp(
    ops: &OpsRegistry,
    stream: &HipStream,
    cfg: &Qwen3MoEConfig,
    ffn_gate_shexp: &DeviceTensor,
    ffn_up_shexp: &DeviceTensor,
    ffn_down_shexp: &DeviceTensor,
    // qwen3next gates the shared expert by sigmoid(x_norm·w);
    // qwen35moe omits this projection (None). Replicated weight, applied
    // pre-AR (linear in the partial down output, so AR commutes with the
    // per-token scalar gate).
    ffn_gate_inp_shexp: Option<&DeviceTensor>,
    scratch: &mut SharedExpertScratch,
    x_norm: DevicePtr,
    shared_delta_out: DevicePtr,
    tp_world: u32,
) -> Result<()> {
    use super::common::qdtype_of;
    if tp_world == 0 {
        bail!("tp_world must be >= 1");
    }
    let world = tp_world as usize;
    let hidden = cfg.hidden_size;
    let inter = cfg
        .shared_expert_intermediate_size
        .context("forward_shared_expert_decode_tp requires cfg.shared_expert_intermediate_size")?;
    if inter % world != 0 {
        bail!("shared_expert_intermediate_size {inter} not divisible by tp_world {tp_world}");
    }
    let local_inter = inter / world;

    let block = flambeau_blocks::SharedExpert::new(
        ffn_gate_inp_shexp.map(|t| t.ptr),
        flambeau_blocks::WeightHandle {
            ptr: ffn_gate_shexp.ptr,
            dtype: qdtype_of(ffn_gate_shexp.dtype)?,
            dims: [local_inter, hidden],
        },
        flambeau_blocks::WeightHandle {
            ptr: ffn_up_shexp.ptr,
            dtype: qdtype_of(ffn_up_shexp.dtype)?,
            dims: [local_inter, hidden],
        },
        flambeau_blocks::WeightHandle {
            ptr: ffn_down_shexp.ptr,
            dtype: qdtype_of(ffn_down_shexp.dtype)?,
            dims: [hidden, local_inter],
        },
        hidden,
        local_inter,
    )
    .context("SharedExpert::new (TP decode)")?;
    let hipops = flambeau_ops::HipOps::new(ops, stream);
    block.forward_decode(&hipops, x_norm, shared_delta_out, scratch.view())
}

/// 2** — L-batched per-rank MoE FFN prefill.
/// Sister of [`forward_moe_ffn_decode_tp`] (M=L instead of M=1). Same
/// per-rank ColParallel (gate/up) + RowParallel (down) sharding; every
/// kernel call is parametrised by `n_tokens` so a single sweep handles
/// the whole prompt. Output is `partial_ffn_out[L, hidden]` —
/// per-rank Σ_k weights[k] * down_local[k, :], no residual. Caller AR's
/// after this layer.
/// Like decode_tp, this is the bandwidth-stable indexed-MoE MMVQ path:
/// no sort+pad MMQ tile8. The PP non-TP `forward_moe_ffn_prefill`
/// auto-routes to tile8 at `n_tokens >= 32`; the TP MMVQ-per-token
/// fallback is correct at any L. tile8 + sort-by-expert TP integration
/// is a V2.x perf lever (would require per-rank sort scratch +
/// expert-id replication invariant).
/// `n_tokens` ≤ `scratch.max_tokens`; caller chunks larger prompts.
#[expect(
    clippy::too_many_arguments,
    reason = "preserves the historic flat parameter list."
)]
pub fn forward_moe_ffn_prefill_tp(
    ops: &OpsRegistry,
    stream: &HipStream,
    cfg: &Qwen3MoEConfig,
    ffn_gate_exps: &DeviceTensor,
    ffn_up_exps: &DeviceTensor,
    ffn_down_exps: &DeviceTensor,
    scratch: &mut MoePrefillScratch,
    x_norm: DevicePtr,
    partial_ffn_out: DevicePtr,
    n_tokens: usize,
    tp_world: u32,
) -> Result<()> {
    if tp_world == 0 {
        bail!("tp_world must be >= 1");
    }
    if n_tokens == 0 {
        bail!("forward_moe_ffn_prefill_tp called with n_tokens = 0");
    }
    if n_tokens > scratch.max_tokens {
        bail!(
            "forward_moe_ffn_prefill_tp: n_tokens={n_tokens} > scratch.max_tokens={}",
            scratch.max_tokens
        );
    }
    let world = tp_world as usize;
    let inter = cfg.moe_intermediate_size;
    if inter % world != 0 {
        bail!("moe_intermediate_size {inter} not divisible by tp_world {tp_world}");
    }
    let local_inter = inter / world;
    validate_moe_dtypes(
        "indexed-MoE (TP) prefill",
        ffn_gate_exps.dtype,
        ffn_up_exps.dtype,
        ffn_down_exps.dtype,
        cfg.hidden_size,
        local_inter,
    )?;
    let mut block = build_moe_experts_block_tp(
        cfg, ffn_gate_exps, ffn_up_exps, ffn_down_exps, local_inter,
    )
    .context("build MoeExperts (TP prefill)")?;
    // TP-only threshold: `tp_world >= 2` keeps tile8 off at n_tokens < 32
    // to avoid the AR-vs-tile8 serialisation cost (qwen3-moe TP cert:
    // -10% on tp2 / N=4). `tp_world == 1` uses the block default
    // (n_pairs >= 8 engages tile8 at every prefill or batched-decode size).
    if tp_world >= 2 {
        block = block.with_tile8_min_tokens(32);
    }
    let hipops = flambeau_ops::HipOps::new(ops, stream);
    block.forward_prefill_tp(&hipops, x_norm, partial_ffn_out, n_tokens, scratch.view())
}


/// 2** — L-batched per-rank shared-expert prefill.
/// Sister of [`forward_shared_expert_decode_tp`] (M=L instead of M=1).
/// Per-rank ColParallel (gate/up) + RowParallel (down) shared-expert
/// FFN over L tokens; output is `shared_delta_out[L, hidden]` (no
/// residual, no scale). The layer driver folds it into
/// `partial_ffn_out` before the post-FFN AR.
/// Note: like decode_tp, this omits the `shared_expert_scale_f32` step
/// the non-TP `forward_shared_expert_prefill` applies. The asymmetry
/// pre-dates (it's how forward_shared_expert_decode_tp was
/// landed at ). Out of scope to fix here.
#[expect(
    clippy::too_many_arguments,
    reason = "matches forward_shared_expert_decode_tp shape"
)]
pub fn forward_shared_expert_prefill_tp(
    ops: &OpsRegistry,
    stream: &HipStream,
    cfg: &Qwen3MoEConfig,
    ffn_gate_shexp: &DeviceTensor,
    ffn_up_shexp: &DeviceTensor,
    ffn_down_shexp: &DeviceTensor,
    // qwen3next gates the shared expert by sigmoid(x_norm·w);
    // qwen35moe omits this projection (None). See decode_tp for the AR
    // commutativity argument.
    ffn_gate_inp_shexp: Option<&DeviceTensor>,
    scratch: &mut SharedExpertPrefillScratch,
    x_norm: DevicePtr,
    shared_delta_out: DevicePtr,
    n_tokens: usize,
    tp_world: u32,
) -> Result<()> {
    use super::common::qdtype_of;
    if tp_world == 0 {
        bail!("tp_world must be >= 1");
    }
    let world = tp_world as usize;
    let hidden = cfg.hidden_size;
    let inter = cfg
        .shared_expert_intermediate_size
        .context("forward_shared_expert_prefill_tp requires cfg.shared_expert_intermediate_size")?;
    if inter % world != 0 {
        bail!("shared_expert_intermediate_size {inter} not divisible by tp_world {tp_world}");
    }
    let local_inter = inter / world;
    let block = flambeau_blocks::SharedExpert::new(
        ffn_gate_inp_shexp.map(|t| t.ptr),
        flambeau_blocks::WeightHandle {
            ptr: ffn_gate_shexp.ptr,
            dtype: qdtype_of(ffn_gate_shexp.dtype)?,
            dims: [local_inter, hidden],
        },
        flambeau_blocks::WeightHandle {
            ptr: ffn_up_shexp.ptr,
            dtype: qdtype_of(ffn_up_shexp.dtype)?,
            dims: [local_inter, hidden],
        },
        flambeau_blocks::WeightHandle {
            ptr: ffn_down_shexp.ptr,
            dtype: qdtype_of(ffn_down_shexp.dtype)?,
            dims: [hidden, local_inter],
        },
        hidden,
        local_inter,
    )
    .context("SharedExpert::new (TP prefill)")?;
    let hipops = flambeau_ops::HipOps::new(ops, stream);
    block.forward_prefill(&hipops, x_norm, shared_delta_out, n_tokens, scratch.view())
}
