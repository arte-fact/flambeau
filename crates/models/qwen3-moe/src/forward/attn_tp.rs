//! tensor-parallel `forward_full_attn_decode`.
//!
//! Thin shim over [`flambeau_blocks::StandardAttention::forward_decode`]
//! with per-rank sliced weights. The block does the full kernel
//! sequence (norm + Q8_1 quant → gated Q proj split → K/V projection
//! with Q4_0/Q8_0 fusion fast paths → per-head Q/K norm → NeoX
//! partial RoPE → KV append (F16 or Q8) → attention decode with
//! splitk dispatch on n_tokens_kv > 256 → post-attn sigmoid_mul →
//! output projection → cast); this wrapper preserves the historic
//! flat parameter list and the `kv_replicated` knob for callers in
//! `forward::tp` + `forward::hybrid`.
//!
//! Per-rank shape contract:
//! - `attn_q.dims == [2 · local_n_heads · head_dim, hidden]` (gated)
//! - `attn_k.dims == [local_n_kv_heads · head_dim, hidden]`
//! - `attn_v.dims == [local_n_kv_heads · head_dim, hidden]`
//! - `attn_output.dims == [hidden, local_n_heads · head_dim]`
//! where `local_n_heads = n_heads / tp_world` and
//! `local_n_kv_heads = if kv_replicated { n_kv_heads } else { n_kv_heads / tp_world }`.

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::{HipDevice, HipStream};
use flambeau_blocks::StandardAttention;
use flambeau_core::DevicePtr;
use flambeau_ops::hip::OpsRegistry;
use flambeau_ops::HipOps;
use flambeau_runtime::{CacheLayout, KvCache};

use super::attn::FullAttnScratch;
use super::common::{mat_shape, qdtype_of};
use crate::config::Qwen3MoEConfig;
use crate::weights::DeviceTensor;

/// Per-rank decode for one full-attention layer. Delegates to
/// [`StandardAttention::forward_decode`] with sliced shapes.
#[expect(
    clippy::too_many_arguments,
    reason = "preserves the historic flat parameter list for callers."
)]
pub fn forward_full_attn_decode_tp<L: CacheLayout>(
    ops: &OpsRegistry,
    stream: &HipStream,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    attn_norm: &DeviceTensor,
    attn_q: &DeviceTensor,
    attn_k: &DeviceTensor,
    attn_v: &DeviceTensor,
    attn_output: &DeviceTensor,
    attn_q_norm: &DeviceTensor,
    attn_k_norm: &DeviceTensor,
    kv_cache: &mut KvCache<L, HipDevice>,
    scratch: &mut FullAttnScratch,
    x_in: DevicePtr,
    partial_attn_out: DevicePtr,
    position: usize,
    tp_world: u32,
    // when true, K/V run with full cfg.num_kv_heads per rank.
    // attn_k/attn_v weights must be Replicated; KvCache is full-sized.
    kv_replicated: bool,
) -> Result<()> {
    let block = build_tp_attn_block(
        cfg, attn_norm, attn_q, attn_k, attn_v, attn_output, attn_q_norm, attn_k_norm,
        tp_world, kv_replicated,
    )
    .context("build StandardAttention (TP decode)")?;
    let hipops = HipOps::new(ops, stream);
    block.forward_decode(
        &hipops,
        device,
        stream,
        x_in,
        partial_attn_out,
        kv_cache,
        &mut scratch.view_mut(),
        position,
        /* slots = */ None,
    )
}

/// Build the per-rank `StandardAttention` block from sliced
/// [`DeviceTensor`]s. Shared by the decode + prefill TP paths.
#[expect(
    clippy::too_many_arguments,
    reason = "matches build_full_attn_block — flat handle list."
)]
fn build_tp_attn_block(
    cfg: &Qwen3MoEConfig,
    attn_norm: &DeviceTensor,
    attn_q: &DeviceTensor,
    attn_k: &DeviceTensor,
    attn_v: &DeviceTensor,
    attn_output: &DeviceTensor,
    attn_q_norm: &DeviceTensor,
    attn_k_norm: &DeviceTensor,
    tp_world: u32,
    kv_replicated: bool,
) -> Result<StandardAttention> {
    if tp_world == 0 {
        bail!("tp_world must be >= 1");
    }
    let world = tp_world as usize;
    let n_heads = cfg.num_heads;
    let n_kv_heads = cfg.num_kv_heads;
    if n_heads % world != 0 {
        bail!("num_heads {n_heads} not divisible by tp_world {tp_world}");
    }
    let local_n_heads = n_heads / world;
    let local_n_kv_heads = if kv_replicated {
        n_kv_heads
    } else {
        if n_kv_heads % world != 0 {
            bail!(
                "num_kv_heads {n_kv_heads} not divisible by tp_world {tp_world} \
                 and kv_replicated=false (caller bug — pass kv_replicated=true \
                 from Qwen35DenseTpLayout::kv_replicated)"
            );
        }
        n_kv_heads / world
    };

    let (q_rows, q_k) = mat_shape(attn_q)?;
    let (k_rows, k_k) = mat_shape(attn_k)?;
    let (v_rows, v_k) = mat_shape(attn_v)?;
    let (o_rows, o_k) = mat_shape(attn_output)?;
    StandardAttention::new(
        flambeau_blocks::WeightHandle {
            ptr: attn_q.ptr,
            dtype: qdtype_of(attn_q.dtype)?,
            dims: [q_rows, q_k],
        },
        flambeau_blocks::WeightHandle {
            ptr: attn_k.ptr,
            dtype: qdtype_of(attn_k.dtype)?,
            dims: [k_rows, k_k],
        },
        Some(flambeau_blocks::WeightHandle {
            ptr: attn_v.ptr,
            dtype: qdtype_of(attn_v.dtype)?,
            dims: [v_rows, v_k],
        }),
        flambeau_blocks::WeightHandle {
            ptr: attn_output.ptr,
            dtype: qdtype_of(attn_output.dtype)?,
            dims: [o_rows, o_k],
        },
        attn_norm.ptr,
        attn_q_norm.ptr,
        attn_k_norm.ptr,
        cfg.hidden_size,
        local_n_heads,
        local_n_kv_heads,
        cfg.head_dim,
        cfg.rms_norm_eps,
        cfg.rope.freq_base,
        cfg.rope.rotated_dims,
        /* gated = */ true,
    )
}

#[allow(dead_code)]

/// 2** — per-rank L-batched full-attn prefill. Counterpart of
/// [`forward_full_attn_decode_tp`] for n_tokens > 1. Mirrors the
/// kernel sequence of [`super::attn::forward_full_attn_prefill`] (the
/// PP version) but emits a `[L, hidden]` partial that the caller folds
/// via one [`flambeau_backend_hip::BarP2pAllReduce::residual_tp{2,4}`]
/// across `L * hidden` elements (instead of L per-token ARs).
/// Reuses [`super::attn::FullAttnPrefillScratch`] verbatim — the
/// scratch is sized for the *full* (`n_heads`, `n_kv_heads`, `q_width
/// = n_heads · head_dim`) shapes; per-rank kernels only touch the
/// head-subset prefix (same waste as `forward_full_attn_decode_tp`,
/// same trade-off — sized scratch is filed but not on the
/// path).
#[expect(
    clippy::too_many_arguments,
    reason = "matches super::attn::forward_full_attn_prefill — flat parameter list \
              avoids struct copies on the prefill path."
)]
pub fn forward_full_attn_prefill_tp<L: flambeau_runtime::CacheLayout>(
    ops: &OpsRegistry,
    stream: &HipStream,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    attn_norm: &DeviceTensor,
    attn_q: &DeviceTensor,
    attn_k: &DeviceTensor,
    attn_v: &DeviceTensor,
    attn_output: &DeviceTensor,
    attn_q_norm: &DeviceTensor,
    attn_k_norm: &DeviceTensor,
    kv_cache: &mut KvCache<L, HipDevice>,
    scratch: &mut super::attn::FullAttnPrefillScratch,
    x_in: DevicePtr,
    partial_attn_out: DevicePtr,
    n_tokens: usize,
    start_position: usize,
    tp_world: u32,
    kv_replicated: bool,
) -> Result<()> {
    let block = build_tp_attn_block(
        cfg, attn_norm, attn_q, attn_k, attn_v, attn_output, attn_q_norm, attn_k_norm,
        tp_world, kv_replicated,
    )
    .context("build StandardAttention (TP prefill)")?;
    let hipops = HipOps::new(ops, stream);
    block.forward_prefill(
        &hipops,
        device,
        stream,
        x_in,
        partial_attn_out,
        kv_cache,
        &mut scratch.view_mut(),
        n_tokens,
        start_position,
        /* slots = */ None,
    )
}

/// **P2.9b-i2-C** — batched decode for one full-attention layer on a
/// `tp_world`-rank TP mesh.
/// Mirrors [`forward_full_attn_prefill_tp`] for the front-end ops
/// (rmsnorm, Q|gate / K / V projection, per-head Q/K rmsnorm, RoPE) at
/// `n_tokens = slot_positions.len()` — those steps batch across slots
/// at fixed N. Steps 8 (KV-append) and 9 (attention) split per-slot
/// because each slot owns its own per-rank KV cache and query history.
/// Output is a per-rank partial `[N, hidden]` F16 in `partial_attn_out`;
/// the caller AllReduces across ranks (BarP2pAllReduce) to produce the
/// replicated `[N, hidden]` attention contribution.
/// Layout:
/// - `slot_kv_caches[s]` is the **rank-local** KV cache for slot `s`
/// (sized for `local_n_kv_heads`). All N caches must be FullAttn
/// F16Contig. Q8 KV is V2.
/// - `slot_positions[s]` is the cache tail for slot `s` *before* this
/// token is appended.
/// - `scratch` is a single shared per-rank `FullAttnPrefillScratch`
/// sized for `max_tokens >= N`.
/// **What this function does NOT do**: AllReduce. Caller schedules
/// `BarP2pAllReduce::residual_*` on `partial_attn_out` immediately
/// after this returns.
#[expect(
    clippy::too_many_arguments,
    reason = "matches forward_full_attn_prefill_tp's flat parameter list — \
              scheduler hot path; struct copies regress measurable wall."
)]
pub fn forward_full_attn_layer_decode_batched_tp(
    ops: &OpsRegistry,
    stream: &HipStream,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    attn_norm: &DeviceTensor,
    attn_q: &DeviceTensor,
    attn_k: &DeviceTensor,
    attn_v: &DeviceTensor,
    attn_output: &DeviceTensor,
    attn_q_norm: &DeviceTensor,
    attn_k_norm: &DeviceTensor,
    slot_kv_caches: &mut [&mut KvCache<flambeau_runtime::F16Contig, HipDevice>],
    scratch: &mut super::attn::FullAttnPrefillScratch,
    x_in: DevicePtr,
    partial_attn_out: DevicePtr,
    slot_positions: &[usize],
    tp_world: u32,
    kv_replicated: bool,
) -> Result<()> {
    // Env-gated escape hatch: FLAMBEAU_KV_APPEND_BATCHED=0 disables
    // the batched-kv-append kernel and falls back to per-slot DtoD
    // memcpys (the original 4N-launch path). Default ON.
    let batch_kv_append = std::env::var("FLAMBEAU_KV_APPEND_BATCHED")
        .as_deref()
        != Ok("0");
    let block = build_tp_attn_block(
        cfg, attn_norm, attn_q, attn_k, attn_v, attn_output, attn_q_norm, attn_k_norm,
        tp_world, kv_replicated,
    )
    .context("build StandardAttention (TP batched-decode)")?;
    let hipops = HipOps::new(ops, stream);
    block.forward_decode_batched_tp(
        &hipops,
        device,
        stream,
        x_in,
        partial_attn_out,
        slot_kv_caches,
        slot_positions,
        &mut scratch.view_mut_batched(),
        batch_kv_append,
    )
}
