//! GDN (gated delta-net) layer loader.

use anyhow::{bail, Result};
use flambeau_backend_hip::HipDevice;
use flambeau_core::DevicePtr;
use flambeau_quant::GgufFile;

use crate::ctx::{GdnDims, GdnWeights};

use super::gdn_shard::{
    upload_f32_array_sharded, upload_gdn_fused_qkv_f32, upload_gdn_fused_qkv_quant,
};
use super::primitives::{upload_dequant_to_f16, upload_f32_tensor};
use super::shard::{upload_col_sharded_quant, upload_quant_weight, upload_row_sharded_quant};
use super::ShardMode;

/// How the loader shards a GDN layer's K vs V heads across TP ranks.
/// Independent from `rep_inner_layout` (the block-kernel K-head
/// repeat math); this enum only controls upload-time tensor slicing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdnTpMode {
    /// Shard both K and V heads uniformly along the head axis.
    /// Geometry gate: `num_k_heads % n_ranks == 0`
    ///   ∧ `num_v_heads % n_ranks == 0`.
    /// Pairs with `rep_inner_layout = true` on qwen3-Next-family
    /// arches.
    FullShard,
    /// Replicate the full `num_k_heads × head_k_dim` slab of Q and K
    /// channels on every rank; shard only the V slab.
    /// Geometry gate: `num_v_heads % n_ranks == 0`
    ///   ∧ `(num_v_heads / n_ranks) % num_k_heads == 0`
    /// (the per-rank kernel computes `n_rep = local_num_v_heads /
    /// num_k_heads` as a clean integer). Pairs with
    /// `rep_inner_layout = false` on qwen35 / qwen35moe / qwen36moe.
    KReplicated,
}

/// Pick `GdnTpMode` from the per-rank geometry: prefer `KReplicated`
/// when `(num_v_heads / n_ranks) % num_k_heads == 0` (the kernel's
/// `n_rep` lands on a clean integer); fall back to `FullShard`
/// otherwise. `n_ranks <= 1` always returns `KReplicated` (SD has no
/// sharding to do; the rep-vs-shard distinction collapses).
///
/// Real-model dispatch:
/// - Qwen3.5/3.6-9B, Qwen3.6-35B-A3B (n_v / n_k = integer) → `KReplicated`
/// - Qwen3.5/3.6-27B (24 v / 16 k = 1.5)                   → `FullShard`
pub fn gdn_tp_mode_for(g: GdnDims, n_ranks: usize) -> GdnTpMode {
    if n_ranks <= 1 {
        return GdnTpMode::KReplicated;
    }
    let local_v = g.num_v_heads / n_ranks;
    if local_v % g.num_k_heads == 0 {
        GdnTpMode::KReplicated
    } else {
        GdnTpMode::FullShard
    }
}

/// Per-rank `GdnDims` under TP. Math differs by mode:
/// `KReplicated` keeps K-heads global (only V shards);
/// `FullShard` divides both K and V heads by `n_ranks`.
/// `n_ranks <= 1` short-circuits to the input `g` (no sharding).
pub fn per_rank_gdn_dims(g: GdnDims, n_ranks: usize) -> GdnDims {
    if n_ranks <= 1 {
        return g;
    }
    let mode = gdn_tp_mode_for(g, n_ranks);
    let num_v_heads_local = g.num_v_heads / n_ranks;
    let num_k_heads_local = match mode {
        GdnTpMode::KReplicated => g.num_k_heads,
        GdnTpMode::FullShard => g.num_k_heads / n_ranks,
    };
    let d_inner_local = num_v_heads_local * g.head_v_dim;
    let conv_channels_local = 2 * num_k_heads_local * g.head_k_dim + d_inner_local;
    GdnDims {
        d_inner: d_inner_local,
        num_v_heads: num_v_heads_local,
        num_k_heads: num_k_heads_local,
        head_k_dim: g.head_k_dim,
        head_v_dim: g.head_v_dim,
        conv_channels: conv_channels_local,
        conv_kernel: g.conv_kernel,
    }
}

pub struct GdnLayerSpec<'a> {
    pub attn_norm_name: &'a str,
    pub attn_qkv_name: &'a str,
    pub attn_gate_name: &'a str,
    pub ssm_alpha_name: &'a str,
    pub ssm_beta_name: &'a str,
    pub ssm_out_name: &'a str,
    pub ssm_dt_bias_name: &'a str,
    pub ssm_a_name: &'a str,
    pub ssm_conv1d_name: &'a str,
    pub ssm_norm_name: &'a str,
    pub hidden: usize,
    /// Per-rank dims when `shard` is `Tp` — caller has already divided
    /// `num_v_heads` (always) and `num_k_heads` (only under `FullShard`)
    /// by `n_ranks` before constructing the spec. `conv_channels` and
    /// `d_inner` are likewise per-rank.
    pub dims: GdnDims,
    pub rms_eps: f32,
    pub rep_inner_layout: bool,
    /// TP shard policy. Ignored under `ShardMode::Replicated`.
    pub tp_mode: GdnTpMode,
}

/// Replicated or TP-sharded GDN layer load. Under
/// `ShardMode::Tp { rank, n_ranks }` the caller must have set
/// `spec.dims` to the per-rank values consistent with `spec.tp_mode`
/// (see `GdnTpMode`). The on-disk Q|K|V geometry of `attn_qkv` /
/// `ssm_conv1d` is interpreted by this loader.
pub fn load_gdn_layer(
    file: &GgufFile,
    device: &HipDevice,
    spec: &GdnLayerSpec,
    shard: ShardMode,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<GdnWeights> {
    let g = spec.dims;
    let (rank, n_ranks, kq_replicated) = match shard {
        ShardMode::Replicated => (0usize, 1usize, false),
        ShardMode::Tp { rank, n_ranks } => {
            let kq_rep = matches!(spec.tp_mode, GdnTpMode::KReplicated);
            let global_num_v_heads = g.num_v_heads * n_ranks;
            if global_num_v_heads % n_ranks != 0 {
                bail!(
                    "load_gdn_layer: global num_v_heads {global_num_v_heads} not divisible by n_ranks {n_ranks}"
                );
            }
            match spec.tp_mode {
                GdnTpMode::FullShard => {
                    let global_num_k_heads = g.num_k_heads * n_ranks;
                    if global_num_k_heads % n_ranks != 0 {
                        bail!(
                            "load_gdn_layer FullShard: global num_k_heads {global_num_k_heads} not divisible by n_ranks {n_ranks}"
                        );
                    }
                }
                GdnTpMode::KReplicated => {
                    if g.num_v_heads % g.num_k_heads != 0 {
                        bail!(
                            "load_gdn_layer KReplicated: per-rank num_v_heads {} not divisible by num_k_heads {} (n_rep must be a clean integer; pick FullShard or change n_ranks)",
                            g.num_v_heads,
                            g.num_k_heads
                        );
                    }
                }
            }
            (rank, n_ranks, kq_rep)
        }
    };

    let attn_norm = upload_dequant_to_f16(file, device, spec.attn_norm_name, spec.hidden, allocs)?;
    // ssm_norm is consumed by `rmsnorm_f32` inside the GDN block,
    // so the weight must be uploaded as F32. (Earlier dequant-to-F16
    // landed in this slot was the v2 chained-prefill bug — the
    // kernel reads F32 bytes; F16 bytes get reinterpreted as
    // ~1/250 the correct values, collapsing the GDN output.)
    let ssm_norm_w = upload_f32_tensor(file, device, spec.ssm_norm_name, g.head_v_dim, allocs)?;

    // Global K-head count for the fused-QKV unpacking: under
    // KReplicated `g.num_k_heads` is already the global count; under
    // FullShard `g.num_k_heads = global/n_ranks`, so re-scale.
    let global_num_k_heads = if kq_replicated {
        g.num_k_heads
    } else {
        g.num_k_heads * n_ranks
    };
    let global_num_v_heads = g.num_v_heads * n_ranks;

    let (attn_qkv, attn_gate, ssm_alpha, ssm_beta, ssm_out, ssm_dt_bias, ssm_a, ssm_conv1d) =
        if matches!(shard, ShardMode::Replicated) {
            (
                upload_quant_weight(
                    file,
                    device,
                    spec.attn_qkv_name,
                    g.conv_channels * spec.hidden,
                    allocs,
                )?,
                upload_quant_weight(
                    file,
                    device,
                    spec.attn_gate_name,
                    g.d_inner * spec.hidden,
                    allocs,
                )?,
                upload_quant_weight(
                    file,
                    device,
                    spec.ssm_alpha_name,
                    g.num_v_heads * spec.hidden,
                    allocs,
                )?,
                upload_quant_weight(
                    file,
                    device,
                    spec.ssm_beta_name,
                    g.num_v_heads * spec.hidden,
                    allocs,
                )?,
                upload_quant_weight(
                    file,
                    device,
                    spec.ssm_out_name,
                    spec.hidden * g.d_inner,
                    allocs,
                )?,
                upload_f32_tensor(file, device, spec.ssm_dt_bias_name, g.num_v_heads, allocs)?,
                upload_f32_tensor(file, device, spec.ssm_a_name, g.num_v_heads, allocs)?,
                upload_f32_tensor(
                    file,
                    device,
                    spec.ssm_conv1d_name,
                    g.conv_kernel * g.conv_channels,
                    allocs,
                )?,
            )
        } else {
            (
                upload_gdn_fused_qkv_quant(
                    file,
                    device,
                    spec.attn_qkv_name,
                    crate::loader::gdn_shard::GdnHeadDims {
                        num_v_heads: global_num_v_heads,
                        num_k_heads: global_num_k_heads,
                        head_v_dim: g.head_v_dim,
                        head_k_dim: g.head_k_dim,
                    },
                    spec.hidden,
                    crate::loader::gdn_shard::GdnShardCtx {
                        kq_replicated,
                        rank,
                        n_ranks,
                    },
                    allocs,
                )?,
                upload_col_sharded_quant(
                    file,
                    device,
                    spec.attn_gate_name,
                    crate::loader::shard::ShardSpec {
                        n_rows: g.d_inner * n_ranks,
                        n_cols: spec.hidden,
                        rank,
                        n_ranks,
                    },
                    allocs,
                )?,
                upload_col_sharded_quant(
                    file,
                    device,
                    spec.ssm_alpha_name,
                    crate::loader::shard::ShardSpec {
                        n_rows: global_num_v_heads,
                        n_cols: spec.hidden,
                        rank,
                        n_ranks,
                    },
                    allocs,
                )?,
                upload_col_sharded_quant(
                    file,
                    device,
                    spec.ssm_beta_name,
                    crate::loader::shard::ShardSpec {
                        n_rows: global_num_v_heads,
                        n_cols: spec.hidden,
                        rank,
                        n_ranks,
                    },
                    allocs,
                )?,
                // ssm_out [hidden, d_inner] — row-shard along d_inner;
                // the partial-hidden output is AR-summed via the
                // composite's hook.
                upload_row_sharded_quant(
                    file,
                    device,
                    spec.ssm_out_name,
                    crate::loader::shard::ShardSpec {
                        n_rows: spec.hidden,
                        n_cols: g.d_inner * n_ranks,
                        rank,
                        n_ranks,
                    },
                    allocs,
                )?,
                upload_f32_array_sharded(
                    file,
                    device,
                    spec.ssm_dt_bias_name,
                    global_num_v_heads,
                    rank,
                    n_ranks,
                    allocs,
                )?,
                upload_f32_array_sharded(
                    file,
                    device,
                    spec.ssm_a_name,
                    global_num_v_heads,
                    rank,
                    n_ranks,
                    allocs,
                )?,
                upload_gdn_fused_qkv_f32(
                    file,
                    device,
                    spec.ssm_conv1d_name,
                    crate::loader::gdn_shard::GdnHeadDims {
                        num_v_heads: global_num_v_heads,
                        num_k_heads: global_num_k_heads,
                        head_v_dim: g.head_v_dim,
                        head_k_dim: g.head_k_dim,
                    },
                    g.conv_kernel,
                    crate::loader::gdn_shard::GdnShardCtx {
                        kq_replicated,
                        rank,
                        n_ranks,
                    },
                    allocs,
                )?,
            )
        };

    Ok(GdnWeights {
        attn_norm,
        attn_qkv,
        attn_gate,
        ssm_alpha,
        ssm_beta,
        ssm_out,
        ssm_dt_bias,
        ssm_a,
        ssm_conv1d,
        ssm_norm_w,
        dims: g,
        rms_eps: spec.rms_eps,
        rep_inner_layout: spec.rep_inner_layout,
    })
}
