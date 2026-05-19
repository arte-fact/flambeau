//! Dense full-attention layer loader. Supports plain Q and the
//! gated `[Q | gate]` head-interleaved layout (qwen3.5 / qwen3.6 /
//! qwen3-Next). Under `ShardMode::Tp`, n_heads / n_kv_heads divide
//! by n_ranks; Q/K/V are col-sharded along the head axis, output
//! proj is row-sharded along its input cols.

use anyhow::{bail, Result};
use flambeau_backend_hip::HipDevice;
use flambeau_core::DevicePtr;
use flambeau_quant::GgufFile;

use crate::ctx::AttnWeights;

use super::primitives::{upload_dequant_to_f16, upload_f16_ones};
use super::shard::{upload_col, upload_row};
use super::ShardMode;

pub struct DenseAttnLayerSpec<'a> {
    pub attn_norm_name: &'a str,
    /// Optional name of the F16 norm tensor applied to the attention
    /// delta BEFORE the outer residual add. Gemma4 sets this to
    /// `post_attention_norm.weight`; other arches pass `None`.
    pub post_attn_norm_name: Option<&'a str>,
    pub attn_q_name: &'a str,
    pub attn_k_name: &'a str,
    /// `None` for gemma4-style "V from K" layers (no attn_v on disk).
    pub attn_v_name: Option<&'a str>,
    pub attn_output_name: &'a str,
    pub attn_q_norm_name: Option<&'a str>,
    pub attn_k_norm_name: Option<&'a str>,
    /// `true` to allocate + upload a `[head_dim]` F16 tensor of ones
    /// used as the V-norm weight (gemma4 trained behavior).
    pub attn_v_unit_norm: bool,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub hidden: usize,
    pub rotated_dims: usize,
    pub rope_theta: f32,
    pub rope_variant: crate::ctx::RopeVariant,
    pub window_size: i32,
    pub rms_eps: f32,
    pub softmax_scale: Option<f32>,
    /// qwen3.5 / qwen3.6 / qwen3-Next: `attn_q` on disk has
    /// `[2 * n_heads * head_dim, hidden]` rows in head-interleaved
    /// `[head_i_Q | head_i_gate]` layout. The composite splits the
    /// Q-projection output per head and applies a sigmoid gate after
    /// attention. `false` for plain-Q arches (gemma4 dense).
    pub attn_q_gated: bool,
}

pub fn load_dense_attn_layer(
    file: &GgufFile,
    device: &HipDevice,
    spec: &DenseAttnLayerSpec,
    shard: ShardMode,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<AttnWeights> {
    let n_ranks = shard.n_ranks();
    if spec.n_heads % n_ranks != 0 {
        bail!(
            "dense_attn: n_heads {} not divisible by n_ranks {n_ranks}",
            spec.n_heads
        );
    }
    if spec.n_kv_heads % n_ranks != 0 {
        bail!(
            "dense_attn: n_kv_heads {} not divisible by n_ranks {n_ranks}",
            spec.n_kv_heads
        );
    }

    let q_width = spec.n_heads * spec.head_dim;
    let kv_width = spec.n_kv_heads * spec.head_dim;
    let n_heads_local = spec.n_heads / n_ranks;
    let n_kv_heads_local = spec.n_kv_heads / n_ranks;

    let attn_norm = upload_dequant_to_f16(file, device, spec.attn_norm_name, spec.hidden, allocs)?;
    let post_attn_norm = spec
        .post_attn_norm_name
        .map(|n| upload_dequant_to_f16(file, device, n, spec.hidden, allocs))
        .transpose()?;
    // Gated `attn_q`: head-interleaved layout, so col-shard with
    // doubled n_rows cleanly partitions heads + their gates.
    let attn_q_rows = if spec.attn_q_gated { 2 * q_width } else { q_width };
    let attn_q = upload_col(file, device, spec.attn_q_name, attn_q_rows, spec.hidden, shard, allocs)?;
    let attn_k = upload_col(file, device, spec.attn_k_name, kv_width, spec.hidden, shard, allocs)?;
    let attn_v = if let Some(name) = spec.attn_v_name {
        Some(upload_col(file, device, name, kv_width, spec.hidden, shard, allocs)?)
    } else {
        None
    };
    let attn_output = upload_row(
        file,
        device,
        spec.attn_output_name,
        spec.hidden,
        q_width,
        shard,
        allocs,
    )?;
    let attn_q_norm = spec
        .attn_q_norm_name
        .map(|n| upload_dequant_to_f16(file, device, n, spec.head_dim, allocs))
        .transpose()
        .ok()
        .flatten();
    let attn_k_norm = spec
        .attn_k_norm_name
        .map(|n| upload_dequant_to_f16(file, device, n, spec.head_dim, allocs))
        .transpose()
        .ok()
        .flatten();
    let attn_v_unit_norm_w = if spec.attn_v_unit_norm {
        Some(upload_f16_ones(device, spec.head_dim, allocs)?)
    } else {
        None
    };

    Ok(AttnWeights {
        attn_norm,
        post_attn_norm,
        attn_q,
        attn_k,
        attn_v,
        attn_output,
        attn_q_norm,
        attn_k_norm,
        attn_v_unit_norm_w,
        n_heads: n_heads_local,
        n_kv_heads: n_kv_heads_local,
        head_dim: spec.head_dim,
        rotated_dims: spec.rotated_dims,
        rope_theta: spec.rope_theta,
        rope_variant: spec.rope_variant,
        window_size: spec.window_size,
        rms_eps: spec.rms_eps,
        softmax_scale: spec.softmax_scale,
        attn_q_gated: spec.attn_q_gated,
    })
}
