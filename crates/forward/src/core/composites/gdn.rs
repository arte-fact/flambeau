//! GDN (Gated-Delta-Net) recurrent layer. N>1 loops the per-token
//! decode step internally — the recurrent state update is sequential.
//! Batched-GDN (project all N tokens then sequentially state-step) is
//! a perf follow-up; this is correctness-only.

use anyhow::{bail, Result};
use flambeau_backend_hip::{HipDevice, HipStream};
use flambeau_blocks::{DeltaNetLayer, WeightHandle};
use flambeau_core::DevicePtr;
use flambeau_model_ops::{Tensor, F16};

use crate::core::{CoreState, TopologyHooks};
use crate::ctx::{GdnWeights, QuantWeight};

fn quant_handle(qw: &QuantWeight, dims: [usize; 2]) -> WeightHandle {
    WeightHandle {
        ptr: qw.ptr,
        dtype: qw.dtype,
        dims,
    }
}

pub fn gdn_layer_local<H: TopologyHooks>(
    state: &mut CoreState<'_>,
    hooks: &mut H,
    input: &Tensor<F16>,
    weights: &GdnWeights,
    layer_idx: usize,
    slot_ids: &[usize],
) -> Result<Tensor<F16>> {
    let hidden = state.hidden();
    let n_tokens = slot_ids.len();
    if n_tokens == 0 {
        bail!("gdn_layer: empty slot_ids");
    }
    let local_idx = layer_idx.checked_sub(state.layer_idx_offset).ok_or_else(|| {
        anyhow::anyhow!(
            "gdn_layer: layer_idx {layer_idx} < layer_idx_offset {}",
            state.layer_idx_offset
        )
    })?;
    if local_idx >= state.pool.gdn_state.len() {
        bail!(
            "gdn_layer: local_idx {local_idx} >= gdn_state.len {}",
            state.pool.gdn_state.len()
        );
    }
    let max_slots = state.pool.config.max_slots.max(1);
    for (i, &slot) in slot_ids.iter().enumerate() {
        if slot >= max_slots {
            bail!("gdn_layer: slot_ids[{i}]={slot} >= max_slots {max_slots}");
        }
    }
    let dims = weights.dims;
    let scratch = state
        .pool
        .gdn_decode_scratch
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("gdn_layer: pool not configured with GDN dims"))?
        .view();
    let layer_state = state.pool.gdn_state[local_idx];
    let state_bytes_per_slot =
        dims.num_v_heads * dims.head_k_dim * dims.head_v_dim * 4;
    let hist_bytes_per_slot = (dims.conv_kernel - 1) * dims.conv_channels * 4;

    let block = DeltaNetLayer::new(
        quant_handle(&weights.attn_qkv, [dims.conv_channels, hidden]),
        quant_handle(&weights.attn_gate, [dims.d_inner, hidden]),
        quant_handle(&weights.ssm_alpha, [dims.num_v_heads, hidden]),
        quant_handle(&weights.ssm_beta, [dims.num_v_heads, hidden]),
        quant_handle(&weights.ssm_out, [hidden, dims.d_inner]),
        weights.ssm_dt_bias.ptr,
        weights.ssm_a.ptr,
        weights.ssm_conv1d.ptr,
        weights.ssm_norm_w.ptr,
        weights.attn_norm.ptr,
        hidden,
        dims.d_inner,
        dims.num_v_heads,
        dims.num_k_heads,
        dims.head_k_dim,
        dims.head_v_dim,
        dims.conv_channels,
        dims.conv_kernel,
        weights.rms_eps,
        weights.rep_inner_layout,
    )?;

    let delta_ptr = state.pool.delta;
    let row_bytes = hidden * 2;
    let ops = state.ops();
    let mut ar_cb = |buf: DevicePtr, n_elems: usize, dev: &HipDevice, stm: &HipStream| -> Result<()> {
        hooks.ar_sum_f32(buf, n_elems, dev, stm)
    };
    for i in 0..n_tokens {
        let slot = slot_ids[i];
        let in_i = input.ptr.offset_bytes(i * row_bytes);
        let out_i = delta_ptr.offset_bytes(i * row_bytes);
        let state_i = layer_state.state.offset_bytes(slot * state_bytes_per_slot);
        let hist_i = layer_state.conv_history.offset_bytes(slot * hist_bytes_per_slot);
        block.forward_decode_with_ar_hook(
            &ops,
            state.device,
            state.stream,
            in_i,
            out_i,
            state_i,
            hist_i,
            scratch,
            Some(&mut ar_cb),
        )?;
    }
    // SAFETY: `delta_ptr` is the pool's `n_tokens * hidden` F16 slot.
    Ok(unsafe { Tensor::<F16>::from_raw(delta_ptr, n_tokens * hidden) })
}
