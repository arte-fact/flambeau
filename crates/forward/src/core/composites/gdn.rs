//! GDN (Gated-Delta-Net) recurrent layer.
//!
//! Three dispatch paths:
//! - `n_tokens == 1`: single-slot decode → `forward_decode_with_ar_hook`.
//! - `n_tokens > 1` AND single-slot + contiguous positions (prefill shape)
//!   → `forward_prefill_with_ar_hook`. All projections + conv + state-step
//!   fire at batched n_tokens=N, dispatching MMQ tile8 kernels at L≥32
//!   and saving N-1 launches per layer for the inner pointwise ops.
//! - `n_tokens > 1` with multi-slot (batched-decode): per-slot loop. The
//!   recurrent state is intrinsically per-slot so a full batched-slots
//!   port is a separate, larger slice; per legacy memory the hybrid
//!   wall stays ~1× even after the full port.

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

    let single_slot = slot_ids.iter().all(|&s| s == slot_ids[0]);
    let prefill_shape = n_tokens > 1 && single_slot;

    if prefill_shape {
        let prefill_scratch = state
            .pool
            .gdn_prefill_scratch
            .as_ref()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "gdn_layer: pool not configured for prefill (max_prefill_tokens <= 1 \
                     or gdn dims unset)"
                )
            })?
            .view();
        let slot = slot_ids[0];
        let state_ptr = layer_state.state.offset_bytes(slot * state_bytes_per_slot);
        let hist_ptr = layer_state.conv_history.offset_bytes(slot * hist_bytes_per_slot);
        let mut ar_cb =
            |buf: DevicePtr, n_elems: usize, dev: &HipDevice, stm: &HipStream| -> Result<()> {
                hooks.ar_sum_f32(buf, n_elems, dev, stm)
            };
        block.forward_prefill_with_ar_hook(
            &ops,
            state.device,
            state.stream,
            input.ptr,
            delta_ptr,
            state_ptr,
            hist_ptr,
            prefill_scratch,
            n_tokens,
            None,
            Some(&mut ar_cb),
        )?;
    } else {
        let scratch = state
            .pool
            .gdn_decode_scratch
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("gdn_layer: pool not configured with GDN dims"))?
            .view();
        let mut ar_cb =
            |buf: DevicePtr, n_elems: usize, dev: &HipDevice, stm: &HipStream| -> Result<()> {
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
    }
    Ok(unsafe { Tensor::<F16>::from_raw(delta_ptr, n_tokens * hidden) })
}
