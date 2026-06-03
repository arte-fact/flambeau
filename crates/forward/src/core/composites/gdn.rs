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
use flambeau_core::{Device, DevicePtr};
use flambeau_model_ops::delta_net::DeltaNetLayer;
use flambeau_model_ops::WeightHandle;
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
    next_norm: Option<&Tensor<F16>>,
) -> Result<Option<Tensor<F16>>> {
    state.pool.input_pre_normed = false;
    let hidden = state.hidden();
    let n_tokens = slot_ids.len();
    if n_tokens == 0 {
        bail!("gdn_layer: empty slot_ids");
    }
    let local_idx = layer_idx
        .checked_sub(state.layer_idx_offset)
        .ok_or_else(|| {
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
    let state_bytes_per_slot = dims.num_v_heads * dims.head_k_dim * dims.head_v_dim * 4;
    let hist_bytes_per_slot = (dims.conv_kernel - 1) * dims.conv_channels * 4;

    let block = DeltaNetLayer::new(
        flambeau_model_ops::DeltaNetWeights {
            attn_qkv: quant_handle(&weights.attn_qkv, [dims.conv_channels, hidden]),
            attn_gate: quant_handle(&weights.attn_gate, [dims.d_inner, hidden]),
            ssm_alpha: quant_handle(&weights.ssm_alpha, [dims.num_v_heads, hidden]),
            ssm_beta: quant_handle(&weights.ssm_beta, [dims.num_v_heads, hidden]),
            ssm_out: quant_handle(&weights.ssm_out, [hidden, dims.d_inner]),
            ssm_dt_bias: weights.ssm_dt_bias.ptr,
            ssm_a: weights.ssm_a.ptr,
            ssm_conv1d: weights.ssm_conv1d.ptr,
            ssm_norm_w: weights.ssm_norm_w.ptr,
            attn_norm_w: weights.attn_norm.ptr,
        },
        flambeau_model_ops::DeltaNetDims {
            hidden,
            d_inner: dims.d_inner,
            num_v_heads: dims.num_v_heads,
            num_k_heads: dims.num_k_heads,
            head_k_dim: dims.head_k_dim,
            head_v_dim: dims.head_v_dim,
            conv_channels: dims.conv_channels,
            conv_kernel: dims.conv_kernel,
        },
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
        let hist_ptr = layer_state
            .conv_history
            .offset_bytes(slot * hist_bytes_per_slot);
        let mut ar_cb =
            |buf: DevicePtr, n_elems: usize, dev: &HipDevice, stm: &HipStream| -> Result<()> {
                hooks.ar_sum_f32(buf, n_elems, dev, stm)
            };
        block.forward_prefill_with_ar_hook(
            &ops,
            flambeau_model_ops::BackendCtx {
                device: state.device,
                stream: state.stream,
            },
            flambeau_model_ops::GdnDecodeBuffers {
                x_in: input.ptr,
                delta_out: delta_ptr,
                state: state_ptr,
                conv_history: hist_ptr,
            },
            prefill_scratch,
            flambeau_model_ops::GdnPrefillSeq {
                n_tokens,
                state_event: None,
            },
            Some(&mut ar_cb),
        )?;
    } else {
        let scratch = state
            .pool
            .gdn_decode_scratch
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("gdn_layer: pool not configured with GDN dims"))?
            .view();
        let fused_path = n_tokens == 1 && hooks.supports_ar_residual_f16();
        if fused_path {
            let slot = slot_ids[0];
            let state_i = layer_state.state.offset_bytes(slot * state_bytes_per_slot);
            let hist_i = layer_state
                .conv_history
                .offset_bytes(slot * hist_bytes_per_slot);
            block.forward_decode_with_ar_hook(
                &ops,
                flambeau_model_ops::BackendCtx {
                    device: state.device,
                    stream: state.stream,
                },
                flambeau_model_ops::GdnDecodeBuffers {
                    x_in: input.ptr,
                    delta_out: delta_ptr,
                    state: state_i,
                    conv_history: hist_i,
                },
                scratch,
                None,
            )?;
            let fuse_norm = next_norm.is_some() && hooks.supports_ar_residual_rmsnorm_f16();
            if fuse_norm {
                let next_w = next_norm.unwrap();
                hooks.ar_residual_rmsnorm_f16(
                    crate::core::ArResidualRmsNormHookBuffers {
                        residual_inout: input.ptr,
                        partial_f16: delta_ptr,
                        rms_weight: next_w.ptr,
                        out_norm: state.pool.norm,
                    },
                    hidden,
                    weights.rms_eps,
                    state.device,
                    state.stream,
                )?;
                state.pool.input_pre_normed = true;
            } else {
                hooks.ar_residual_f16(input.ptr, delta_ptr, hidden, state.device, state.stream)?;
            }
            return Ok(None);
        }
        let mut ar_cb =
            |buf: DevicePtr, n_elems: usize, dev: &HipDevice, stm: &HipStream| -> Result<()> {
                hooks.ar_sum_f32(buf, n_elems, dev, stm)
            };
        if let Some(batched_scratch) = state.pool.gdn_decode_batched_scratch.as_ref() {
            let batched_view = batched_scratch.view();
            if n_tokens > batched_view.max_slots {
                bail!(
                    "gdn_layer batched: n_tokens={n_tokens} > max_slots={}",
                    batched_view.max_slots
                );
            }
            let slot_state_ptrs = state.pool.gdn_slot_state_ptrs;
            let slot_hist_ptrs = state.pool.gdn_slot_history_ptrs;
            let mut host_state_ptrs: Vec<u64> = Vec::with_capacity(n_tokens);
            let mut host_hist_ptrs: Vec<u64> = Vec::with_capacity(n_tokens);
            for &slot in slot_ids.iter() {
                let s = layer_state.state.offset_bytes(slot * state_bytes_per_slot);
                let h = layer_state
                    .conv_history
                    .offset_bytes(slot * hist_bytes_per_slot);
                host_state_ptrs.push(s.as_usize() as u64);
                host_hist_ptrs.push(h.as_usize() as u64);
            }
            // SAFETY: pool reserves n_tokens*8 bytes per array when
            // max_slots > 1; n_tokens ≤ max_slots checked above.
            unsafe {
                state.device.memcpy_async(
                    state.stream,
                    flambeau_core::CopyDirection::HostToDevice,
                    slot_state_ptrs,
                    DevicePtr(host_state_ptrs.as_ptr() as usize),
                    n_tokens * 8,
                )?;
                state.device.memcpy_async(
                    state.stream,
                    flambeau_core::CopyDirection::HostToDevice,
                    slot_hist_ptrs,
                    DevicePtr(host_hist_ptrs.as_ptr() as usize),
                    n_tokens * 8,
                )?;
            }
            flambeau_core::Stream::synchronize(state.stream)?;
            block.forward_decode_with_ar_hook_batched_slots(
                &ops,
                flambeau_model_ops::BackendCtx {
                    device: state.device,
                    stream: state.stream,
                },
                flambeau_model_ops::GdnDecodeBatchedBuffers {
                    x_in_base: input.ptr,
                    delta_out_base: delta_ptr,
                    state_in_ptrs_dev: slot_state_ptrs,
                    state_out_ptrs_dev: slot_state_ptrs,
                    conv_history_ptrs_dev: slot_hist_ptrs,
                },
                batched_view,
                n_tokens,
                Some(&mut ar_cb),
            )?;
        } else {
            for i in 0..n_tokens {
                let slot = slot_ids[i];
                let in_i = input.ptr.offset_bytes(i * row_bytes);
                let out_i = delta_ptr.offset_bytes(i * row_bytes);
                let state_i = layer_state.state.offset_bytes(slot * state_bytes_per_slot);
                let hist_i = layer_state
                    .conv_history
                    .offset_bytes(slot * hist_bytes_per_slot);
                block.forward_decode_with_ar_hook(
                    &ops,
                    flambeau_model_ops::BackendCtx {
                        device: state.device,
                        stream: state.stream,
                    },
                    flambeau_model_ops::GdnDecodeBuffers {
                        x_in: in_i,
                        delta_out: out_i,
                        state: state_i,
                        conv_history: hist_i,
                    },
                    scratch,
                    Some(&mut ar_cb),
                )?;
            }
        }
    }
    Ok(Some(unsafe {
        Tensor::<F16>::from_raw(delta_ptr, n_tokens * hidden)
    }))
}
