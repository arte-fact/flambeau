//! Sarathi-Serve mixed-batch GDN (Phase K2).
//!
//! Rows `[0..K)` are a prefill chunk for slot `slot_p` (single slot,
//! K contiguous timesteps). Rows `[K..K+N)` are batched decodes
//! across N distinct slots. The split calls the two existing GDN
//! entry points back-to-back on sliced views of the shared input /
//! delta buffers:
//!
//! - K prefill rows → `DeltaNetLayer::forward_prefill_with_ar_hook`
//!   on slot_p's state + conv history.
//! - N decode rows → `DeltaNetLayer::forward_decode_with_ar_hook_batched_slots`
//!   on N slots' state + conv-history pointer arrays.
//!
//! AR fires once per phase (twice per layer). Matches the v1 driver
//! shape (see `project_lever1_mixed_batch_v1`).
//!
//! K2 slice: requires both `gdn_prefill_scratch` AND
//! `gdn_decode_batched_scratch` to be configured. Same caller
//! contract as standard_attn_mixed: prefill rows share `slot_ids[0]`;
//! decode rows must not collide with slot_p.
//!
//! See `doc/MIXED_BATCH_V2_PLAN.md`.

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

pub fn gdn_layer_mixed_local<H: TopologyHooks>(
    state: &mut CoreState<'_>,
    hooks: &mut H,
    input: &Tensor<F16>,
    weights: &GdnWeights,
    layer_idx: usize,
    slot_ids: &[usize],
    prefill_rows: usize,
    next_norm: Option<&Tensor<F16>>,
) -> Result<Option<Tensor<F16>>> {
    state.pool.input_pre_normed = false;
    let hidden = state.hidden();
    let n_tokens = slot_ids.len();
    let k = prefill_rows;
    if k == 0 || k >= n_tokens {
        bail!("gdn_layer_mixed: prefill_rows must satisfy 0 < K < n (got K={k}, n={n_tokens})");
    }
    let n_dec = n_tokens - k;
    let slot_p = slot_ids[0];
    for i in 0..k {
        if slot_ids[i] != slot_p {
            bail!(
                "gdn_layer_mixed: prefill row {i} slot {} != slot_p {slot_p}",
                slot_ids[i]
            );
        }
    }
    for i in k..n_tokens {
        if slot_ids[i] == slot_p {
            bail!(
                "gdn_layer_mixed: decode row {i} slot {} collides with prefill slot_p {slot_p}",
                slot_ids[i]
            );
        }
    }

    let local_idx = layer_idx
        .checked_sub(state.layer_idx_offset)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "gdn_layer_mixed: layer_idx {layer_idx} < layer_idx_offset {}",
                state.layer_idx_offset
            )
        })?;
    if local_idx >= state.pool.gdn_state.len() {
        bail!(
            "gdn_layer_mixed: local_idx {local_idx} >= gdn_state.len {}",
            state.pool.gdn_state.len()
        );
    }
    let max_slots = state.pool.config.max_slots.max(1);
    for (i, &slot) in slot_ids.iter().enumerate() {
        if slot >= max_slots {
            bail!("gdn_layer_mixed: slot_ids[{i}]={slot} >= max_slots {max_slots}");
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

    let prefill_scratch = state
        .pool
        .gdn_prefill_scratch
        .as_ref()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "gdn_layer_mixed: pool not configured for prefill \
                 (max_prefill_tokens <= 1 or gdn dims unset)"
            )
        })?
        .view();
    let batched_scratch = state
        .pool
        .gdn_decode_batched_scratch
        .as_ref()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "gdn_layer_mixed: pool needs gdn_decode_batched_scratch \
                 (max_slots > 1 and gdn dims set)"
            )
        })?
        .view();
    if n_dec > batched_scratch.max_slots {
        bail!(
            "gdn_layer_mixed: n_dec={n_dec} > batched_scratch.max_slots={}",
            batched_scratch.max_slots
        );
    }

    // ============ K prefill rows for slot_p ============
    {
        let state_ptr = layer_state.state.offset_bytes(slot_p * state_bytes_per_slot);
        let hist_ptr = layer_state
            .conv_history
            .offset_bytes(slot_p * hist_bytes_per_slot);
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
                n_tokens: k,
                state_event: None,
            },
            Some(&mut ar_cb),
        )?;
    }

    // ============ N decode rows for slots [K..K+N) ============
    let in_dec_ptr = input.ptr.offset_bytes(k * row_bytes);
    let out_dec_ptr = delta_ptr.offset_bytes(k * row_bytes);
    {
        let slot_state_ptrs = state.pool.gdn_slot_state_ptrs;
        let slot_hist_ptrs = state.pool.gdn_slot_history_ptrs;
        if slot_state_ptrs.as_usize() == 0 || slot_hist_ptrs.as_usize() == 0 {
            bail!(
                "gdn_layer_mixed: pool gdn_slot_state_ptrs / gdn_slot_history_ptrs unset \
                 (max_slots must be > 1)"
            );
        }
        let mut host_state_ptrs: Vec<u64> = Vec::with_capacity(n_dec);
        let mut host_hist_ptrs: Vec<u64> = Vec::with_capacity(n_dec);
        for i in 0..n_dec {
            let slot = slot_ids[k + i];
            let s = layer_state.state.offset_bytes(slot * state_bytes_per_slot);
            let h = layer_state
                .conv_history
                .offset_bytes(slot * hist_bytes_per_slot);
            host_state_ptrs.push(s.as_usize() as u64);
            host_hist_ptrs.push(h.as_usize() as u64);
        }
        // SAFETY: pool reserves max_slots * 8 bytes per array when
        // max_slots > 1; n_dec <= batched_scratch.max_slots <= max_slots.
        unsafe {
            state.device.memcpy_async(
                state.stream,
                flambeau_core::CopyDirection::HostToDevice,
                slot_state_ptrs,
                DevicePtr(host_state_ptrs.as_ptr() as usize),
                n_dec * 8,
            )?;
            state.device.memcpy_async(
                state.stream,
                flambeau_core::CopyDirection::HostToDevice,
                slot_hist_ptrs,
                DevicePtr(host_hist_ptrs.as_ptr() as usize),
                n_dec * 8,
            )?;
        }
        flambeau_core::Stream::synchronize(state.stream)?;
        let mut ar_cb =
            |buf: DevicePtr, n_elems: usize, dev: &HipDevice, stm: &HipStream| -> Result<()> {
                hooks.ar_sum_f32(buf, n_elems, dev, stm)
            };
        block.forward_decode_with_ar_hook_batched_slots(
            &ops,
            flambeau_model_ops::BackendCtx {
                device: state.device,
                stream: state.stream,
            },
            flambeau_model_ops::GdnDecodeBatchedBuffers {
                x_in_base: in_dec_ptr,
                delta_out_base: out_dec_ptr,
                state_in_ptrs_dev: slot_state_ptrs,
                state_out_ptrs_dev: slot_state_ptrs,
                conv_history_ptrs_dev: slot_hist_ptrs,
            },
            batched_scratch,
            n_dec,
            Some(&mut ar_cb),
        )?;
    }

    let _ = next_norm;
    Ok(Some(unsafe {
        Tensor::<F16>::from_raw(delta_ptr, n_tokens * hidden)
    }))
}
