//! rmsnorm-quant → Q/K/V proj → optional Q/K norm → RoPE → KV append
//! → attn_decode → quantise → output proj (with TP AR) → cast.

use anyhow::{bail, Context, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr};
use flambeau_model_ops::{Tensor, F16, F32, I32, Q8_1};

use crate::core::{CoreState, TopologyHooks};
use crate::ctx::AttnWeights;

pub fn standard_attn_local<H: TopologyHooks>(
    state: &mut CoreState<'_>,
    hooks: &mut H,
    input: &Tensor<F16>,
    weights: &AttnWeights,
    layer_idx: usize,
    position: usize,
) -> Result<Tensor<F16>> {
    let hidden = state.hidden();
    let local_idx = layer_idx.checked_sub(state.layer_idx_offset).ok_or_else(|| {
        anyhow::anyhow!(
            "standard_attn: layer_idx {layer_idx} < layer_idx_offset {}",
            state.layer_idx_offset
        )
    })?;
    if local_idx >= state.pool.kv_caches.len() {
        bail!(
            "standard_attn: local_idx {local_idx} (layer_idx {layer_idx} - offset {}) >= owned_layers {}",
            state.layer_idx_offset,
            state.pool.kv_caches.len()
        );
    }
    if position >= state.pool.config.max_seq_len {
        bail!(
            "standard_attn: position {position} >= max_seq_len {}",
            state.pool.config.max_seq_len
        );
    }
    let q_width = weights.n_heads * weights.head_dim;
    let kv_width = weights.n_kv_heads * weights.head_dim;
    if q_width != state.pool.config.q_width {
        bail!(
            "standard_attn: weights q_width {q_width} != ctx.q_width {}",
            state.pool.config.q_width
        );
    }
    if kv_width != state.pool.config.kv_width {
        bail!(
            "standard_attn: weights kv_width {kv_width} != ctx.kv_width {}",
            state.pool.config.kv_width
        );
    }

    let ops = state.ops();

    let mut norm_q8_1 = unsafe { Tensor::<Q8_1>::from_raw(state.pool.norm_q8_1, hidden) };
    flambeau_model_ops::rmsnorm_quant_q8_1(
        input,
        &weights.attn_norm,
        &mut norm_q8_1,
        1,
        hidden,
        weights.rms_eps,
        &ops,
    )?;

    let act_mmq_null = unsafe { Tensor::<Q8_1>::from_raw(DevicePtr::NULL, 0) };
    let q_f32_buf = state.pool.attn_proj_f32;
    let mut q_f32 = unsafe { Tensor::<F32>::from_raw(q_f32_buf, q_width) };
    weights
        .attn_q
        .qmatmul(&norm_q8_1, &act_mmq_null, &mut q_f32, 1, hidden, q_width, &ops)?;
    let mut q_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.q_f16, q_width) };
    flambeau_model_ops::cast_f32_to_f16(&q_f32, &mut q_f16, q_width, &ops)?;

    let mut k_f32 = unsafe { Tensor::<F32>::from_raw(q_f32_buf, kv_width) };
    weights
        .attn_k
        .qmatmul(&norm_q8_1, &act_mmq_null, &mut k_f32, 1, hidden, kv_width, &ops)?;
    let mut k_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.k_f16, kv_width) };
    flambeau_model_ops::cast_f32_to_f16(&k_f32, &mut k_f16, kv_width, &ops)?;

    if let Some(v_w) = weights.attn_v.as_ref() {
        let mut v_f32 = unsafe { Tensor::<F32>::from_raw(q_f32_buf, kv_width) };
        v_w.qmatmul(&norm_q8_1, &act_mmq_null, &mut v_f32, 1, hidden, kv_width, &ops)?;
        let mut v_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.v_f16, kv_width) };
        flambeau_model_ops::cast_f32_to_f16(&v_f32, &mut v_f16, kv_width, &ops)?;
    } else {
        // Gemma4 V-from-K: V = K (no separate projection).
        let bytes = kv_width * 2;
        // SAFETY: k_f16 and v_f16 slots are both sized for kv_width F16.
        unsafe {
            state
                .device
                .memcpy_async(
                    state.stream,
                    CopyDirection::DeviceToDevice,
                    state.pool.v_f16,
                    state.pool.k_f16,
                    bytes,
                )
                .context("standard_attn: V-from-K DtoD memcpy")?;
        }
    }
    let _ = q_f32_buf;

    // Per-head Q/K norm. Output routed through `attn_out_f16` (the
    // only F16 slot sized to `q_width`) and copied back.
    if let Some(q_norm_w) = weights.attn_q_norm.as_ref() {
        let q_normed = unsafe { Tensor::<F16>::from_raw(state.pool.q_f16, q_width) };
        let mut tmp = unsafe { Tensor::<F16>::from_raw(state.pool.attn_out_f16, q_width) };
        flambeau_model_ops::rmsnorm_f16(
            &q_normed,
            q_norm_w,
            &mut tmp,
            weights.n_heads,
            weights.head_dim,
            weights.rms_eps,
            &ops,
        )?;
        let bytes = q_width * 2;
        unsafe {
            state
                .device
                .memcpy_async(
                    state.stream,
                    CopyDirection::DeviceToDevice,
                    state.pool.q_f16,
                    tmp.ptr,
                    bytes,
                )
                .context("standard_attn: q_norm DtoD copy back")?;
        }
        let _ = q_normed;
    }
    if let Some(k_norm_w) = weights.attn_k_norm.as_ref() {
        let k_normed = unsafe { Tensor::<F16>::from_raw(state.pool.k_f16, kv_width) };
        let mut tmp = unsafe { Tensor::<F16>::from_raw(state.pool.attn_out_f16, kv_width) };
        flambeau_model_ops::rmsnorm_f16(
            &k_normed,
            k_norm_w,
            &mut tmp,
            weights.n_kv_heads,
            weights.head_dim,
            weights.rms_eps,
            &ops,
        )?;
        let bytes = kv_width * 2;
        unsafe {
            state
                .device
                .memcpy_async(
                    state.stream,
                    CopyDirection::DeviceToDevice,
                    state.pool.k_f16,
                    tmp.ptr,
                    bytes,
                )
                .context("standard_attn: k_norm DtoD copy back")?;
        }
        let _ = k_normed;
    }

    let pos_val = [position as i32];
    unsafe {
        state
            .device
            .memcpy_async(
                state.stream,
                CopyDirection::HostToDevice,
                state.pool.position_i32,
                DevicePtr(pos_val.as_ptr() as usize),
                4,
            )
            .context("standard_attn: positions HtoD")?;
    }
    let positions = unsafe { Tensor::<I32>::from_raw(state.pool.position_i32, 1) };
    let mut q_f16_rope = unsafe { Tensor::<F16>::from_raw(state.pool.q_f16, q_width) };
    let mut k_f16_rope = unsafe { Tensor::<F16>::from_raw(state.pool.k_f16, kv_width) };
    if weights.rotated_dims == weights.head_dim {
        flambeau_model_ops::rope_f16(
            &mut q_f16_rope,
            &positions,
            weights.rope_theta,
            1,
            weights.n_heads,
            weights.head_dim,
            &ops,
        )?;
        flambeau_model_ops::rope_f16(
            &mut k_f16_rope,
            &positions,
            weights.rope_theta,
            1,
            weights.n_kv_heads,
            weights.head_dim,
            &ops,
        )?;
    } else {
        flambeau_model_ops::rope_neox_partial_f16(
            &mut q_f16_rope,
            &positions,
            weights.rope_theta,
            1,
            weights.n_heads,
            weights.head_dim,
            weights.rotated_dims,
            &ops,
        )?;
        flambeau_model_ops::rope_neox_partial_f16(
            &mut k_f16_rope,
            &positions,
            weights.rope_theta,
            1,
            weights.n_kv_heads,
            weights.head_dim,
            weights.rotated_dims,
            &ops,
        )?;
    }

    let kv = state.pool.kv_caches[local_idx];
    let mut k_cache =
        unsafe { Tensor::<F16>::from_raw(kv.k, state.pool.config.max_seq_len * kv_width) };
    let mut v_cache =
        unsafe { Tensor::<F16>::from_raw(kv.v, state.pool.config.max_seq_len * kv_width) };
    let v_f16_view = unsafe { Tensor::<F16>::from_raw(state.pool.v_f16, kv_width) };
    flambeau_model_ops::kv_append_f16(
        &k_f16_rope,
        &v_f16_view,
        &mut k_cache,
        &mut v_cache,
        1,
        kv_width,
        position,
        state.pool.config.max_seq_len,
        state.device,
        state.stream,
    )?;

    let n_tokens_kv = position + 1;
    let mut attn_out = unsafe { Tensor::<F16>::from_raw(state.pool.attn_out_f16, q_width) };
    let scale = weights
        .softmax_scale
        .unwrap_or_else(|| (weights.head_dim as f32).sqrt().recip());
    flambeau_model_ops::attn_decode_f16(
        &q_f16_rope,
        &k_cache,
        &v_cache,
        &mut attn_out,
        weights.n_heads,
        weights.n_kv_heads,
        weights.head_dim,
        n_tokens_kv,
        scale,
        weights.window_size,
        &ops,
    )?;

    let mut attn_out_q8_1 =
        unsafe { Tensor::<Q8_1>::from_raw(state.pool.attn_out_q8_1, q_width) };
    flambeau_model_ops::quantize_f16_to_q8_1(&attn_out, &mut attn_out_q8_1, q_width, &ops)?;

    // Row-parallel output proj — AR-sum collapses per-rank partials under TP.
    let mut proj_f32 = unsafe { Tensor::<F32>::from_raw(state.pool.attn_proj_f32, hidden) };
    weights.attn_output.qmatmul(
        &attn_out_q8_1,
        &act_mmq_null,
        &mut proj_f32,
        1,
        q_width,
        hidden,
        &ops,
    )?;
    hooks.ar_sum_f32(proj_f32.ptr, hidden, state.device, state.stream)?;
    let mut delta = unsafe { Tensor::<F16>::from_raw(state.pool.delta, hidden) };
    flambeau_model_ops::cast_f32_to_f16(&proj_f32, &mut delta, hidden, &ops)?;
    Ok(delta)
}
