//! rmsnorm-quant → Q/K/V proj → optional Q/K norm → RoPE → KV append
//! → attn_decode (N=1) / attn_prefill (N>1) → quantise → output proj
//! (with TP AR) → cast.

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
    start_position: usize,
    n_tokens: usize,
) -> Result<Tensor<F16>> {
    let hidden = state.hidden();
    let n = n_tokens;
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
    if start_position + n > state.pool.config.max_seq_len {
        bail!(
            "standard_attn: start_position {start_position} + n_tokens {n} > max_seq_len {}",
            state.pool.config.max_seq_len
        );
    }
    if n > state.pool.config.max_prefill_tokens {
        bail!(
            "standard_attn: n_tokens {n} > max_prefill_tokens {}",
            state.pool.config.max_prefill_tokens
        );
    }
    let q_width = weights.n_heads * weights.head_dim;
    let kv_width = weights.n_kv_heads * weights.head_dim;
    if q_width > state.pool.config.q_width {
        bail!(
            "standard_attn: weights q_width {q_width} > ctx.q_width {} (scratch too small)",
            state.pool.config.q_width
        );
    }
    if kv_width > state.pool.config.kv_width {
        bail!(
            "standard_attn: weights kv_width {kv_width} > ctx.kv_width {} (scratch too small)",
            state.pool.config.kv_width
        );
    }
    let slot_kv_width = state.pool.kv_caches[local_idx].kv_width;
    if slot_kv_width != kv_width {
        bail!(
            "standard_attn: kv_caches[{local_idx}].kv_width {slot_kv_width} != \
             weights kv_width {kv_width} (per-layer KV cache sizing mismatch)"
        );
    }

    let ops = state.ops();

    let mut norm_q8_1 = unsafe { Tensor::<Q8_1>::from_raw(state.pool.norm_q8_1, n * hidden) };
    let act_mmq_null = unsafe { Tensor::<Q8_1>::from_raw(DevicePtr::NULL, 0) };
    let norm_mmq_t;
    let act_norm_mmq: &Tensor<Q8_1> = if n > 1 {
        let mut norm_f16 =
            unsafe { Tensor::<F16>::from_raw(state.pool.norm, n * hidden) };
        flambeau_model_ops::rmsnorm_f16(
            input,
            &weights.attn_norm,
            &mut norm_f16,
            n,
            hidden,
            weights.rms_eps,
            &ops,
        )?;
        flambeau_model_ops::quantize_f16_to_q8_1(&norm_f16, &mut norm_q8_1, n * hidden, &ops)?;
        let mut norm_mmq =
            unsafe { Tensor::<Q8_1>::from_raw(state.pool.norm_q8_1_mmq, n * hidden) };
        flambeau_model_ops::quantize_f16_to_q8_1_mmq(&norm_f16, &mut norm_mmq, hidden, n, &ops)?;
        norm_mmq_t = norm_mmq;
        &norm_mmq_t
    } else {
        flambeau_model_ops::rmsnorm_quant_q8_1(
            input,
            &weights.attn_norm,
            &mut norm_q8_1,
            n,
            hidden,
            weights.rms_eps,
            &ops,
        )?;
        &act_mmq_null
    };
    let q_f32_buf = state.pool.attn_proj_f32;
    if weights.attn_q_gated {
        let fused_n = 2 * q_width;
        let mut q_fused_f32 = unsafe { Tensor::<F32>::from_raw(q_f32_buf, n * fused_n) };
        weights.attn_q.qmatmul(
            &norm_q8_1,
            act_norm_mmq,
            &mut q_fused_f32,
            n,
            hidden,
            fused_n,
            &ops,
        )?;
        let q_fused = unsafe { Tensor::<F16>::from_raw(state.pool.q_fused_f16, n * fused_n) };
        let mut q_fused_mut =
            unsafe { Tensor::<F16>::from_raw(state.pool.q_fused_f16, n * fused_n) };
        flambeau_model_ops::cast_f32_to_f16(&q_fused_f32, &mut q_fused_mut, n * fused_n, &ops)?;
        let mut q_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.q_f16, n * q_width) };
        let mut gate_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.gate_f16, n * q_width) };
        flambeau_model_ops::split_q_gate_f16(
            &q_fused,
            &mut q_f16,
            &mut gate_f16,
            n,
            weights.n_heads,
            weights.head_dim,
            &ops,
        )?;
    } else {
        let mut q_f32 = unsafe { Tensor::<F32>::from_raw(q_f32_buf, n * q_width) };
        weights
            .attn_q
            .qmatmul(&norm_q8_1, act_norm_mmq, &mut q_f32, n, hidden, q_width, &ops)?;
        let mut q_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.q_f16, n * q_width) };
        flambeau_model_ops::cast_f32_to_f16(&q_f32, &mut q_f16, n * q_width, &ops)?;
    }

    let mut k_f32 = unsafe { Tensor::<F32>::from_raw(q_f32_buf, n * kv_width) };
    weights
        .attn_k
        .qmatmul(&norm_q8_1, act_norm_mmq, &mut k_f32, n, hidden, kv_width, &ops)?;
    let mut k_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.k_f16, n * kv_width) };
    flambeau_model_ops::cast_f32_to_f16(&k_f32, &mut k_f16, n * kv_width, &ops)?;

    if let Some(v_w) = weights.attn_v.as_ref() {
        let mut v_f32 = unsafe { Tensor::<F32>::from_raw(q_f32_buf, n * kv_width) };
        v_w.qmatmul(&norm_q8_1, act_norm_mmq, &mut v_f32, n, hidden, kv_width, &ops)?;
        let mut v_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.v_f16, n * kv_width) };
        flambeau_model_ops::cast_f32_to_f16(&v_f32, &mut v_f16, n * kv_width, &ops)?;
    } else {
        // Gemma4 V-from-K: V = K (no separate projection).
        let bytes = n * kv_width * 2;
        // SAFETY: k_f16 and v_f16 slots are both sized for max_prefill_tokens * kv_width F16.
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

    // Per-head Q/K norm over (n_tokens * n_heads) rows of head_dim.
    if let Some(q_norm_w) = weights.attn_q_norm.as_ref() {
        let q_normed = unsafe { Tensor::<F16>::from_raw(state.pool.q_f16, n * q_width) };
        let mut tmp =
            unsafe { Tensor::<F16>::from_raw(state.pool.attn_out_f16, n * q_width) };
        flambeau_model_ops::rmsnorm_f16(
            &q_normed,
            q_norm_w,
            &mut tmp,
            n * weights.n_heads,
            weights.head_dim,
            weights.rms_eps,
            &ops,
        )?;
        let bytes = n * q_width * 2;
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
        let k_normed = unsafe { Tensor::<F16>::from_raw(state.pool.k_f16, n * kv_width) };
        let mut tmp =
            unsafe { Tensor::<F16>::from_raw(state.pool.attn_out_f16, n * kv_width) };
        flambeau_model_ops::rmsnorm_f16(
            &k_normed,
            k_norm_w,
            &mut tmp,
            n * weights.n_kv_heads,
            weights.head_dim,
            weights.rms_eps,
            &ops,
        )?;
        let bytes = n * kv_width * 2;
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

    let positions: Vec<i32> = (0..n).map(|i| (start_position + i) as i32).collect();
    let pos_bytes = n * 4;
    // SAFETY: position_i32 sized max_prefill_tokens * i32.
    unsafe {
        state
            .device
            .memcpy_async(
                state.stream,
                CopyDirection::HostToDevice,
                state.pool.position_i32,
                DevicePtr(positions.as_ptr() as usize),
                pos_bytes,
            )
            .context("standard_attn: positions HtoD")?;
    }
    let pos_tensor = unsafe { Tensor::<I32>::from_raw(state.pool.position_i32, n) };
    let mut q_f16_rope = unsafe { Tensor::<F16>::from_raw(state.pool.q_f16, n * q_width) };
    let mut k_f16_rope = unsafe { Tensor::<F16>::from_raw(state.pool.k_f16, n * kv_width) };
    flambeau_model_ops::rope_neox_partial_f16(
        &mut q_f16_rope,
        &pos_tensor,
        weights.rope_theta,
        n,
        weights.n_heads,
        weights.head_dim,
        weights.rotated_dims,
        &ops,
    )?;
    flambeau_model_ops::rope_neox_partial_f16(
        &mut k_f16_rope,
        &pos_tensor,
        weights.rope_theta,
        n,
        weights.n_kv_heads,
        weights.head_dim,
        weights.rotated_dims,
        &ops,
    )?;
    let _ = weights.rope_variant;

    let kv = state.pool.kv_caches[local_idx];
    let mut k_cache =
        unsafe { Tensor::<F16>::from_raw(kv.k, state.pool.config.max_seq_len * kv_width) };
    let mut v_cache =
        unsafe { Tensor::<F16>::from_raw(kv.v, state.pool.config.max_seq_len * kv_width) };
    let v_f16_view = unsafe { Tensor::<F16>::from_raw(state.pool.v_f16, n * kv_width) };
    flambeau_model_ops::kv_append_f16(
        &k_f16_rope,
        &v_f16_view,
        &mut k_cache,
        &mut v_cache,
        n,
        kv_width,
        start_position,
        state.pool.config.max_seq_len,
        state.device,
        state.stream,
    )?;

    let scale = weights
        .softmax_scale
        .unwrap_or_else(|| (weights.head_dim as f32).sqrt().recip());
    let mut attn_out = unsafe { Tensor::<F16>::from_raw(state.pool.attn_out_f16, n * q_width) };
    if n == 1 {
        let n_tokens_kv = start_position + 1;
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
    } else {
        let n_k_tokens = start_position + n;
        flambeau_model_ops::attn_prefill_f16(
            &q_f16_rope,
            &k_cache,
            &v_cache,
            &mut attn_out,
            n,
            weights.n_heads,
            weights.n_kv_heads,
            weights.head_dim,
            n_k_tokens,
            start_position,
            scale,
            weights.window_size,
            &ops,
        )?;
    }

    let post_attn_ptr = if weights.attn_q_gated {
        let gate = unsafe { Tensor::<F16>::from_raw(state.pool.gate_f16, n * q_width) };
        let attn_in = unsafe { Tensor::<F16>::from_raw(state.pool.attn_out_f16, n * q_width) };
        let mut gated_out =
            unsafe { Tensor::<F16>::from_raw(state.pool.q_fused_f16, n * q_width) };
        flambeau_model_ops::sigmoid_mul_f16(&gate, &attn_in, &mut gated_out, n * q_width, &ops)?;
        state.pool.q_fused_f16
    } else {
        state.pool.attn_out_f16
    };
    let _ = attn_out;
    let post_attn = unsafe { Tensor::<F16>::from_raw(post_attn_ptr, n * q_width) };

    let mut attn_out_q8_1 =
        unsafe { Tensor::<Q8_1>::from_raw(state.pool.attn_out_q8_1, n * q_width) };
    flambeau_model_ops::quantize_f16_to_q8_1(&post_attn, &mut attn_out_q8_1, n * q_width, &ops)?;
    let attn_out_mmq_t;
    let act_attn_out_mmq: &Tensor<Q8_1> = if n > 1 {
        let mut attn_out_mmq =
            unsafe { Tensor::<Q8_1>::from_raw(state.pool.attn_out_q8_1_mmq, n * q_width) };
        flambeau_model_ops::quantize_f16_to_q8_1_mmq(
            &post_attn,
            &mut attn_out_mmq,
            q_width,
            n,
            &ops,
        )?;
        attn_out_mmq_t = attn_out_mmq;
        &attn_out_mmq_t
    } else {
        &act_mmq_null
    };

    let mut proj_f32 = unsafe { Tensor::<F32>::from_raw(state.pool.attn_proj_f32, n * hidden) };
    weights.attn_output.qmatmul(
        &attn_out_q8_1,
        act_attn_out_mmq,
        &mut proj_f32,
        n,
        q_width,
        hidden,
        &ops,
    )?;
    hooks.ar_sum_f32(proj_f32.ptr, n * hidden, state.device, state.stream)?;
    let mut delta = unsafe { Tensor::<F16>::from_raw(state.pool.delta, n * hidden) };
    flambeau_model_ops::cast_f32_to_f16(&proj_f32, &mut delta, n * hidden, &ops)?;
    Ok(delta)
}
