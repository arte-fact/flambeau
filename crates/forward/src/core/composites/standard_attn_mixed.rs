//! Sarathi-Serve mixed-batch standard_attn (Phase K1a).
//!
//! Rows `[0..K)` are a prefill chunk for slot `slot_p` at contiguous
//! positions `[pos0, pos0 + K)`. Rows `[K..K+N)` are batched decodes
//! across N distinct slots at arbitrary positions. QKV / RoPE /
//! output-proj / AR all run once at `n = K + N` (weight amortisation).
//! Attention itself splits: existing `attn_prefill_f16` over the K
//! prefill rows + existing `attn_decode_f16_batched` over the N decode
//! rows. No new device kernels.
//!
//! K1a slice: bail on paged / kv-shared / `attn_v_unit_norm_w` /
//! `attn_q_gated` / `post_attn_norm` paths. Future slices handle
//! those. Clean Qwen3.5-9B-class dense path only.
//!
//! See `doc/MIXED_BATCH_V2_PLAN.md`.

use anyhow::{bail, Context, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr};
use flambeau_model_ops::{Tensor, F16, F32, I32, Q8_1};

use crate::core::{CoreState, TopologyHooks};
use crate::ctx::AttnWeights;

#[allow(clippy::too_many_arguments)]
pub fn standard_attn_mixed_local<H: TopologyHooks>(
    state: &mut CoreState<'_>,
    hooks: &mut H,
    input: &Tensor<F16>,
    weights: &AttnWeights,
    layer_idx: usize,
    positions: &[usize],
    slot_ids: &[usize],
    prefill_rows: usize,
    next_norm: Option<&Tensor<F16>>,
) -> Result<Option<Tensor<F16>>> {
    let input_pre_normed = state.pool.input_pre_normed;
    state.pool.input_pre_normed = false;
    let hidden = state.hidden();
    if positions.len() != slot_ids.len() {
        bail!(
            "standard_attn_mixed: positions.len {} != slot_ids.len {}",
            positions.len(),
            slot_ids.len()
        );
    }
    let n = positions.len();
    let k = prefill_rows;
    if k == 0 || k >= n {
        bail!(
            "standard_attn_mixed: prefill_rows must satisfy 0 < K < n (got K={k}, n={n})"
        );
    }
    let n_dec = n - k;
    let slot_p = slot_ids[0];
    // Prefill rows must all share slot_p and have contiguous positions.
    let pos0 = positions[0];
    for i in 0..k {
        if slot_ids[i] != slot_p {
            bail!(
                "standard_attn_mixed: prefill row {i} slot {} != slot_p {slot_p}",
                slot_ids[i]
            );
        }
        if positions[i] != pos0 + i {
            bail!(
                "standard_attn_mixed: prefill row {i} pos {} not contiguous from pos0={pos0}",
                positions[i]
            );
        }
    }
    // Decode rows must not collide with slot_p (Sarathi disjointness).
    for i in k..n {
        if slot_ids[i] == slot_p {
            bail!(
                "standard_attn_mixed: decode row {i} slot {} collides with prefill slot_p {slot_p}",
                slot_ids[i]
            );
        }
    }

    // K1a bails: structural shapes not yet supported in the mixed path.
    if state.pool.paged_kv_caches.is_some() {
        bail!("standard_attn_mixed: paged KV not yet supported (Phase K-paged)");
    }
    if weights.kv_share_src.is_some() {
        bail!("standard_attn_mixed: shared-KV layers not yet supported");
    }
    if weights.attn_v_unit_norm_w.is_some() {
        bail!("standard_attn_mixed: V unit-norm fusion (gemma4) not yet supported");
    }
    if weights.post_attn_norm.is_some() {
        bail!("standard_attn_mixed: post_attn_norm (gemma4) not yet supported");
    }
    if weights.window_size > 0 {
        bail!(
            "standard_attn_mixed: sliding-window attention not yet supported (window_size={})",
            weights.window_size
        );
    }

    let local_idx = layer_idx
        .checked_sub(state.layer_idx_offset)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "standard_attn_mixed: layer_idx {layer_idx} < layer_idx_offset {}",
                state.layer_idx_offset
            )
        })?;
    if local_idx >= state.pool.kv_caches.len() {
        bail!(
            "standard_attn_mixed: local_idx {local_idx} >= owned_layers {}",
            state.pool.kv_caches.len()
        );
    }
    let kv_local_idx = local_idx;
    let max_seq_len = state.pool.config.max_seq_len;
    let max_slots = state.pool.config.max_slots.max(1);
    for (i, (&pos, &slot)) in positions.iter().zip(slot_ids.iter()).enumerate() {
        if pos >= max_seq_len {
            bail!("standard_attn_mixed: positions[{i}]={pos} >= max_seq_len {max_seq_len}");
        }
        if slot >= max_slots {
            bail!("standard_attn_mixed: slot_ids[{i}]={slot} >= max_slots {max_slots}");
        }
    }
    if n > state.pool.config.max_prefill_tokens {
        bail!(
            "standard_attn_mixed: n_tokens {n} > max_prefill_tokens {}",
            state.pool.config.max_prefill_tokens
        );
    }
    if state.pool.attn_slot_k_dst_ptrs.as_usize() == 0 {
        bail!(
            "standard_attn_mixed: needs max_slots > 1 in ScratchConfig \
             (got max_slots={})",
            state.pool.config.max_slots
        );
    }

    let q_width = weights.n_heads * weights.head_dim;
    let kv_width = weights.n_kv_heads * weights.head_dim;
    if q_width > state.pool.config.q_width {
        bail!(
            "standard_attn_mixed: q_width {q_width} > ctx.q_width {}",
            state.pool.config.q_width
        );
    }
    if kv_width > state.pool.config.kv_width {
        bail!(
            "standard_attn_mixed: kv_width {kv_width} > ctx.kv_width {}",
            state.pool.config.kv_width
        );
    }
    let slot_kv_width = state.pool.kv_caches[kv_local_idx].kv_width;
    if slot_kv_width != kv_width {
        bail!(
            "standard_attn_mixed: kv_caches[{kv_local_idx}].kv_width {slot_kv_width} != \
             weights kv_width {kv_width}"
        );
    }

    let ops = state.ops();

    // ---- Norm + Q8_1 quantise (n > 1 path — n = K + N >= 2 by guard) ----
    let mut norm_q8_1 = unsafe { Tensor::<Q8_1>::from_raw(state.pool.norm_q8_1, n * hidden) };
    let mut norm_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.norm, n * hidden) };
    if input_pre_normed {
        // Norm already in state.pool.norm; just re-quantise.
        flambeau_model_ops::quantize_f16_to_q8_1(&norm_f16, &mut norm_q8_1, n * hidden, &ops)?;
    } else {
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
    }
    let mut norm_mmq =
        unsafe { Tensor::<Q8_1>::from_raw(state.pool.norm_q8_1_mmq, n * hidden) };
    flambeau_model_ops::quantize_f16_to_q8_1_mmq(&norm_f16, &mut norm_mmq, hidden, n, &ops)?;
    let norm_mmq_t = norm_mmq;
    let act_norm_mmq: &Tensor<Q8_1> = &norm_mmq_t;

    // ---- Q projection (n = K + N rows). Gated path produces a fused
    // [n, 2*q_width] F32, casts to F16, then splits Q and the per-head
    // gate (consumed post-attention). Non-gated path emits Q only.
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
        weights.attn_q.qmatmul(
            &norm_q8_1,
            act_norm_mmq,
            &mut q_f32,
            n,
            hidden,
            q_width,
            &ops,
        )?;
        let mut q_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.q_f16, n * q_width) };
        flambeau_model_ops::cast_f32_to_f16(&q_f32, &mut q_f16, n * q_width, &ops)?;
    }

    // ---- K projection ----
    {
        let mut k_f32 = unsafe { Tensor::<F32>::from_raw(q_f32_buf, n * kv_width) };
        weights.attn_k.qmatmul(
            &norm_q8_1,
            act_norm_mmq,
            &mut k_f32,
            n,
            hidden,
            kv_width,
            &ops,
        )?;
        let mut k_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.k_f16, n * kv_width) };
        flambeau_model_ops::cast_f32_to_f16(&k_f32, &mut k_f16, n * kv_width, &ops)?;
    }

    // ---- V projection (require separate attn_v — K1a bails on V-from-K) ----
    let Some(v_w) = weights.attn_v.as_ref() else {
        bail!("standard_attn_mixed: V-from-K (gemma4 fused) not yet supported");
    };
    {
        let mut v_f32 = unsafe { Tensor::<F32>::from_raw(q_f32_buf, n * kv_width) };
        v_w.qmatmul(
            &norm_q8_1,
            act_norm_mmq,
            &mut v_f32,
            n,
            hidden,
            kv_width,
            &ops,
        )?;
        let mut v_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.v_f16, n * kv_width) };
        flambeau_model_ops::cast_f32_to_f16(&v_f32, &mut v_f16, n * kv_width, &ops)?;
    }
    let _ = q_f32_buf;

    // ---- positions HtoD + RoPE on Q and K (n = K + N) ----
    let positions_i32: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
    let pos_bytes = n * 4;
    unsafe {
        state
            .device
            .memcpy_async(
                state.stream,
                CopyDirection::HostToDevice,
                state.pool.position_i32,
                DevicePtr(positions_i32.as_ptr() as usize),
                pos_bytes,
            )
            .context("standard_attn_mixed: positions HtoD")?;
    }
    let pos_tensor = unsafe { Tensor::<I32>::from_raw(state.pool.position_i32, n) };
    // Q-norm + RoPE: when `attn_q_norm` is set (Qwen3.5/3.6, gemma4),
    // fuse rmsnorm + rope into one in-place launch. Saves 2 launches
    // + 1 DtoD per layer per row.
    use flambeau_ops::Ops;
    if let Some(q_norm_w) = weights.attn_q_norm.as_ref() {
        ops.rmsnorm_rope_neox_partial_f16(
            state.pool.q_f16,
            q_norm_w.ptr,
            state.pool.position_i32,
            weights.rope_theta,
            weights.rms_eps,
            n,
            weights.n_heads,
            weights.head_dim,
            weights.rotated_dims,
        )?;
    } else {
        let mut q_f16_rope = unsafe { Tensor::<F16>::from_raw(state.pool.q_f16, n * q_width) };
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
    }
    if let Some(k_norm_w) = weights.attn_k_norm.as_ref() {
        ops.rmsnorm_rope_neox_partial_f16(
            state.pool.k_f16,
            k_norm_w.ptr,
            state.pool.position_i32,
            weights.rope_theta,
            weights.rms_eps,
            n,
            weights.n_kv_heads,
            weights.head_dim,
            weights.rotated_dims,
        )?;
    } else {
        let mut k_f16_rope = unsafe { Tensor::<F16>::from_raw(state.pool.k_f16, n * kv_width) };
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
    }
    let _ = weights.rope_variant;

    let kv = state.pool.kv_caches[kv_local_idx];
    let slot_stride_elems = max_seq_len * kv_width;
    let slot_stride_bytes = slot_stride_elems * 2;
    let scale = weights
        .softmax_scale
        .unwrap_or_else(|| (weights.head_dim as f32).sqrt().recip());

    // ============================================================
    //   Attention split:
    //     rows [0..K) → kv_append_f16 (range write) + attn_prefill_f16
    //     rows [K..n) → kv_append_f16_batched_slots + attn_decode_f16_batched
    //   Each operates on a sliced view of q_f16 / k_f16 / v_f16 /
    //   attn_out_f16 buffers via Tensor::from_raw with the byte offset.
    // ============================================================

    // ---- K prefill rows ----
    let q_pref_ptr = state.pool.q_f16;
    let k_pref_ptr = state.pool.k_f16;
    let v_pref_ptr = state.pool.v_f16;
    let out_pref_ptr = state.pool.attn_out_f16;

    let slot_offset_p = slot_p * slot_stride_bytes;
    let k_slot_ptr = kv.k.offset_bytes(slot_offset_p);
    let v_slot_ptr = kv.v.offset_bytes(slot_offset_p);
    {
        let k_pref_src = unsafe { Tensor::<F16>::from_raw(k_pref_ptr, k * kv_width) };
        let v_pref_src = unsafe { Tensor::<F16>::from_raw(v_pref_ptr, k * kv_width) };
        let mut k_cache = unsafe { Tensor::<F16>::from_raw(k_slot_ptr, slot_stride_elems) };
        let mut v_cache = unsafe { Tensor::<F16>::from_raw(v_slot_ptr, slot_stride_elems) };
        flambeau_model_ops::kv_append_f16(
            &k_pref_src,
            &v_pref_src,
            &mut k_cache,
            &mut v_cache,
            k,
            kv_width,
            pos0,
            max_seq_len,
            state.device,
            state.stream,
        )?;
        let q_pref = unsafe { Tensor::<F16>::from_raw(q_pref_ptr, k * q_width) };
        let mut out_pref = unsafe { Tensor::<F16>::from_raw(out_pref_ptr, k * q_width) };
        let n_k_tokens = pos0 + k;
        flambeau_model_ops::attn_prefill_f16(
            &q_pref,
            &k_cache,
            &v_cache,
            &mut out_pref,
            k,
            weights.n_heads,
            weights.n_kv_heads,
            weights.head_dim,
            n_k_tokens,
            pos0,
            scale,
            weights.window_size,
            &ops,
        )?;
    }

    // ---- N decode rows ----
    let q_dec_ptr = q_pref_ptr.offset_bytes(k * q_width * 2);
    let k_dec_ptr = k_pref_ptr.offset_bytes(k * kv_width * 2);
    let v_dec_ptr = v_pref_ptr.offset_bytes(k * kv_width * 2);
    let out_dec_ptr = out_pref_ptr.offset_bytes(k * q_width * 2);
    {
        let mut host_k_ptrs: Vec<u64> = Vec::with_capacity(n_dec);
        let mut host_v_ptrs: Vec<u64> = Vec::with_capacity(n_dec);
        let mut host_write_pos: Vec<i32> = Vec::with_capacity(n_dec);
        let mut host_n_kv: Vec<i32> = Vec::with_capacity(n_dec);
        for i in 0..n_dec {
            let slot = slot_ids[k + i];
            let pos = positions[k + i];
            let slot_offset = slot * slot_stride_bytes;
            host_k_ptrs.push(kv.k.offset_bytes(slot_offset).as_usize() as u64);
            host_v_ptrs.push(kv.v.offset_bytes(slot_offset).as_usize() as u64);
            host_write_pos.push(pos as i32);
            host_n_kv.push((pos + 1) as i32);
        }
        unsafe {
            state
                .device
                .memcpy_async(
                    state.stream,
                    CopyDirection::HostToDevice,
                    state.pool.attn_slot_k_dst_ptrs,
                    DevicePtr(host_k_ptrs.as_ptr() as usize),
                    n_dec * 8,
                )
                .context("standard_attn_mixed: attn_slot_k_dst_ptrs HtoD")?;
            state
                .device
                .memcpy_async(
                    state.stream,
                    CopyDirection::HostToDevice,
                    state.pool.attn_slot_v_dst_ptrs,
                    DevicePtr(host_v_ptrs.as_ptr() as usize),
                    n_dec * 8,
                )
                .context("standard_attn_mixed: attn_slot_v_dst_ptrs HtoD")?;
            state
                .device
                .memcpy_async(
                    state.stream,
                    CopyDirection::HostToDevice,
                    state.pool.attn_slot_write_pos,
                    DevicePtr(host_write_pos.as_ptr() as usize),
                    n_dec * 4,
                )
                .context("standard_attn_mixed: attn_slot_write_pos HtoD")?;
            state
                .device
                .memcpy_async(
                    state.stream,
                    CopyDirection::HostToDevice,
                    state.pool.attn_slot_n_kv,
                    DevicePtr(host_n_kv.as_ptr() as usize),
                    n_dec * 4,
                )
                .context("standard_attn_mixed: attn_slot_n_kv HtoD")?;
        }
        flambeau_core::Stream::synchronize(state.stream)?;
        let k_dec_src = unsafe { Tensor::<F16>::from_raw(k_dec_ptr, n_dec * kv_width) };
        let v_dec_src = unsafe { Tensor::<F16>::from_raw(v_dec_ptr, n_dec * kv_width) };
        flambeau_model_ops::kv_append_f16_batched_slots(
            &k_dec_src,
            &v_dec_src,
            state.pool.attn_slot_k_dst_ptrs,
            state.pool.attn_slot_v_dst_ptrs,
            state.pool.attn_slot_write_pos,
            n_dec,
            kv_width,
            &ops,
        )?;
        let q_dec = unsafe { Tensor::<F16>::from_raw(q_dec_ptr, n_dec * q_width) };
        let mut out_dec = unsafe { Tensor::<F16>::from_raw(out_dec_ptr, n_dec * q_width) };
        flambeau_model_ops::attn_decode_f16_batched(
            &q_dec,
            state.pool.attn_slot_k_dst_ptrs,
            state.pool.attn_slot_v_dst_ptrs,
            &mut out_dec,
            state.pool.attn_slot_n_kv,
            weights.n_heads,
            weights.n_kv_heads,
            weights.head_dim,
            n_dec,
            scale,
            &ops,
        )?;
    }

    // ---- Apply Q-gate sigmoid·mul if gated, then output proj on K+N rows ----
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
    let post_attn = unsafe { Tensor::<F16>::from_raw(post_attn_ptr, n * q_width) };
    let mut attn_out_q8_1 =
        unsafe { Tensor::<Q8_1>::from_raw(state.pool.attn_out_q8_1, n * q_width) };
    flambeau_model_ops::quantize_f16_to_q8_1(&post_attn, &mut attn_out_q8_1, n * q_width, &ops)?;
    let mut attn_out_mmq =
        unsafe { Tensor::<Q8_1>::from_raw(state.pool.attn_out_q8_1_mmq, n * q_width) };
    flambeau_model_ops::quantize_f16_to_q8_1_mmq(
        &post_attn,
        &mut attn_out_mmq,
        q_width,
        n,
        &ops,
    )?;
    let act_attn_out_mmq: &Tensor<Q8_1> = &attn_out_mmq;

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

    // ---- AR + residual + (optional) next-norm fold ----
    if hooks.supports_ar_residual_f16() && next_norm.is_none() {
        let mut partial_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.delta, n * hidden) };
        flambeau_model_ops::cast_f32_to_f16(&proj_f32, &mut partial_f16, n * hidden, &ops)?;
        hooks.ar_residual_f16(
            input.ptr,
            partial_f16.ptr,
            n * hidden,
            state.device,
            state.stream,
        )?;
        return Ok(None);
    }
    hooks.ar_sum_f32(proj_f32.ptr, n * hidden, state.device, state.stream)?;
    let mut delta_mut = unsafe { Tensor::<F16>::from_raw(state.pool.delta, n * hidden) };
    flambeau_model_ops::cast_f32_to_f16(&proj_f32, &mut delta_mut, n * hidden, &ops)?;
    let delta = unsafe { Tensor::<F16>::from_raw(state.pool.delta, n * hidden) };
    let _ = next_norm;
    Ok(Some(delta))
}
