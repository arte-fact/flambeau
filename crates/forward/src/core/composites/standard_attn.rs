//! rmsnorm-quant → Q/K/V proj → optional Q/K norm → RoPE → KV append
//! → attn_decode (N=1) / attn_prefill (N>1) → quantise → output proj
//! (with TP AR) → cast.

use anyhow::{bail, Context, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr};
use flambeau_model_ops::{Tensor, F16, F32, I32, Q8_1};

use crate::core::{CoreState, TopologyHooks};
use crate::ctx::AttnWeights;

#[allow(clippy::too_many_arguments)]
pub fn standard_attn_local<H: TopologyHooks>(
    state: &mut CoreState<'_>,
    hooks: &mut H,
    input: &Tensor<F16>,
    weights: &AttnWeights,
    layer_idx: usize,
    positions: &[usize],
    slot_ids: &[usize],
    next_norm: Option<&Tensor<F16>>,
) -> Result<Option<Tensor<F16>>> {
    let input_pre_normed = state.pool.input_pre_normed;
    state.pool.input_pre_normed = false;
    let hidden = state.hidden();
    if positions.len() != slot_ids.len() {
        bail!(
            "standard_attn: positions.len {} != slot_ids.len {}",
            positions.len(),
            slot_ids.len()
        );
    }
    let n = positions.len();
    if n == 0 {
        bail!("standard_attn: empty positions");
    }
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
    // gemma 4n shared-KV: read/write through the share-source's slot
    // and skip the K/V cache append. Source layer's own KV write
    // populated the cache on its earlier pass; this layer only reuses
    // it. `attn_q` is still this layer's, so Q projection runs.
    let kv_local_idx = match weights.kv_share_src {
        Some(src) => src.checked_sub(state.layer_idx_offset).ok_or_else(|| {
            anyhow::anyhow!(
                "standard_attn: kv_share_src {src} < layer_idx_offset {}",
                state.layer_idx_offset
            )
        })?,
        None => local_idx,
    };
    if kv_local_idx >= state.pool.kv_caches.len() {
        bail!(
            "standard_attn: kv_local_idx {kv_local_idx} (src {:?}) >= owned_layers {}",
            weights.kv_share_src,
            state.pool.kv_caches.len()
        );
    }
    let is_kv_shared = weights.kv_share_src.is_some();
    let max_seq_len = state.pool.config.max_seq_len;
    let max_slots = state.pool.config.max_slots.max(1);
    for (i, (&pos, &slot)) in positions.iter().zip(slot_ids.iter()).enumerate() {
        if pos >= max_seq_len {
            bail!("standard_attn: positions[{i}]={pos} >= max_seq_len {max_seq_len}");
        }
        if slot >= max_slots {
            bail!("standard_attn: slot_ids[{i}]={slot} >= max_slots {max_slots}");
        }
    }
    if n > state.pool.config.max_prefill_tokens {
        bail!(
            "standard_attn: n_tokens {n} > max_prefill_tokens {}",
            state.pool.config.max_prefill_tokens
        );
    }
    // Detect the prefill-shape pattern: all tokens on the same slot,
    // positions contiguous starting at positions[0]. That path uses the
    // batched attn_prefill kernel + a single kv_append range write.
    let single_slot = slot_ids.iter().all(|&s| s == slot_ids[0]);
    let contiguous = positions
        .iter()
        .enumerate()
        .all(|(i, &p)| p == positions[0] + i);
    let prefill_shape = single_slot && contiguous;
    let primary_slot = slot_ids[0];
    let start_position = positions[0];
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
    let slot_kv_width = state.pool.kv_caches[kv_local_idx].kv_width;
    if slot_kv_width != kv_width {
        bail!(
            "standard_attn: kv_caches[{kv_local_idx}].kv_width {slot_kv_width} != \
             weights kv_width {kv_width} (per-layer KV cache sizing mismatch)"
        );
    }

    let ops = state.ops();

    let mut norm_q8_1 = unsafe { Tensor::<Q8_1>::from_raw(state.pool.norm_q8_1, n * hidden) };
    let act_mmq_null = unsafe { Tensor::<Q8_1>::from_raw(DevicePtr::NULL, 0) };
    let norm_mmq_t;
    let act_norm_mmq: &Tensor<Q8_1> = if input_pre_normed && n == 1 {
        let norm_view = unsafe { Tensor::<F16>::from_raw(state.pool.norm, n * hidden) };
        flambeau_model_ops::quantize_f16_to_q8_1(&norm_view, &mut norm_q8_1, n * hidden, &ops)?;
        &act_mmq_null
    } else if n > 1 {
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
    } else if n == 1 && weights.attn_q.supports_decode_to_f16() {
        // Decode F16-direct: skip the qmatmul→F32→cast pair, write Q
        // straight to the F16 KV-cache-shape buffer via mmvq's
        // saturating F16 cast. No AR on attn_q (TP col-shards heads).
        let mut q_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.q_f16, n * q_width) };
        weights
            .attn_q
            .qmatmul_decode_to_f16(&norm_q8_1, &mut q_f16, hidden, q_width, &ops)?;
    } else {
        let mut q_f32 = unsafe { Tensor::<F32>::from_raw(q_f32_buf, n * q_width) };
        weights
            .attn_q
            .qmatmul(&norm_q8_1, act_norm_mmq, &mut q_f32, n, hidden, q_width, &ops)?;
        let mut q_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.q_f16, n * q_width) };
        flambeau_model_ops::cast_f32_to_f16(&q_f32, &mut q_f16, n * q_width, &ops)?;
    }

    if n == 1 && weights.attn_k.supports_decode_to_f16() {
        let mut k_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.k_f16, n * kv_width) };
        weights
            .attn_k
            .qmatmul_decode_to_f16(&norm_q8_1, &mut k_f16, hidden, kv_width, &ops)?;
    } else {
        let mut k_f32 = unsafe { Tensor::<F32>::from_raw(q_f32_buf, n * kv_width) };
        weights
            .attn_k
            .qmatmul(&norm_q8_1, act_norm_mmq, &mut k_f32, n, hidden, kv_width, &ops)?;
        let mut k_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.k_f16, n * kv_width) };
        flambeau_model_ops::cast_f32_to_f16(&k_f32, &mut k_f16, n * kv_width, &ops)?;
    }

    if let Some(v_w) = weights.attn_v.as_ref() {
        if n == 1 && v_w.supports_decode_to_f16() {
            let mut v_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.v_f16, n * kv_width) };
            v_w.qmatmul_decode_to_f16(&norm_q8_1, &mut v_f16, hidden, kv_width, &ops)?;
        } else {
            let mut v_f32 = unsafe { Tensor::<F32>::from_raw(q_f32_buf, n * kv_width) };
            v_w.qmatmul(&norm_q8_1, act_norm_mmq, &mut v_f32, n, hidden, kv_width, &ops)?;
            let mut v_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.v_f16, n * kv_width) };
            flambeau_model_ops::cast_f32_to_f16(&v_f32, &mut v_f16, n * kv_width, &ops)?;
        }
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

    // Per-head V unit-weights rmsnorm (gemma4 trained behavior). The
    // legacy stack does this via a scratch.v_ones_f16 vector; v2 reuses
    // attn_out_f16 as the tmp and writes back to v_f16. Applied AFTER
    // V is materialised (both V-from-K and explicit V-proj paths).
    if let Some(v_unit_w) = weights.attn_v_unit_norm_w.as_ref() {
        let v_in = unsafe { Tensor::<F16>::from_raw(state.pool.v_f16, n * kv_width) };
        let mut v_tmp = unsafe { Tensor::<F16>::from_raw(state.pool.attn_out_f16, n * kv_width) };
        flambeau_model_ops::rmsnorm_f16(
            &v_in,
            v_unit_w,
            &mut v_tmp,
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
                    state.pool.v_f16,
                    v_tmp.ptr,
                    bytes,
                )
                .context("standard_attn: v_unit_norm DtoD copy back")?;
        }
    }

    // Positions HtoD hoisted ahead of the Q/K norm step so the fused
    // rmsnorm+RoPE kernel can read positions when q_norm / k_norm are set.
    let positions_i32: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
    let pos_bytes = n * 4;
    // SAFETY: position_i32 sized max_prefill_tokens * i32.
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
            .context("standard_attn: positions HtoD")?;
    }
    let pos_tensor = unsafe { Tensor::<I32>::from_raw(state.pool.position_i32, n) };
    let mut q_f16_rope = unsafe { Tensor::<F16>::from_raw(state.pool.q_f16, n * q_width) };
    let mut k_f16_rope = unsafe { Tensor::<F16>::from_raw(state.pool.k_f16, n * kv_width) };

    // gemma4 Q/K-norm + RoPE: fuse into one in-place launch each. Replaces
    // (rmsnorm_f16 → DtoD memcpy back → rope_neox_partial_f16) for archs
    // that set attn_q_norm / attn_k_norm. Saves 2 launches + 1 DtoD per
    // path per layer per token (Q and K independently).
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

    if prefill_shape {
        // Single-slot, contiguous positions → batched kv_append + attn_prefill.
        let slot_offset = primary_slot * slot_stride_bytes;
        let k_slot_ptr = kv.k.offset_bytes(slot_offset);
        let v_slot_ptr = kv.v.offset_bytes(slot_offset);
        let mut k_cache =
            unsafe { Tensor::<F16>::from_raw(k_slot_ptr, slot_stride_elems) };
        let mut v_cache =
            unsafe { Tensor::<F16>::from_raw(v_slot_ptr, slot_stride_elems) };
        let v_f16_view = unsafe { Tensor::<F16>::from_raw(state.pool.v_f16, n * kv_width) };
        if !is_kv_shared {
            flambeau_model_ops::kv_append_f16(
                &k_f16_rope,
                &v_f16_view,
                &mut k_cache,
                &mut v_cache,
                n,
                kv_width,
                start_position,
                max_seq_len,
                state.device,
                state.stream,
            )?;
        }
        let mut attn_out =
            unsafe { Tensor::<F16>::from_raw(state.pool.attn_out_f16, n * q_width) };
        if n == 1 {
            let n_tokens_kv = start_position + 1;
            let chunk_size = flambeau_model_ops::splitk_chunk_size(n_tokens_kv);
            let n_chunks = n_tokens_kv.div_ceil(chunk_size);
            let use_splitk = n_tokens_kv > 256
                && n_chunks > 1
                && n_chunks <= crate::core::scratch::MAX_SPLITK_CHUNKS
                && state.pool.splitk_partials_m.as_usize() != 0;
            if use_splitk {
                let partials_m_n = weights.n_heads * n_chunks;
                let partials_o_n = partials_m_n * weights.head_dim;
                let mut partials_m = unsafe {
                    Tensor::<F32>::from_raw(state.pool.splitk_partials_m, partials_m_n)
                };
                let mut partials_s = unsafe {
                    Tensor::<F32>::from_raw(state.pool.splitk_partials_s, partials_m_n)
                };
                let mut partials_o = unsafe {
                    Tensor::<F32>::from_raw(state.pool.splitk_partials_o, partials_o_n)
                };
                flambeau_model_ops::attn_decode_f16_splitk(
                    &q_f16_rope,
                    &k_cache,
                    &v_cache,
                    &mut attn_out,
                    &mut partials_m,
                    &mut partials_s,
                    &mut partials_o,
                    weights.n_heads,
                    weights.n_kv_heads,
                    weights.head_dim,
                    n_tokens_kv,
                    chunk_size,
                    scale,
                    weights.window_size,
                    &ops,
                )?;
            } else {
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
            }
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
    } else {
        // Multi-slot or non-contiguous positions → one-shot batched
        // KV-append + batched-decode attention. Caller's pool sized
        // for `max_slots > 1` provided the host→device upload arrays.
        if state.pool.attn_slot_k_dst_ptrs.as_usize() == 0 {
            bail!(
                "standard_attn: non-prefill shape requires max_slots > 1 in ScratchConfig \
                 (got max_slots={})",
                state.pool.config.max_slots
            );
        }
        let mut host_k_ptrs: Vec<u64> = Vec::with_capacity(n);
        let mut host_v_ptrs: Vec<u64> = Vec::with_capacity(n);
        let mut host_write_pos: Vec<i32> = Vec::with_capacity(n);
        let mut host_n_kv: Vec<i32> = Vec::with_capacity(n);
        for i in 0..n {
            let slot = slot_ids[i];
            let pos = positions[i];
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
                    n * 8,
                )
                .context("standard_attn: attn_slot_k_dst_ptrs HtoD")?;
            state
                .device
                .memcpy_async(
                    state.stream,
                    CopyDirection::HostToDevice,
                    state.pool.attn_slot_v_dst_ptrs,
                    DevicePtr(host_v_ptrs.as_ptr() as usize),
                    n * 8,
                )
                .context("standard_attn: attn_slot_v_dst_ptrs HtoD")?;
            state
                .device
                .memcpy_async(
                    state.stream,
                    CopyDirection::HostToDevice,
                    state.pool.attn_slot_write_pos,
                    DevicePtr(host_write_pos.as_ptr() as usize),
                    n * 4,
                )
                .context("standard_attn: attn_slot_write_pos HtoD")?;
            state
                .device
                .memcpy_async(
                    state.stream,
                    CopyDirection::HostToDevice,
                    state.pool.attn_slot_n_kv,
                    DevicePtr(host_n_kv.as_ptr() as usize),
                    n * 4,
                )
                .context("standard_attn: attn_slot_n_kv HtoD")?;
        }
        flambeau_core::Stream::synchronize(state.stream)?;
        let k_src_full = unsafe { Tensor::<F16>::from_raw(state.pool.k_f16, n * kv_width) };
        let v_src_full = unsafe { Tensor::<F16>::from_raw(state.pool.v_f16, n * kv_width) };
        if !is_kv_shared {
            flambeau_model_ops::kv_append_f16_batched_slots(
                &k_src_full,
                &v_src_full,
                state.pool.attn_slot_k_dst_ptrs,
                state.pool.attn_slot_v_dst_ptrs,
                state.pool.attn_slot_write_pos,
                n,
                kv_width,
                &ops,
            )?;
        }
        // The two `_src_full` slot pointer arrays must point at the
        // share-source's KV cache slabs when reading shared KV. The
        // host-side arrays above were filled from `kv` which already
        // routed via `kv_local_idx`, so this is correct.
        let _ = (&k_src_full, &v_src_full);
        let q_batched =
            unsafe { Tensor::<F16>::from_raw(state.pool.q_f16, n * q_width) };
        let mut attn_out_batched =
            unsafe { Tensor::<F16>::from_raw(state.pool.attn_out_f16, n * q_width) };
        flambeau_model_ops::attn_decode_f16_batched(
            &q_batched,
            state.pool.attn_slot_k_dst_ptrs,
            state.pool.attn_slot_v_dst_ptrs,
            &mut attn_out_batched,
            state.pool.attn_slot_n_kv,
            weights.n_heads,
            weights.n_kv_heads,
            weights.head_dim,
            n,
            scale,
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

    // Decode F16-direct fast path: at n=1 with AR-fold semantics
    // (post_attn_norm=None + caller wants F16 partial), the projection
    // can write straight to `pool.delta` via mmvq's saturating F16-cast
    // — skips one `cast_f32_to_f16` launch per layer per token.
    let f16_fast = n == 1
        && weights.post_attn_norm.is_none()
        && weights.attn_output.supports_decode_to_f16()
        && (hooks.supports_ar_residual_rmsnorm_f16()
            || hooks.supports_ar_residual_f16());
    let mut proj_f32 = unsafe { Tensor::<F32>::from_raw(state.pool.attn_proj_f32, n * hidden) };
    if !f16_fast {
        weights.attn_output.qmatmul(
            &attn_out_q8_1,
            act_attn_out_mmq,
            &mut proj_f32,
            n,
            q_width,
            hidden,
            &ops,
        )?;
    }
    // Fast path: BAR1 TP=2 with no post-attn-norm folds the
    // F32→F16 cast + AR + residual-add into 1 launch (residual_tp2)
    // instead of 4 (qmatmul + cast + ar_sum + add). Returns None so the
    // model skips its own residual_add.
    if n == 1
        && weights.post_attn_norm.is_none()
        && next_norm.is_some()
        && hooks.supports_ar_residual_rmsnorm_f16()
    {
        let next_w = next_norm.unwrap();
        let mut partial_f16 =
            unsafe { Tensor::<F16>::from_raw(state.pool.delta, n * hidden) };
        if f16_fast {
            weights.attn_output.qmatmul_decode_to_f16(
                &attn_out_q8_1,
                &mut partial_f16,
                q_width,
                hidden,
                &ops,
            )?;
        } else {
            flambeau_model_ops::cast_f32_to_f16(&proj_f32, &mut partial_f16, n * hidden, &ops)?;
        }
        hooks.ar_residual_rmsnorm_f16(
            input.ptr,
            partial_f16.ptr,
            next_w.ptr,
            state.pool.norm,
            n * hidden,
            weights.rms_eps,
            state.device,
            state.stream,
        )?;
        state.pool.input_pre_normed = true;
        return Ok(None);
    }
    if hooks.supports_ar_residual_f16() && weights.post_attn_norm.is_none() {
        let mut partial_f16 =
            unsafe { Tensor::<F16>::from_raw(state.pool.delta, n * hidden) };
        if f16_fast {
            weights.attn_output.qmatmul_decode_to_f16(
                &attn_out_q8_1,
                &mut partial_f16,
                q_width,
                hidden,
                &ops,
            )?;
        } else {
            flambeau_model_ops::cast_f32_to_f16(&proj_f32, &mut partial_f16, n * hidden, &ops)?;
        }
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
    let delta = unsafe { Tensor::<F16>::from_raw(state.pool.delta, n * hidden) };
    if let Some(post_norm) = weights.post_attn_norm.as_ref() {
        // Fused F32→F16 + rmsnorm: skip the cast_f32_to_f16 launch and
        // do the post-norm in one pass over the row.
        let mut delta_out = unsafe { Tensor::<F16>::from_raw(state.pool.delta, n * hidden) };
        flambeau_model_ops::rmsnorm_f32_to_f16(
            &proj_f32,
            post_norm,
            &mut delta_out,
            n,
            hidden,
            weights.rms_eps,
            &ops,
        )?;
    } else {
        let mut delta_mut = unsafe { Tensor::<F16>::from_raw(state.pool.delta, n * hidden) };
        flambeau_model_ops::cast_f32_to_f16(&proj_f32, &mut delta_mut, n * hidden, &ops)?;
    }
    Ok(Some(delta))
}
