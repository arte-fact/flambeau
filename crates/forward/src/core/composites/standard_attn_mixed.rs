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

use crate::core::{CoreState, MixedBatch, TopologyHooks};
use crate::ctx::AttnWeights;

pub fn standard_attn_mixed_local<H: TopologyHooks>(
    state: &mut CoreState<'_>,
    hooks: &mut H,
    input: &Tensor<F16>,
    weights: &AttnWeights,
    layer_idx: usize,
    batch: MixedBatch<'_>,
    next_norm: Option<&Tensor<F16>>,
) -> Result<Option<Tensor<F16>>> {
    let MixedBatch { positions, slot_ids, prefill_rows } = batch;
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
    for (i, (&slot, &pos)) in slot_ids.iter().zip(positions.iter()).enumerate().take(k) {
        if slot != slot_p {
            bail!("standard_attn_mixed: prefill row {i} slot {slot} != slot_p {slot_p}");
        }
        if pos != pos0 + i {
            bail!(
                "standard_attn_mixed: prefill row {i} pos {pos} not contiguous from pos0={pos0}"
            );
        }
    }
    for (i, &slot) in slot_ids.iter().enumerate().take(n).skip(k) {
        if slot == slot_p {
            bail!(
                "standard_attn_mixed: decode row {i} slot {slot} collides with prefill slot_p {slot_p}"
            );
        }
    }

    // Remaining bails: paged KV (Phase K-paged) and gemma 4n's
    // shared-KV layers. SWA / V-unit-norm / post_attn_norm / V-from-K
    // ship in this slice for gemma4 dense and 26B-A4B MoE.
    if state.pool.paged_kv_caches.is_some() {
        bail!("standard_attn_mixed: paged KV not yet supported (Phase K-paged)");
    }
    if weights.kv_share_src.is_some() {
        bail!("standard_attn_mixed: shared-KV layers (gemma 4n) not yet supported");
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
            flambeau_ops::MatmulShape {
                m: n,
                k: hidden,
                n: fused_n,
            },
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
            flambeau_ops::MatmulShape {
                m: n,
                k: hidden,
                n: q_width,
            },
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
            flambeau_ops::MatmulShape {
                m: n,
                k: hidden,
                n: kv_width,
            },
            &ops,
        )?;
        let mut k_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.k_f16, n * kv_width) };
        flambeau_model_ops::cast_f32_to_f16(&k_f32, &mut k_f16, n * kv_width, &ops)?;
    }

    // ---- V projection. When `attn_v` is None (gemma4 V-from-K),
    //      V = K so we DtoD the K buffer into v_f16 at n=K+N. ----
    if let Some(v_w) = weights.attn_v.as_ref() {
        let mut v_f32 = unsafe { Tensor::<F32>::from_raw(q_f32_buf, n * kv_width) };
        v_w.qmatmul(
            &norm_q8_1,
            act_norm_mmq,
            &mut v_f32,
            flambeau_ops::MatmulShape {
                m: n,
                k: hidden,
                n: kv_width,
            },
            &ops,
        )?;
        let mut v_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.v_f16, n * kv_width) };
        flambeau_model_ops::cast_f32_to_f16(&v_f32, &mut v_f16, n * kv_width, &ops)?;
    } else {
        let bytes = n * kv_width * 2;
        // SAFETY: k_f16 and v_f16 pool slots are both sized for
        // max_prefill_tokens * kv_width F16 (validated at pool boot).
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
                .context("standard_attn_mixed: V-from-K DtoD memcpy")?;
        }
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
            flambeau_ops::RopeFusedBuffers {
                x: state.pool.q_f16,
                norm_w: q_norm_w.ptr,
                positions: state.pool.position_i32,
            },
            flambeau_ops::RopePartialShape {
                n_tokens: n,
                n_heads: weights.n_heads,
                head_dim: weights.head_dim,
                rotated_dims: weights.rotated_dims,
            },
            weights.rope_theta,
            weights.rms_eps,
        )?;
    } else {
        let mut q_f16_rope = unsafe { Tensor::<F16>::from_raw(state.pool.q_f16, n * q_width) };
        flambeau_model_ops::rope_neox_partial_f16(
            &mut q_f16_rope,
            &pos_tensor,
            weights.rope_theta,
            flambeau_ops::RopePartialShape {
                n_tokens: n,
                n_heads: weights.n_heads,
                head_dim: weights.head_dim,
                rotated_dims: weights.rotated_dims,
            },
            &ops,
        )?;
    }
    if let Some(k_norm_w) = weights.attn_k_norm.as_ref() {
        ops.rmsnorm_rope_neox_partial_f16(
            flambeau_ops::RopeFusedBuffers {
                x: state.pool.k_f16,
                norm_w: k_norm_w.ptr,
                positions: state.pool.position_i32,
            },
            flambeau_ops::RopePartialShape {
                n_tokens: n,
                n_heads: weights.n_kv_heads,
                head_dim: weights.head_dim,
                rotated_dims: weights.rotated_dims,
            },
            weights.rope_theta,
            weights.rms_eps,
        )?;
    } else {
        let mut k_f16_rope = unsafe { Tensor::<F16>::from_raw(state.pool.k_f16, n * kv_width) };
        flambeau_model_ops::rope_neox_partial_f16(
            &mut k_f16_rope,
            &pos_tensor,
            weights.rope_theta,
            flambeau_ops::RopePartialShape {
                n_tokens: n,
                n_heads: weights.n_kv_heads,
                head_dim: weights.head_dim,
                rotated_dims: weights.rotated_dims,
            },
            &ops,
        )?;
    }
    let _ = weights.rope_variant;

    // ---- gemma4 V unit-norm: per-head RMSNorm on V at n = K + N
    //      before kv_append. Unit weights (no learnable gamma) —
    //      the kernel matches the rmsnorm inside the legacy fused
    //      `kv_append_v_unit_norm_f16`. After this, both kv_append
    //      paths (range-write for K, batched-slots for N) write the
    //      normed V into their respective cache slots.
    if weights.attn_v_unit_norm_w.is_some() {
        let mut v_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.v_f16, n * kv_width) };
        flambeau_model_ops::v_unit_norm_per_head_f16(
            &mut v_f16,
            n,
            weights.n_kv_heads,
            weights.head_dim,
            weights.rms_eps,
            &ops,
        )?;
    }

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
            flambeau_model_ops::KvAppendSpec {
                n_tokens: k,
                kv_width,
                write_pos: pos0,
                max_seq_len,
            },
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
            flambeau_ops::AttnPrefillShape {
                n_q_tokens: k,
                n_heads_q: weights.n_heads,
                n_heads_kv: weights.n_kv_heads,
                head_dim: weights.head_dim,
                n_k_tokens,
                q_offset: pos0,
            },
            flambeau_ops::AttnKnobs { scale, window_size: weights.window_size, ring_depth: 0 },
            &ops,
        )?;
    }

    // ---- N decode rows ----
    let q_dec_ptr = q_pref_ptr.offset_bytes(k * q_width * 2);
    let k_dec_ptr = k_pref_ptr.offset_bytes(k * kv_width * 2);
    let v_dec_ptr = v_pref_ptr.offset_bytes(k * kv_width * 2);
    let out_dec_ptr = out_pref_ptr.offset_bytes(k * q_width * 2);
    {
        // Per-slot SWA window-offset mirror of standard_attn.rs.
        // kv_append writes at absolute slot positions; attention
        // reads from each slot's window start. When any slot has
        // pos+1 > window we re-upload offset ptrs before attention.
        let window_size = weights.window_size;
        let window = window_size as usize;
        let bytes_per_row = kv.bytes_per_row;
        let mut host_k_base_ptrs: Vec<u64> = Vec::with_capacity(n_dec);
        let mut host_v_base_ptrs: Vec<u64> = Vec::with_capacity(n_dec);
        let mut host_k_read_ptrs: Vec<u64> = Vec::with_capacity(n_dec);
        let mut host_v_read_ptrs: Vec<u64> = Vec::with_capacity(n_dec);
        let mut host_write_pos: Vec<i32> = Vec::with_capacity(n_dec);
        let mut host_n_kv: Vec<i32> = Vec::with_capacity(n_dec);
        let mut any_offset = false;
        for i in 0..n_dec {
            let slot = slot_ids[k + i];
            let pos = positions[k + i];
            let slot_offset = slot * slot_stride_bytes;
            let slot_k_base = kv.k.offset_bytes(slot_offset);
            let slot_v_base = kv.v.offset_bytes(slot_offset);
            host_k_base_ptrs.push(slot_k_base.as_usize() as u64);
            host_v_base_ptrs.push(slot_v_base.as_usize() as u64);
            let n_kv_full = pos + 1;
            let (k_read_ptr, v_read_ptr, n_kv_eff) =
                if window_size > 0 && window < n_kv_full {
                    any_offset = true;
                    let off_tokens = n_kv_full - window;
                    let off_bytes = off_tokens * bytes_per_row;
                    (
                        slot_k_base.offset_bytes(off_bytes),
                        slot_v_base.offset_bytes(off_bytes),
                        window,
                    )
                } else {
                    (slot_k_base, slot_v_base, n_kv_full)
                };
            host_k_read_ptrs.push(k_read_ptr.as_usize() as u64);
            host_v_read_ptrs.push(v_read_ptr.as_usize() as u64);
            host_write_pos.push(pos as i32);
            host_n_kv.push(n_kv_eff as i32);
        }
        let kernel_window: i32 = if any_offset { 0 } else { window_size };
        unsafe {
            state
                .device
                .memcpy_async(
                    state.stream,
                    CopyDirection::HostToDevice,
                    state.pool.attn_slot_k_dst_ptrs,
                    DevicePtr(host_k_base_ptrs.as_ptr() as usize),
                    n_dec * 8,
                )
                .context("standard_attn_mixed: attn_slot_k_dst_ptrs HtoD")?;
            state
                .device
                .memcpy_async(
                    state.stream,
                    CopyDirection::HostToDevice,
                    state.pool.attn_slot_v_dst_ptrs,
                    DevicePtr(host_v_base_ptrs.as_ptr() as usize),
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
        flambeau_model_ops::kv_append_f16_batched_slots(
            flambeau_ops::KvAppendBatchedSlotsBuffers {
                k_src: k_dec_ptr,
                v_src: v_dec_ptr,
                slot_k_dst_ptrs: state.pool.attn_slot_k_dst_ptrs,
                slot_v_dst_ptrs: state.pool.attn_slot_v_dst_ptrs,
                slot_write_pos: state.pool.attn_slot_write_pos,
            },
            flambeau_ops::KvAppendBatchedSlotsShape {
                n_slots: n_dec,
                kv_width,
            },
            &ops,
        )?;
        if any_offset {
            unsafe {
                state
                    .device
                    .memcpy_async(
                        state.stream,
                        CopyDirection::HostToDevice,
                        state.pool.attn_slot_k_dst_ptrs,
                        DevicePtr(host_k_read_ptrs.as_ptr() as usize),
                        n_dec * 8,
                    )
                    .context("standard_attn_mixed: attn_slot_k_dst_ptrs HtoD (SWA)")?;
                state
                    .device
                    .memcpy_async(
                        state.stream,
                        CopyDirection::HostToDevice,
                        state.pool.attn_slot_v_dst_ptrs,
                        DevicePtr(host_v_read_ptrs.as_ptr() as usize),
                        n_dec * 8,
                    )
                    .context("standard_attn_mixed: attn_slot_v_dst_ptrs HtoD (SWA)")?;
            }
            flambeau_core::Stream::synchronize(state.stream)?;
        }
        flambeau_model_ops::attn_decode_f16_batched(
            flambeau_ops::AttnBatchedBuffers {
                q_batched: q_dec_ptr,
                k_cache_ptrs: state.pool.attn_slot_k_dst_ptrs,
                v_cache_ptrs: state.pool.attn_slot_v_dst_ptrs,
                out_batched: out_dec_ptr,
                n_tokens_kv_ptrs: state.pool.attn_slot_n_kv,
            },
            flambeau_ops::AttnDecodeBatchedShape {
                n_heads_q: weights.n_heads,
                n_heads_kv: weights.n_kv_heads,
                head_dim: weights.head_dim,
                n_slots: n_dec,
            },
            flambeau_ops::AttnKnobs {
                scale,
                window_size: kernel_window,
                ring_depth: 0,
            },
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
        flambeau_ops::MatmulShape {
            m: n,
            k: q_width,
            n: hidden,
        },
        &ops,
    )?;

    // ---- AR + residual + (optional) post_attn_norm fold ----
    // BAR1 ar_residual_f16 fast path is only safe when there's no
    // post_attn_norm AND no next-norm fold — gemma4 needs the fused
    // rmsnorm_f32_to_f16_add_residual after AR, which the fast path
    // would skip.
    if hooks.supports_ar_residual_f16()
        && next_norm.is_none()
        && weights.post_attn_norm.is_none()
    {
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
    if let Some(post_norm) = weights.post_attn_norm.as_ref() {
        // Fused F32→F16 rmsnorm + add to residual. Advances the
        // pool's residual slot directly and sets
        // `fused_residual_already_done` so the model's `residual_add`
        // skips re-doing the add. Mirrors `standard_attn`'s post-AR
        // gemma4 path at n = K + N.
        let resid_in_ptr = input.ptr;
        let new_resid_ptr = state.pool.next_residual_slot();
        ops.rmsnorm_f32_to_f16_add_residual(
            flambeau_ops::NormResidualBuffers {
                input: proj_f32.ptr,
                weight: post_norm.ptr,
                resid_in: resid_in_ptr,
                resid_out: new_resid_ptr,
            },
            flambeau_ops::NormShape { m: n, k: hidden },
            weights.rms_eps,
        )?;
        state.pool.fused_residual_already_done = true;
        let new_resid = unsafe { Tensor::<F16>::from_raw(new_resid_ptr, n * hidden) };
        let _ = next_norm;
        return Ok(Some(new_resid));
    }
    let mut delta_mut = unsafe { Tensor::<F16>::from_raw(state.pool.delta, n * hidden) };
    flambeau_model_ops::cast_f32_to_f16(&proj_f32, &mut delta_mut, n * hidden, &ops)?;
    let delta = unsafe { Tensor::<F16>::from_raw(state.pool.delta, n * hidden) };
    let _ = next_norm;
    Ok(Some(delta))
}
