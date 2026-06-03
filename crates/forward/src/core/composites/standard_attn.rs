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
    let local_idx = layer_idx
        .checked_sub(state.layer_idx_offset)
        .ok_or_else(|| {
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
        let mut norm_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.norm, n * hidden) };
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

    let fuse_kv_q4_0 = n == 1
        && weights.attn_k.dtype == flambeau_core::op::QDtype::Q4_0
        && weights
            .attn_v
            .as_ref()
            .map(|v| v.dtype == flambeau_core::op::QDtype::Q4_0)
            .unwrap_or(false);
    if fuse_kv_q4_0 {
        let k_w = unsafe {
            Tensor::<flambeau_model_ops::Q4_0>::from_raw(
                weights.attn_k.ptr,
                weights.attn_k.n_elems,
            )
        };
        let v_qw = weights.attn_v.as_ref().unwrap();
        let v_w = unsafe {
            Tensor::<flambeau_model_ops::Q4_0>::from_raw(v_qw.ptr, v_qw.n_elems)
        };
        let mut k_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.k_f16, n * kv_width) };
        let mut v_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.v_f16, n * kv_width) };
        flambeau_model_ops::mmvq_q4_0_kv_decode_f16(
            &k_w,
            &v_w,
            &norm_q8_1,
            &mut k_f16,
            &mut v_f16,
            flambeau_ops::MmvqShape { n_rows: kv_width, k: hidden },
            &ops,
        )?;
    } else if n == 1 && weights.attn_k.supports_decode_to_f16() {
        let mut k_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.k_f16, n * kv_width) };
        weights
            .attn_k
            .qmatmul_decode_to_f16(&norm_q8_1, &mut k_f16, hidden, kv_width, &ops)?;
    } else {
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

    if fuse_kv_q4_0 {
        // V was produced by the fused K+V kernel above.
    } else if let Some(v_w) = weights.attn_v.as_ref() {
        if n == 1 && v_w.supports_decode_to_f16() {
            let mut v_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.v_f16, n * kv_width) };
            v_w.qmatmul_decode_to_f16(&norm_q8_1, &mut v_f16, hidden, kv_width, &ops)?;
        } else {
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

    // V unit-RMSNorm (gemma4 trained behavior) is fused into the
    // KV-cache append below for the single-slot path when
    // `attn_v_unit_norm_w` is set. The fused kernel writes the normed
    // V directly to the cache slot — eliminates this standalone
    // rmsnorm + DtoD memcpy pair from the per-layer per-token cost.
    let _ = weights.attn_v_unit_norm_w; // consumed in the prefill_shape branch below.

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

    let kv = state.pool.kv_caches[kv_local_idx];
    let slot_stride_elems = max_seq_len * kv_width;
    let slot_stride_bytes = slot_stride_elems * 2;
    let scale = weights
        .softmax_scale
        .unwrap_or_else(|| (weights.head_dim as f32).sqrt().recip());

    if let Some(paged_caches) = state.pool.paged_kv_caches.as_ref().filter(|_| prefill_shape) {
        // Paged prefill / single-slot path. Pre-acquire all pages
        // spanned by `[start_position, start_position + n)`, patch
        // the slot's row of the block table, write L tokens via the
        // paged-prefill kv_append, then run attention as a per-token
        // loop over `attention_decode_f16_paged`. The per-token loop
        // is the E3d "correctness over speed" stub — a future slice
        // replaces it with a real paged-prefill attention kernel.
        if weights.attn_v_unit_norm_w.is_some() {
            bail!(
                "standard_attn paged: V unit-norm fusion (gemma4) not yet supported on paged \
                 path — disable paged_kv for this arch until the paged V-norm kernel ships"
            );
        }
        if weights.window_size > 0 {
            bail!(
                "standard_attn paged: sliding-window attention not yet supported on paged path \
                 (window_size={})",
                weights.window_size
            );
        }
        let paged_cache = paged_caches[kv_local_idx];
        let page_size = paged_cache.page_size;
        let mpps = paged_cache.max_pages_per_slot;
        let slot = primary_slot;
        let total_kv_after = start_position + n;
        let pages_needed = total_kv_after.div_ceil(page_size);
        let new_pages = state.pool.page_pools[kv_local_idx]
            .ensure_pages_up_to(slot, pages_needed)
            .map_err(|acquired| {
                anyhow::anyhow!(
                    "standard_attn paged: PagePool for layer {layer_idx} exhausted at slot {slot} \
                     trying to grow to {pages_needed} pages (acquired {acquired} before exhaustion; \
                     free={})",
                    state.pool.page_pools[kv_local_idx].n_free()
                )
            })?;
        let slot_table_byte_offset = slot * mpps * std::mem::size_of::<u32>();
        for (page_idx_in_slot, page) in &new_pages {
            let host_page = [*page];
            let dst = paged_cache
                .block_tables
                .offset_bytes(slot_table_byte_offset + page_idx_in_slot * std::mem::size_of::<u32>());
            // SAFETY: block_tables owns max_slots*mpps*4 bytes;
            // slot*mpps*4 + page_idx*4 < max_slots*mpps*4 (bounded by
            // ensure_pages_up_to's `pages_needed <= mpps` check).
            unsafe {
                state.device.memcpy_async(
                    state.stream,
                    CopyDirection::HostToDevice,
                    dst,
                    DevicePtr(host_page.as_ptr() as usize),
                    std::mem::size_of::<u32>(),
                )?;
            }
        }
        flambeau_core::Stream::synchronize(state.stream)?;

        let slot_block_table_ptr = paged_cache
            .block_tables
            .offset_bytes(slot_table_byte_offset);
        let v_f16_view = unsafe { Tensor::<F16>::from_raw(state.pool.v_f16, n * kv_width) };
        if !is_kv_shared {
            flambeau_model_ops::kv_append_f16_paged_prefill(
                flambeau_ops::KvAppendPagedPrefillBuffers {
                    k_src: k_f16_rope.ptr,
                    v_src: v_f16_view.ptr,
                    k_pool: paged_cache.k_pool,
                    v_pool: paged_cache.v_pool,
                    block_table: slot_block_table_ptr,
                },
                flambeau_ops::KvAppendPagedPrefillShape {
                    n_tokens: n,
                    kv_width,
                    start_pos: start_position,
                    page_size,
                },
                &ops,
            )?;
        }

        // Single-launch paged prefill attention. Same flash-attn-v2
        // body as the contiguous `attn_prefill_f16`; per-`t` K/V row
        // is resolved through the slot's row of the block table. The
        // earlier per-token `attn_decode_f16_paged` loop was the E3d
        // stub; this slice replaces it with the real multi-row paged
        // kernel — one launch instead of L.
        flambeau_model_ops::attn_prefill_f16_paged(
            flambeau_ops::AttnPagedPrefillBuffers {
                q: state.pool.q_f16,
                k_pool: paged_cache.k_pool,
                v_pool: paged_cache.v_pool,
                block_table: slot_block_table_ptr,
                out: state.pool.attn_out_f16,
            },
            flambeau_ops::AttnPrefillPagedShape {
                n_q_tokens: n,
                n_heads_q: weights.n_heads,
                n_heads_kv: weights.n_kv_heads,
                head_dim: weights.head_dim,
                n_k_tokens: start_position + n,
                q_offset: start_position,
                page_size,
            },
            flambeau_ops::AttnKnobs {
                scale,
                // window_size guard above already bailed on SWA when paged on.
                window_size: 0,
            },
            &ops,
        )?;
        // Suppress unused-mpps lint when the paged-prefill kernel
        // signature doesn't need max_pages_per_slot.
        let _ = mpps;
    } else if prefill_shape {
        // Single-slot, contiguous positions → batched kv_append + attn_prefill.
        // Q8Contig path (S7c) replaces F16 kv_append + attn_decode/prefill with
        // the Q8 model-ops siblings. The Q8 dispatch is gated upstream by
        // `scratch_config_for`'s Q8 viability check (head_dim ∈ {64,128,256,512})
        // — at runtime we just trust the cache layout.
        let is_q8 = kv.layout == crate::core::KvLayout::Q8Contig;
        let q8_slot_stride_bytes = max_seq_len * kv.bytes_per_row;
        let slot_offset = if is_q8 {
            primary_slot * q8_slot_stride_bytes
        } else {
            primary_slot * slot_stride_bytes
        };
        let k_slot_ptr = kv.k.offset_bytes(slot_offset);
        let v_slot_ptr = kv.v.offset_bytes(slot_offset);
        let mut k_cache = unsafe { Tensor::<F16>::from_raw(k_slot_ptr, slot_stride_elems) };
        let mut v_cache = unsafe { Tensor::<F16>::from_raw(v_slot_ptr, slot_stride_elems) };
        let v_f16_view = unsafe { Tensor::<F16>::from_raw(state.pool.v_f16, n * kv_width) };
        if !is_kv_shared {
            if weights.attn_v_unit_norm_w.is_some() && !is_q8 {
                // gemma4 F16 path: fuse V unit-RMSNorm into the cache
                // write. K copy + V normalize-then-copy in one launch.
                ops.kv_append_v_unit_norm_f16(
                    flambeau_ops::KvAppendBuffers {
                        k_src: state.pool.k_f16,
                        v_src: state.pool.v_f16,
                        k_dst: k_cache.ptr,
                        v_dst: v_cache.ptr,
                    },
                    flambeau_ops::KvAppendVUnitShape {
                        n_tokens: n,
                        n_kv_heads: weights.n_kv_heads,
                        head_dim: weights.head_dim,
                    },
                    start_position,
                    weights.rms_eps,
                )?;
            } else if weights.attn_v_unit_norm_w.is_some() && is_q8 {
                // gemma4 Q8 path: V-unit-norm in place on the F16 scratch,
                // then F16→Q8 quantize-and-write. The fused F16 kernel
                // can't target a Q8 slab (different byte stride per row),
                // so this slice splits the two ops.
                let mut v_for_norm = unsafe {
                    Tensor::<F16>::from_raw(state.pool.v_f16, n * kv_width)
                };
                flambeau_model_ops::v_unit_norm_per_head_f16(
                    &mut v_for_norm,
                    n,
                    weights.n_kv_heads,
                    weights.head_dim,
                    weights.rms_eps,
                    &ops,
                )?;
                let mut k_cache_q8 = unsafe {
                    Tensor::<flambeau_model_ops::Q8_0>::from_raw(
                        k_slot_ptr,
                        max_seq_len * kv_width,
                    )
                };
                let mut v_cache_q8 = unsafe {
                    Tensor::<flambeau_model_ops::Q8_0>::from_raw(
                        v_slot_ptr,
                        max_seq_len * kv_width,
                    )
                };
                flambeau_model_ops::kv_append_f16_to_q8(
                    &k_f16_rope,
                    &v_f16_view,
                    &mut k_cache_q8,
                    &mut v_cache_q8,
                    flambeau_model_ops::KvAppendSpec {
                        n_tokens: n,
                        kv_width,
                        write_pos: start_position,
                        max_seq_len,
                    },
                    &ops,
                )?;
            } else if is_q8 {
                let mut k_cache_q8 = unsafe {
                    Tensor::<flambeau_model_ops::Q8_0>::from_raw(
                        k_slot_ptr,
                        max_seq_len * kv_width,
                    )
                };
                let mut v_cache_q8 = unsafe {
                    Tensor::<flambeau_model_ops::Q8_0>::from_raw(
                        v_slot_ptr,
                        max_seq_len * kv_width,
                    )
                };
                flambeau_model_ops::kv_append_f16_to_q8(
                    &k_f16_rope,
                    &v_f16_view,
                    &mut k_cache_q8,
                    &mut v_cache_q8,
                    flambeau_model_ops::KvAppendSpec {
                        n_tokens: n,
                        kv_width,
                        write_pos: start_position,
                        max_seq_len,
                    },
                    &ops,
                )?;
            } else {
                flambeau_model_ops::kv_append_f16(
                    &k_f16_rope,
                    &v_f16_view,
                    &mut k_cache,
                    &mut v_cache,
                    flambeau_model_ops::KvAppendSpec {
                        n_tokens: n,
                        kv_width,
                        write_pos: start_position,
                        max_seq_len,
                    },
                    state.device,
                    state.stream,
                )?;
            }
        }
        let mut attn_out = unsafe { Tensor::<F16>::from_raw(state.pool.attn_out_f16, n * q_width) };
        if n == 1 {
            let n_tokens_kv = start_position + 1;
            if is_q8 {
                // SWA at decode: when window < n_tokens_kv, slide the
                // cache pointer to the window's start and pass the
                // window length as n_tokens_kv. The kernel's `qpos =
                // n_tokens_kv - 1` then makes t_start = 0 (no SWA
                // masking inside the kernel) and there's no waste over
                // out-of-window chunks. window_size = 0 keeps the full
                // causal range.
                let (eff_n_tokens_kv, k_ptr_eff, v_ptr_eff) = if weights.window_size > 0
                    && (weights.window_size as usize) < n_tokens_kv
                {
                    let w = weights.window_size as usize;
                    let offset_tokens = n_tokens_kv - w;
                    let offset_bytes = offset_tokens * kv.bytes_per_row;
                    (w, k_slot_ptr.offset_bytes(offset_bytes), v_slot_ptr.offset_bytes(offset_bytes))
                } else {
                    (n_tokens_kv, k_slot_ptr, v_slot_ptr)
                };
                let k_cache_q8 = unsafe {
                    Tensor::<flambeau_model_ops::Q8_0>::from_raw(
                        k_ptr_eff,
                        max_seq_len * kv_width,
                    )
                };
                let v_cache_q8 = unsafe {
                    Tensor::<flambeau_model_ops::Q8_0>::from_raw(
                        v_ptr_eff,
                        max_seq_len * kv_width,
                    )
                };
                // Effective window length is the kernel's view of the
                // cache; the SWA mask becomes a no-op inside the kernel.
                let kernel_window = 0i32;
                let chunk_size = flambeau_model_ops::splitk_chunk_size(eff_n_tokens_kv);
                let n_chunks = eff_n_tokens_kv.div_ceil(chunk_size);
                let use_splitk = eff_n_tokens_kv > 256
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
                    flambeau_model_ops::attn_decode_q8_kv_splitk(
                        &q_f16_rope,
                        &k_cache_q8,
                        &v_cache_q8,
                        &mut attn_out,
                        &mut partials_m,
                        &mut partials_s,
                        &mut partials_o,
                        flambeau_ops::AttnSplitkShape {
                            n_heads_q: weights.n_heads,
                            n_heads_kv: weights.n_kv_heads,
                            head_dim: weights.head_dim,
                            n_tokens_kv: eff_n_tokens_kv,
                            chunk_size,
                        },
                        flambeau_ops::AttnKnobs { scale, window_size: kernel_window },
                        &ops,
                    )?;
                } else {
                    flambeau_model_ops::attn_decode_q8_kv(
                        &q_f16_rope,
                        &k_cache_q8,
                        &v_cache_q8,
                        &mut attn_out,
                        flambeau_ops::AttnDecodeShape {
                            n_heads_q: weights.n_heads,
                            n_heads_kv: weights.n_kv_heads,
                            head_dim: weights.head_dim,
                            n_tokens_kv: eff_n_tokens_kv,
                        },
                        flambeau_ops::AttnKnobs { scale, window_size: kernel_window },
                        &ops,
                    )?;
                }
                let _ = (k_cache, v_cache);
            } else {
            // SWA at decode (mirror of the Q8 lever above): when
            // window_size < n_tokens_kv, slide the F16 cache pointer
            // to the window start and pass n_tokens_kv = window. The
            // kernel's internal SWA mask becomes a no-op. Eliminates
            // splitk launches for out-of-window chunks and lets the
            // chunk_size heuristic pick a tighter chunking for the
            // active window.
            let (eff_n_tokens_kv, k_cache_eff_t, v_cache_eff_t) =
                if weights.window_size > 0 && (weights.window_size as usize) < n_tokens_kv {
                    let w = weights.window_size as usize;
                    let offset_tokens = n_tokens_kv - w;
                    let offset_bytes = offset_tokens * kv.bytes_per_row;
                    let k_ptr = k_slot_ptr.offset_bytes(offset_bytes);
                    let v_ptr = v_slot_ptr.offset_bytes(offset_bytes);
                    (
                        w,
                        unsafe { Tensor::<F16>::from_raw(k_ptr, slot_stride_elems) },
                        unsafe { Tensor::<F16>::from_raw(v_ptr, slot_stride_elems) },
                    )
                } else {
                    (
                        n_tokens_kv,
                        unsafe { Tensor::<F16>::from_raw(k_slot_ptr, slot_stride_elems) },
                        unsafe { Tensor::<F16>::from_raw(v_slot_ptr, slot_stride_elems) },
                    )
                };
            let kernel_window = 0i32;
            let chunk_size = flambeau_model_ops::splitk_chunk_size(eff_n_tokens_kv);
            let n_chunks = eff_n_tokens_kv.div_ceil(chunk_size);
            let use_splitk = eff_n_tokens_kv > 256
                && n_chunks > 1
                && n_chunks <= crate::core::scratch::MAX_SPLITK_CHUNKS
                && state.pool.splitk_partials_m.as_usize() != 0;
            if use_splitk {
                let partials_m_n = weights.n_heads * n_chunks;
                let partials_o_n = partials_m_n * weights.head_dim;
                let mut partials_m =
                    unsafe { Tensor::<F32>::from_raw(state.pool.splitk_partials_m, partials_m_n) };
                let mut partials_s =
                    unsafe { Tensor::<F32>::from_raw(state.pool.splitk_partials_s, partials_m_n) };
                let mut partials_o =
                    unsafe { Tensor::<F32>::from_raw(state.pool.splitk_partials_o, partials_o_n) };
                flambeau_model_ops::attn_decode_f16_splitk(
                    &q_f16_rope,
                    &k_cache_eff_t,
                    &v_cache_eff_t,
                    &mut attn_out,
                    &mut partials_m,
                    &mut partials_s,
                    &mut partials_o,
                    flambeau_ops::AttnSplitkShape {
                        n_heads_q: weights.n_heads,
                        n_heads_kv: weights.n_kv_heads,
                        head_dim: weights.head_dim,
                        n_tokens_kv: eff_n_tokens_kv,
                        chunk_size,
                    },
                    flambeau_ops::AttnKnobs { scale, window_size: kernel_window },
                    &ops,
                )?;
            } else {
                flambeau_model_ops::attn_decode_f16(
                    &q_f16_rope,
                    &k_cache_eff_t,
                    &v_cache_eff_t,
                    &mut attn_out,
                    flambeau_ops::AttnDecodeShape {
                        n_heads_q: weights.n_heads,
                        n_heads_kv: weights.n_kv_heads,
                        head_dim: weights.head_dim,
                        n_tokens_kv: eff_n_tokens_kv,
                    },
                    flambeau_ops::AttnKnobs { scale, window_size: kernel_window },
                    &ops,
                )?;
            }
            let _ = (k_cache, v_cache);
            } // end of else (non-Q8) decode branch
        } else if is_q8 {
            let k_cache_q8 = unsafe {
                Tensor::<flambeau_model_ops::Q8_0>::from_raw(
                    k_slot_ptr,
                    max_seq_len * kv_width,
                )
            };
            let v_cache_q8 = unsafe {
                Tensor::<flambeau_model_ops::Q8_0>::from_raw(
                    v_slot_ptr,
                    max_seq_len * kv_width,
                )
            };
            let n_k_tokens = start_position + n;
            flambeau_model_ops::attn_prefill_q8_kv(
                &q_f16_rope,
                &k_cache_q8,
                &v_cache_q8,
                &mut attn_out,
                flambeau_ops::AttnPrefillShape {
                    n_q_tokens: n,
                    n_heads_q: weights.n_heads,
                    n_heads_kv: weights.n_kv_heads,
                    head_dim: weights.head_dim,
                    n_k_tokens,
                    q_offset: start_position,
                },
                flambeau_ops::AttnKnobs { scale, window_size: weights.window_size },
                &ops,
            )?;
            let _ = (k_cache, v_cache);
        } else {
            let n_k_tokens = start_position + n;
            flambeau_model_ops::attn_prefill_f16(
                &q_f16_rope,
                &k_cache,
                &v_cache,
                &mut attn_out,
                flambeau_ops::AttnPrefillShape {
                    n_q_tokens: n,
                    n_heads_q: weights.n_heads,
                    n_heads_kv: weights.n_kv_heads,
                    head_dim: weights.head_dim,
                    n_k_tokens,
                    q_offset: start_position,
                },
                flambeau_ops::AttnKnobs { scale, window_size: weights.window_size },
                &ops,
            )?;
        }
    } else if let Some(paged_caches) = state.pool.paged_kv_caches.as_ref() {
        // Paged path: K/V live in a shared page pool; per-slot KV is
        // looked up via a `[max_slots, max_pages_per_slot]` u32 block
        // table. The host acquires a new page from this layer's
        // PagePool on every `position % page_size == 0` boundary,
        // writes the page index into the per-slot row of the block
        // table, then uploads the patched row (or whole table) before
        // the kv_append + attention kernels fire.
        let paged_cache = paged_caches[kv_local_idx];
        let page_size = paged_cache.page_size;
        let mpps = paged_cache.max_pages_per_slot;
        let mut host_write_pos: Vec<i32> = Vec::with_capacity(n);
        let mut host_n_kv: Vec<i32> = Vec::with_capacity(n);
        // Patch the block table for any slot crossing a page boundary
        // this step. We acquire pages from this layer's PagePool and
        // memcpy the updated u32 to the device-side block_tables.
        for i in 0..n {
            let slot = slot_ids[i];
            let pos = positions[i];
            host_write_pos.push(pos as i32);
            host_n_kv.push((pos + 1) as i32);
            if pos >= page_size * mpps {
                bail!(
                    "standard_attn paged: slot {slot} position {pos} exceeds \
                     page_size*max_pages_per_slot ({page_size}*{mpps})"
                );
            }
            let page_idx_in_slot = pos / page_size;
            if pos % page_size == 0 {
                let page = state.pool.page_pools[kv_local_idx]
                    .acquire_for(slot)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "standard_attn paged: PagePool for layer {layer_idx} exhausted at \
                             slot {slot} pos {pos} (n_pages={}, free={})",
                            paged_cache.n_pages,
                            state.pool.page_pools[kv_local_idx].n_free()
                        )
                    })?;
                let host_page = [page];
                let table_entry_offset_bytes =
                    (slot * mpps + page_idx_in_slot) * std::mem::size_of::<u32>();
                let dst = paged_cache
                    .block_tables
                    .offset_bytes(table_entry_offset_bytes);
                // SAFETY: block_tables owns max_slots * mpps * 4 bytes;
                // table_entry_offset_bytes < max_slots * mpps * 4.
                unsafe {
                    state.device.memcpy_async(
                        state.stream,
                        CopyDirection::HostToDevice,
                        dst,
                        DevicePtr(host_page.as_ptr() as usize),
                        std::mem::size_of::<u32>(),
                    )?;
                }
            }
        }
        // Reuse attn_slot_write_pos / attn_slot_n_kv scratch for the
        // per-slot scalar arrays (sized for max_slots * i32 already).
        if state.pool.attn_slot_write_pos.as_usize() == 0 {
            bail!(
                "standard_attn paged: non-prefill shape requires max_slots > 1 in ScratchConfig \
                 (got max_slots={})",
                state.pool.config.max_slots
            );
        }
        // SAFETY: attn_slot_write_pos / attn_slot_n_kv both own >= n * 4
        // bytes when max_slots > 1 (allocated in scratch.rs:483).
        unsafe {
            state.device.memcpy_async(
                state.stream,
                CopyDirection::HostToDevice,
                state.pool.attn_slot_write_pos,
                DevicePtr(host_write_pos.as_ptr() as usize),
                n * 4,
            )?;
            state.device.memcpy_async(
                state.stream,
                CopyDirection::HostToDevice,
                state.pool.attn_slot_n_kv,
                DevicePtr(host_n_kv.as_ptr() as usize),
                n * 4,
            )?;
        }
        flambeau_core::Stream::synchronize(state.stream)?;
        let k_src_full = unsafe { Tensor::<F16>::from_raw(state.pool.k_f16, n * kv_width) };
        let v_src_full = unsafe { Tensor::<F16>::from_raw(state.pool.v_f16, n * kv_width) };
        if !is_kv_shared {
            flambeau_model_ops::kv_append_f16_paged_slots(
                flambeau_ops::KvAppendPagedSlotsBuffers {
                    k_src: k_src_full.ptr,
                    v_src: v_src_full.ptr,
                    k_pool: paged_cache.k_pool,
                    v_pool: paged_cache.v_pool,
                    block_tables: paged_cache.block_tables,
                    slot_write_pos: state.pool.attn_slot_write_pos,
                },
                flambeau_ops::KvAppendPagedSlotsShape {
                    n_slots: n,
                    kv_width,
                    page_size,
                    max_pages_per_slot: mpps,
                },
                &ops,
            )?;
        }
        let _ = (&k_src_full, &v_src_full);
        flambeau_model_ops::attn_decode_f16_paged(
            flambeau_ops::AttnPagedDecodeBuffers {
                q_batched: state.pool.q_f16,
                k_pool: paged_cache.k_pool,
                v_pool: paged_cache.v_pool,
                block_tables: paged_cache.block_tables,
                out_batched: state.pool.attn_out_f16,
                n_tokens_kv_ptrs: state.pool.attn_slot_n_kv,
            },
            flambeau_ops::AttnDecodePagedShape {
                n_heads_q: weights.n_heads,
                n_heads_kv: weights.n_kv_heads,
                head_dim: weights.head_dim,
                n_slots: n,
                page_size,
                max_pages_per_slot: mpps,
            },
            scale,
            &ops,
        )?;
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
        // SWA window-offset: each slot independently slides its K/V
        // read pointer to its window start. The batched attention
        // kernel then sees a per-slot `n_kv = window` cache starting
        // at relative position 0; passing `kernel_window = 0` makes
        // the kernel's internal SWA mask a no-op. kv_append still
        // writes to the absolute slot position, so we use a separate
        // per-slot pointer set for the append step (slot bases,
        // unmodified) and upload the offset pointers later before
        // the attention launch.
        let window_size = weights.window_size;
        let window = window_size as usize;
        let bytes_per_row = kv.bytes_per_row;
        let mut host_k_base_ptrs: Vec<u64> = Vec::with_capacity(n);
        let mut host_v_base_ptrs: Vec<u64> = Vec::with_capacity(n);
        let mut host_k_read_ptrs: Vec<u64> = Vec::with_capacity(n);
        let mut host_v_read_ptrs: Vec<u64> = Vec::with_capacity(n);
        let mut host_write_pos: Vec<i32> = Vec::with_capacity(n);
        let mut host_n_kv: Vec<i32> = Vec::with_capacity(n);
        let mut any_offset = false;
        for i in 0..n {
            let slot = slot_ids[i];
            let pos = positions[i];
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
        // When we offset at least one slot the per-slot windowing
        // bounds reads to `[0..n_kv[i])` and the kernel's internal
        // SWA mask is not needed. When no slot was offset the
        // original window param still works (it's either 0 or
        // covers all slot positions).
        let kernel_window: i32 = if any_offset { 0 } else { window_size };
        // Suppress unused warning when SWA offset doesn't apply.
        let _ = &host_k_read_ptrs;
        let _ = &host_v_read_ptrs;
        let swa_offset_active = any_offset;
        // Step 1: upload slot-base pointers (used for kv_append) and
        // write_pos + n_kv. Run kv_append.
        unsafe {
            state
                .device
                .memcpy_async(
                    state.stream,
                    CopyDirection::HostToDevice,
                    state.pool.attn_slot_k_dst_ptrs,
                    DevicePtr(host_k_base_ptrs.as_ptr() as usize),
                    n * 8,
                )
                .context("standard_attn: attn_slot_k_dst_ptrs HtoD")?;
            state
                .device
                .memcpy_async(
                    state.stream,
                    CopyDirection::HostToDevice,
                    state.pool.attn_slot_v_dst_ptrs,
                    DevicePtr(host_v_base_ptrs.as_ptr() as usize),
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
                flambeau_ops::KvAppendBatchedSlotsBuffers {
                    k_src: k_src_full.ptr,
                    v_src: v_src_full.ptr,
                    slot_k_dst_ptrs: state.pool.attn_slot_k_dst_ptrs,
                    slot_v_dst_ptrs: state.pool.attn_slot_v_dst_ptrs,
                    slot_write_pos: state.pool.attn_slot_write_pos,
                },
                flambeau_ops::KvAppendBatchedSlotsShape { n_slots: n, kv_width },
                &ops,
            )?;
        }
        let _ = (&k_src_full, &v_src_full);
        // Step 2: when SWA windowing applies, replace the slot ptr
        // arrays with the offset (window-start) ptrs before
        // attention. The kv_append above already landed against the
        // absolute slot bases; attention reads only the window from
        // the offset position. `kernel_window = 0` makes the kernel's
        // SWA mask a no-op.
        if swa_offset_active {
            unsafe {
                state
                    .device
                    .memcpy_async(
                        state.stream,
                        CopyDirection::HostToDevice,
                        state.pool.attn_slot_k_dst_ptrs,
                        DevicePtr(host_k_read_ptrs.as_ptr() as usize),
                        n * 8,
                    )
                    .context("standard_attn: attn_slot_k_dst_ptrs HtoD (SWA offset)")?;
                state
                    .device
                    .memcpy_async(
                        state.stream,
                        CopyDirection::HostToDevice,
                        state.pool.attn_slot_v_dst_ptrs,
                        DevicePtr(host_v_read_ptrs.as_ptr() as usize),
                        n * 8,
                    )
                    .context("standard_attn: attn_slot_v_dst_ptrs HtoD (SWA offset)")?;
            }
            flambeau_core::Stream::synchronize(state.stream)?;
        }
        flambeau_model_ops::attn_decode_f16_batched(
            flambeau_ops::AttnBatchedBuffers {
                q_batched: state.pool.q_f16,
                k_cache_ptrs: state.pool.attn_slot_k_dst_ptrs,
                v_cache_ptrs: state.pool.attn_slot_v_dst_ptrs,
                out_batched: state.pool.attn_out_f16,
                n_tokens_kv_ptrs: state.pool.attn_slot_n_kv,
            },
            flambeau_ops::AttnDecodeBatchedShape {
                n_heads_q: weights.n_heads,
                n_heads_kv: weights.n_kv_heads,
                head_dim: weights.head_dim,
                n_slots: n,
            },
            flambeau_ops::AttnKnobs {
                scale,
                window_size: kernel_window,
            },
            &ops,
        )?;
    }

    let post_attn_ptr = if weights.attn_q_gated {
        let gate = unsafe { Tensor::<F16>::from_raw(state.pool.gate_f16, n * q_width) };
        let attn_in = unsafe { Tensor::<F16>::from_raw(state.pool.attn_out_f16, n * q_width) };
        let mut gated_out = unsafe { Tensor::<F16>::from_raw(state.pool.q_fused_f16, n * q_width) };
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
        && (hooks.supports_ar_residual_rmsnorm_f16() || hooks.supports_ar_residual_f16());
    let mut proj_f32 = unsafe { Tensor::<F32>::from_raw(state.pool.attn_proj_f32, n * hidden) };
    if !f16_fast {
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
    }
    // Fast path: BAR1 TP=2 with no post-attn-norm folds the
    // F32→F16 cast + AR + residual-add into 1 launch (residual_tp2)
    // instead of 4 (qmatmul + cast + ar_sum + add). Returns None so the
    // model skips its own residual_add.
    if let Some(next_w) = next_norm
        .filter(|_| n == 1
            && weights.post_attn_norm.is_none()
            && hooks.supports_ar_residual_rmsnorm_f16())
    {
        let mut partial_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.delta, n * hidden) };
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
        let mut partial_f16 = unsafe { Tensor::<F16>::from_raw(state.pool.delta, n * hidden) };
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
    if let Some(post_norm) = weights.post_attn_norm.as_ref() {
        // Fused F32→F16 rmsnorm + add to residual. Advances the
        // residual slot ourselves and writes the new residual
        // directly. residual_add_local picks this up via the
        // fused_residual_already_done flag and skips re-doing the add.
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
        Ok(Some(new_resid))
    } else {
        let mut delta_mut = unsafe { Tensor::<F16>::from_raw(state.pool.delta, n * hidden) };
        flambeau_model_ops::cast_f32_to_f16(&proj_f32, &mut delta_mut, n * hidden, &ops)?;
        let delta = unsafe { Tensor::<F16>::from_raw(state.pool.delta, n * hidden) };
        Ok(Some(delta))
    }
}
