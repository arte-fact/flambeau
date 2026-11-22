//! Attention — decode (F16 KV + Q8_0 KV) and prefill (F16 KV).
//! GQA is handled via the `n_heads_kv` argument; the kernel broadcasts one
//! KV head across `n_heads_q / n_heads_kv` Q heads.

use anyhow::Result;
use flambeau_core::{LaunchArgs, LaunchCfg};


/// Decode attention, F16 KV. One Q row per call (`n_tokens_q = 1` by
/// construction — this is the hot-path for token generation). Kernel does
/// online-softmax over the full KV cache.
/// Shapes:
/// - `q[n_heads_q, head_dim]` F16
/// - `k_cache[n_tokens_kv, n_heads_kv, head_dim]` F16 contiguous
/// - `v_cache` same shape as K
/// - `out[n_heads_q, head_dim]` F16
///   Launch: one block per Q head, `head_dim` threads/block (one thread per
///   output lane). Kernel supports `head_dim ∈ {128, 256}` — both
///   Qwen3.5 (GQA-32/4, head_dim=128) and Qwen3.6 (GQA-16/2, head_dim=256).
pub fn attention_decode_f16<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::AttnBuffers,
    shape: crate::AttnDecodeShape,
    knobs: crate::AttnKnobs,
) -> Result<()> {
    attention_decode_f16_slots(reg, stream, buffers, shape, knobs, None)
}

/// Graph-captureable variant of [`attention_decode_f16`].
/// Identical behaviour for non-capture callers (slots=None). When
/// `slots` is Some, the n_tokens_kv kernel arg is tagged via
/// `KernelArgs::push_slot` so the graph recorder can bind the slot.
/// Caller then updates the slot per replay via `HipGraphExec::set_slot`
/// to track the growing KV cache tail.
pub fn attention_decode_f16_slots<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::AttnBuffers,
    shape: crate::AttnDecodeShape,
    knobs: crate::AttnKnobs,
    slots: Option<crate::AttnDecodeSlots>,
) -> Result<()> {
    let crate::AttnBuffers { q, k, v, out } = buffers;
    let crate::AttnDecodeShape {
        n_heads_q,
        n_heads_kv,
        head_dim,
        n_tokens_kv,
    } = shape;
    let crate::AttnKnobs { scale, window_size, ring_depth } = knobs;
    assert!(
        head_dim == 64 || head_dim == 128 || head_dim == 256 || head_dim == 512,
        "attention_decode_f16: head_dim {head_dim} not supported (expected 64, 128, 256, or 512)"
    );
    let module = reg.expect_module("attention_decode_f16")?;
    let kernel = B::kernel(module, "flambeau_attention_decode_f16")?;

    let n_heads_q_i = n_heads_q as i32;
    let n_heads_kv_i = n_heads_kv as i32;
    let head_dim_i = head_dim as i32;
    let n_tokens_i = n_tokens_kv as i32;
    let q_ptr: u64 = q.as_usize() as u64;
    let k_ptr: u64 = k.as_usize() as u64;
    let v_ptr: u64 = v.as_usize() as u64;
    let o_ptr: u64 = out.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&q_ptr);
    args.push_arg(&k_ptr);
    args.push_arg(&v_ptr);
    args.push_arg(&o_ptr);
    args.push_arg(&n_heads_q_i);
    args.push_arg(&n_heads_kv_i);
    args.push_arg(&head_dim_i);
    match slots {
        Some(s) => args.push_slot(&n_tokens_i, s.n_tokens_kv),
        None => args.push_arg(&n_tokens_i),
    }
    args.push_arg(&scale);
    args.push_arg(&window_size);
    args.push_arg(&ring_depth);
    let cfg = LaunchCfg::one_d(n_heads_q as u32, head_dim as u32);
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Single-launch batched-decode attention over `n_slots`
/// (Q-row, per-slot KV-cache) pairs. Replaces the per-slot loop in
/// `forward_full_attn_layer_decode_batched_*`, which structurally caps
/// hybrid throughput.
/// Each grid block owns one `(q_head, slot)` pair and runs the same
/// flash-attn-v2 online-softmax body as [`attention_decode_f16`]. Slot
/// addressing comes from device-side tables built once per call by the
/// dispatcher:
/// - `k_cache_ptrs[n_slots]`: device addresses of each slot's K cache.
/// - `v_cache_ptrs[n_slots]`: device addresses of each slot's V cache.
/// - `n_tokens_kv[n_slots]`: per-slot KV-tail length (post-append).
///   Q and out are contiguous batched layouts:
///   `q[n_slots, n_heads_q, head_dim]` F16
///   `out[n_slots, n_heads_q, head_dim]` F16
///   Per-slot K/V layouts (pointed-to memory) match
///   [`attention_decode_f16`]: `[n_tokens, n_heads_kv, head_dim]` F16.
///   Launch: `gridDim = (n_heads_q, n_slots)`, `blockDim = (head_dim,)`.
///   At `n_slots = 1` the kernel produces output bit-identical to
///   [`attention_decode_f16_slots`] for the same inputs.
/// # Safety
/// All device pointers must outlive the kernel launch and remain valid
/// on the stream's device. `k_cache_ptrs` / `v_cache_ptrs` / `n_tokens_kv`
/// must each point at device buffers of length ≥ `n_slots`. Each
/// `k_cache_ptrs[s]` and `v_cache_ptrs[s]` must point at ≥ `n_tokens_kv[s] *
/// n_heads_kv * head_dim` F16 elements. `q` / `out` must each point at
/// ≥ `n_slots * n_heads_q * head_dim` F16 elements.
pub fn attention_decode_f16_batched<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::AttnBatchedBuffers,
    shape: crate::AttnDecodeBatchedShape,
    knobs: crate::AttnKnobs,
) -> Result<()> {
    let crate::AttnBatchedBuffers {
        q_batched,
        k_cache_ptrs,
        v_cache_ptrs,
        out_batched,
        n_tokens_kv_ptrs: n_tokens_kv,
    } = buffers;
    let crate::AttnDecodeBatchedShape {
        n_heads_q,
        n_heads_kv,
        head_dim,
        n_slots,
    } = shape;
    let crate::AttnKnobs { scale, window_size, ring_depth } = knobs;
    assert!(
        head_dim == 64 || head_dim == 128 || head_dim == 256 || head_dim == 512,
        "attention_decode_f16_batched: head_dim {head_dim} not supported (expected 64, 128, 256, or 512)"
    );
    assert!(
        (1..=32).contains(&n_slots),
        "attention_decode_f16_batched: n_slots {n_slots} out of supported range [1, 32]"
    );
    let module = reg.expect_module("attention_decode_f16_batched")?;
    let kernel = B::kernel(module, "flambeau_attention_decode_f16_batched")?;

    let n_heads_q_i = n_heads_q as i32;
    let n_heads_kv_i = n_heads_kv as i32;
    let head_dim_i = head_dim as i32;
    let n_slots_i = n_slots as i32;
    let q_ptr: u64 = q_batched.as_usize() as u64;
    let k_ptrs_ptr: u64 = k_cache_ptrs.as_usize() as u64;
    let v_ptrs_ptr: u64 = v_cache_ptrs.as_usize() as u64;
    let o_ptr: u64 = out_batched.as_usize() as u64;
    let n_kv_ptr: u64 = n_tokens_kv.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&q_ptr);
    args.push_arg(&k_ptrs_ptr);
    args.push_arg(&v_ptrs_ptr);
    args.push_arg(&o_ptr);
    args.push_arg(&n_kv_ptr);
    args.push_arg(&n_heads_q_i);
    args.push_arg(&n_heads_kv_i);
    args.push_arg(&head_dim_i);
    args.push_arg(&n_slots_i);
    args.push_arg(&scale);
    args.push_arg(&window_size);
    args.push_arg(&ring_depth);
    let cfg = LaunchCfg {
        grid: (n_heads_q as u32, n_slots as u32, 1),
        block: (head_dim as u32, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// PagedAttention sibling of [`attention_decode_f16_batched`]. Same
/// flash-attn-v2 online-softmax body; K/V per-token rows are fetched
/// from a shared `[n_pages, page_size, kv_width]` F16 page pool via
/// a per-slot block table indirection.
///
/// `block_tables` is `[n_slots, max_pages_per_slot]` `u32` device
/// memory, row-major. For token `t` of slot `s`, the page that holds
/// it is `block_tables[s * max_pages_per_slot + t / page_size]` and
/// the in-page row is `t % page_size`. `page_size` MUST be a power
/// of two so the kernel can replace the divide / modulo with shifts
/// and AND masks.
///
/// Identity-mapped block table (`block_tables[s][p] = s * max_pages
/// _per_slot + p`) with `n_pages = n_slots * max_pages_per_slot`
/// makes the output bit-identical to
/// [`attention_decode_f16_batched`] running on the same K/V data —
/// that's the regression guard the paired test relies on.
///
/// # Safety
/// All device pointers must outlive the kernel launch and remain
/// valid on the stream's device. `k_pool` / `v_pool` must point at
/// ≥ `n_pages * page_size * (n_heads_kv * head_dim)` F16 elements.
/// `block_tables` must point at ≥ `n_slots * max_pages_per_slot`
/// u32 elements. `n_tokens_kv` must point at ≥ `n_slots` i32
/// elements; each entry must satisfy
/// `(n_tokens_kv[s] - 1) / page_size < max_pages_per_slot`. `q_batched`
/// and `out_batched` must each point at ≥ `n_slots * n_heads_q *
/// head_dim` F16 elements.
pub fn attention_decode_f16_paged<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::AttnPagedDecodeBuffers,
    shape: crate::AttnDecodePagedShape,
    scale: f32,
) -> Result<()> {
    let crate::AttnPagedDecodeBuffers {
        q_batched,
        k_pool,
        v_pool,
        block_tables,
        out_batched,
        n_tokens_kv_ptrs: n_tokens_kv,
    } = buffers;
    let crate::AttnDecodePagedShape {
        n_heads_q,
        n_heads_kv,
        head_dim,
        n_slots,
        page_size,
        max_pages_per_slot,
    } = shape;
    assert!(
        head_dim == 64 || head_dim == 128 || head_dim == 256 || head_dim == 512,
        "attention_decode_f16_paged: head_dim {head_dim} not supported (expected 64, 128, 256, or 512)"
    );
    assert!(
        (1..=32).contains(&n_slots),
        "attention_decode_f16_paged: n_slots {n_slots} out of supported range [1, 32]"
    );
    assert!(
        page_size > 0 && page_size.is_power_of_two(),
        "attention_decode_f16_paged: page_size {page_size} must be a positive power of two"
    );
    assert!(
        max_pages_per_slot >= 1,
        "attention_decode_f16_paged: max_pages_per_slot must be >= 1"
    );
    let module = reg.expect_module("attention_decode_f16_paged")?;
    let kernel = B::kernel(module, "flambeau_attention_decode_f16_paged")?;

    let n_heads_q_i = n_heads_q as i32;
    let n_heads_kv_i = n_heads_kv as i32;
    let head_dim_i = head_dim as i32;
    let n_slots_i = n_slots as i32;
    let page_size_i = page_size as i32;
    let max_pps_i = max_pages_per_slot as i32;
    let q_ptr: u64 = q_batched.as_usize() as u64;
    let k_pool_ptr: u64 = k_pool.as_usize() as u64;
    let v_pool_ptr: u64 = v_pool.as_usize() as u64;
    let bt_ptr: u64 = block_tables.as_usize() as u64;
    let o_ptr: u64 = out_batched.as_usize() as u64;
    let n_kv_ptr: u64 = n_tokens_kv.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&q_ptr);
    args.push_arg(&k_pool_ptr);
    args.push_arg(&v_pool_ptr);
    args.push_arg(&bt_ptr);
    args.push_arg(&o_ptr);
    args.push_arg(&n_kv_ptr);
    args.push_arg(&n_heads_q_i);
    args.push_arg(&n_heads_kv_i);
    args.push_arg(&head_dim_i);
    args.push_arg(&n_slots_i);
    args.push_arg(&page_size_i);
    args.push_arg(&max_pps_i);
    args.push_arg(&scale);
    let cfg = LaunchCfg {
        grid: (n_heads_q as u32, n_slots as u32, 1),
        block: (head_dim as u32, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// PagedAttention sibling of [`kv_append_f16_batched_slots`]. Writes
/// one new K+V row per slot into a shared `[n_pages, page_size,
/// kv_width]` F16 page pool, with the per-slot destination resolved
/// through a `[n_slots, max_pages_per_slot]` u32 block table.
///
/// The host allocator must populate
/// `block_tables[s * max_pages_per_slot + slot_write_pos[s] / page_size]`
/// with a valid page index before this kernel fires; the kernel
/// never allocates pages.
///
/// # Safety
/// Mirrors [`attention_decode_f16_paged`]'s requirements. `k_src` /
/// `v_src` must point at ≥ `n_slots * kv_width` F16 elements.
/// `slot_write_pos` must point at ≥ `n_slots` i32 elements with each
/// `(slot_write_pos[s] / page_size) < max_pages_per_slot`.
pub fn kv_append_f16_paged_slots<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buf: crate::KvAppendPagedSlotsBuffers,
    shape: crate::KvAppendPagedSlotsShape,
) -> Result<()> {
    assert!(
        shape.page_size > 0 && shape.page_size.is_power_of_two(),
        "kv_append_f16_paged_slots: page_size {} must be a positive power of two",
        shape.page_size
    );
    assert!(
        shape.max_pages_per_slot >= 1,
        "kv_append_f16_paged_slots: max_pages_per_slot must be >= 1"
    );
    let module = reg.expect_module("kv_append_f16_paged_slots")?;
    let kernel = B::kernel(module, "flambeau_kv_append_f16_paged_slots")?;

    let n_slots_i = shape.n_slots as i32;
    let kv_width_i = shape.kv_width as i32;
    let page_size_i = shape.page_size as i32;
    let max_pps_i = shape.max_pages_per_slot as i32;
    let k_src_ptr: u64 = buf.k_src.as_usize() as u64;
    let v_src_ptr: u64 = buf.v_src.as_usize() as u64;
    let k_pool_ptr: u64 = buf.k_pool.as_usize() as u64;
    let v_pool_ptr: u64 = buf.v_pool.as_usize() as u64;
    let bt_ptr: u64 = buf.block_tables.as_usize() as u64;
    let wpos_ptr: u64 = buf.slot_write_pos.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&k_src_ptr);
    args.push_arg(&v_src_ptr);
    args.push_arg(&k_pool_ptr);
    args.push_arg(&v_pool_ptr);
    args.push_arg(&bt_ptr);
    args.push_arg(&wpos_ptr);
    args.push_arg(&n_slots_i);
    args.push_arg(&kv_width_i);
    args.push_arg(&page_size_i);
    args.push_arg(&max_pps_i);
    let block_threads: u32 = shape.kv_width.min(128) as u32;
    let cfg = LaunchCfg {
        grid: (shape.n_slots as u32, 1, 1),
        block: (block_threads.max(1), 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// PagedAttention prefill attention sibling of
/// `attention_prefill_f16`. Same flash-attn-v2 online-softmax body;
/// per-`t` K/V row resolved via
/// `block_table[t / page_size] * page_size + (t & (page_size - 1))`.
/// `page_size` MUST be a power of two so the divide and modulo
/// compile to shifts and AND masks.
///
/// Identity-mapped `block_table` (`block_table[p] = p`) with
/// `n_pages * page_size >= n_k_tokens` makes the output bit-identical
/// to `attention_prefill_f16` running on the same K/V data — the
/// regression guard the paired parity test relies on.
///
/// # Safety
/// Mirrors `attention_decode_f16_paged`'s safety contract for K/V
/// pool sizes and block-table extent.
pub fn attention_prefill_f16_paged<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::AttnPagedPrefillBuffers,
    shape: crate::AttnPrefillPagedShape,
    knobs: crate::AttnKnobs,
) -> Result<()> {
    let crate::AttnPagedPrefillBuffers {
        q,
        k_pool,
        v_pool,
        block_table,
        out,
    } = buffers;
    let crate::AttnPrefillPagedShape {
        n_q_tokens,
        n_heads_q,
        n_heads_kv,
        head_dim,
        n_k_tokens,
        q_offset,
        page_size,
    } = shape;
    let crate::AttnKnobs { scale, window_size, ring_depth } = knobs;
    assert!(
        head_dim == 64 || head_dim == 128 || head_dim == 256 || head_dim == 512,
        "attention_prefill_f16_paged: head_dim {head_dim} not supported (expected 64, 128, 256, or 512)"
    );
    assert!(
        page_size > 0 && page_size.is_power_of_two(),
        "attention_prefill_f16_paged: page_size {page_size} must be a positive power of two"
    );
    let module = reg.expect_module("attention_prefill_f16_paged")?;
    let kernel = B::kernel(module, "flambeau_attention_prefill_f16_paged")?;

    let n_q_i = n_q_tokens as i32;
    let n_heads_q_i = n_heads_q as i32;
    let n_heads_kv_i = n_heads_kv as i32;
    let head_dim_i = head_dim as i32;
    let n_k_i = n_k_tokens as i32;
    let q_off_i = q_offset as i32;
    let page_size_i = page_size as i32;
    let q_ptr: u64 = q.as_usize() as u64;
    let k_ptr: u64 = k_pool.as_usize() as u64;
    let v_ptr: u64 = v_pool.as_usize() as u64;
    let bt_ptr: u64 = block_table.as_usize() as u64;
    let o_ptr: u64 = out.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&q_ptr);
    args.push_arg(&k_ptr);
    args.push_arg(&v_ptr);
    args.push_arg(&bt_ptr);
    args.push_arg(&o_ptr);
    args.push_arg(&n_q_i);
    args.push_arg(&n_heads_q_i);
    args.push_arg(&n_heads_kv_i);
    args.push_arg(&head_dim_i);
    args.push_arg(&n_k_i);
    args.push_arg(&q_off_i);
    args.push_arg(&page_size_i);
    args.push_arg(&scale);
    args.push_arg(&window_size);
    args.push_arg(&ring_depth);
    let cfg = LaunchCfg {
        grid: (n_q_tokens as u32, n_heads_q as u32, 1),
        block: (head_dim as u32, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// PagedAttention sibling of `kv_append_f16` for prefill. Writes L
/// K + V rows for a single slot's prefill into the slot's paged KV
/// cache, walking the slot's row of the block table per token.
///
/// `block_table` is a pointer at the slot's row of the global
/// `[max_slots, max_pages_per_slot]` block table — i.e. element 0
/// of `block_tables_global + slot * max_pages_per_slot * 4 bytes`.
/// The host must pre-populate it for the position range
/// `[start_pos, start_pos + n_tokens)` BEFORE this kernel fires
/// (typically by calling `PagePool::acquire_for` for each new
/// `position % page_size == 0` boundary).
///
/// `page_size` MUST be a power of two.
///
/// # Safety
/// Mirrors [`kv_append_f16_paged_slots`]. `k_pool` / `v_pool` must
/// own `≥ n_pages * page_size * kv_width` F16 elements. `block_table`
/// must point at ≥ `(start_pos + n_tokens) / page_size + 1` u32
/// elements. `k_src` / `v_src` must each point at ≥ `n_tokens *
/// kv_width` F16 elements.
pub fn kv_append_f16_paged_prefill<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buf: crate::KvAppendPagedPrefillBuffers,
    shape: crate::KvAppendPagedPrefillShape,
) -> Result<()> {
    assert!(
        shape.page_size > 0 && shape.page_size.is_power_of_two(),
        "kv_append_f16_paged_prefill: page_size {} must be a positive power of two",
        shape.page_size
    );
    let module = reg.expect_module("kv_append_f16_paged_prefill")?;
    let kernel = B::kernel(module, "flambeau_kv_append_f16_paged_prefill")?;

    let n_tokens_i = shape.n_tokens as i32;
    let kv_width_i = shape.kv_width as i32;
    let start_pos_i = shape.start_pos as i32;
    let page_size_i = shape.page_size as i32;
    let k_src_ptr: u64 = buf.k_src.as_usize() as u64;
    let v_src_ptr: u64 = buf.v_src.as_usize() as u64;
    let k_pool_ptr: u64 = buf.k_pool.as_usize() as u64;
    let v_pool_ptr: u64 = buf.v_pool.as_usize() as u64;
    let bt_ptr: u64 = buf.block_table.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&k_src_ptr);
    args.push_arg(&v_src_ptr);
    args.push_arg(&k_pool_ptr);
    args.push_arg(&v_pool_ptr);
    args.push_arg(&bt_ptr);
    args.push_arg(&n_tokens_i);
    args.push_arg(&kv_width_i);
    args.push_arg(&start_pos_i);
    args.push_arg(&page_size_i);
    let block_threads: u32 = shape.kv_width.min(128) as u32;
    let cfg = LaunchCfg {
        grid: (shape.n_tokens as u32, 1, 1),
        block: (block_threads.max(1), 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Batched K+V append across N decode slots, each writing one new
/// token row into its own KV cache. Companion to
/// [`attention_decode_f16_batched`] — replaces the N×2
/// `memcpy_async(DtoD)` calls that previously walked per-slot KV
/// caches.
///
/// `slot_k_dst_ptrs` / `slot_v_dst_ptrs` are `[N] u64` device arrays
/// of per-slot KV-cache base pointers (one per slot). `slot_write_pos`
/// is `[N] i32` with each slot's pre-bump tail index. The kernel
/// writes one row of `kv_width = n_kv_heads * head_dim` F16 values
/// per slot from `k_src` / `v_src` (both `[N, kv_width]` slot-major)
/// at `dst + write_pos * kv_width`.
pub fn kv_append_f16_batched_slots<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buf: crate::KvAppendBatchedSlotsBuffers,
    shape: crate::KvAppendBatchedSlotsShape,
) -> Result<()> {
    let module = reg.expect_module("kv_append_f16_batched_slots")?;
    let kernel = B::kernel(module, "flambeau_kv_append_f16_batched_slots")?;

    let n_slots_i = shape.n_slots as i32;
    let kv_width_i = shape.kv_width as i32;
    let ring_depth_i = shape.ring_depth as i32;
    let k_src_ptr: u64 = buf.k_src.as_usize() as u64;
    let v_src_ptr: u64 = buf.v_src.as_usize() as u64;
    let k_dst_arr: u64 = buf.slot_k_dst_ptrs.as_usize() as u64;
    let v_dst_arr: u64 = buf.slot_v_dst_ptrs.as_usize() as u64;
    let wpos_ptr: u64 = buf.slot_write_pos.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&k_src_ptr);
    args.push_arg(&v_src_ptr);
    args.push_arg(&k_dst_arr);
    args.push_arg(&v_dst_arr);
    args.push_arg(&wpos_ptr);
    args.push_arg(&n_slots_i);
    args.push_arg(&kv_width_i);
    args.push_arg(&ring_depth_i);
    let block_threads: u32 = shape.kv_width.min(128) as u32;
    let cfg = LaunchCfg {
        grid: (shape.n_slots as u32, 1, 1),
        block: (block_threads.max(1), 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Split-K (flash-decoding) decode attention, F16 KV. Same math
/// as [`attention_decode_f16`] but partitions the context across grid.y to
/// attack the single-pass kernel's occupancy starvation on Qwen3.6
/// (16 heads × 1 block = 27 % of 60 CUs at head_dim=256).
/// Two passes:
/// 1. `flambeau_attention_decode_f16_splitk_chunk` — grid
///    `(n_heads_q, n_chunks)`, each block owns one (q_head, chunk) pair
///    and emits `(m_c, s_c, o_c[head_dim])` into `partials_*` scratch.
/// 2. `flambeau_attention_decode_f16_splitk_combine` — grid
///    `(n_heads_q,)`, merges the `n_chunks` partials per head into the
///    final output using online-softmax rescaling.
///
/// Scratch sizing (caller-provided, f32):
/// * `partials_m` / `partials_s`: `n_heads_q * n_chunks` floats each
/// * `partials_o`: `n_heads_q * n_chunks * head_dim` floats
///
/// Measured (Qwen3.6 shape, head_dim=256, 16/2, MI50):
/// * n_tokens=2048: single-pass 2647 µs → split-K 340 µs = **7.78×**
/// * n_tokens=4096: single-pass 5210 µs → split-K 662 µs = **7.87×**
pub fn attention_decode_f16_splitk<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::AttnBuffers,
    partials: crate::AttnSplitkPartials,
    shape: crate::AttnSplitkShape,
    knobs: crate::AttnKnobs,
) -> Result<()> {
    attention_decode_f16_splitk_slots(reg, stream, buffers, partials, shape, knobs, None)
}

/// Graph-captureable variant of [`attention_decode_f16_splitk`]. With
/// `slots=None` the behaviour is byte-identical to the wrapper. With `Some`,
/// the chunk kernel's `n_tokens_kv` + `chunk_base` scalar args — and, when
/// `k_ptr`/`v_ptr` are `Some`, its K/V cache pointer args (SWA SlideBand
/// slide) — are tagged via `KernelArgs::push_slot` so a captured graph can
/// rebind them per replay — valid while the derived `n_chunks` grid is
/// unchanged (a token that would change `n_chunks` needs recapture, not a
/// slot update). Only the chunk kernel reads K/V; the combine kernel reads
/// partials and is never tagged.
pub fn attention_decode_f16_splitk_slots<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::AttnBuffers,
    partials: crate::AttnSplitkPartials,
    shape: crate::AttnSplitkShape,
    knobs: crate::AttnKnobs,
    slots: Option<crate::AttnSplitkSlots>,
) -> Result<()> {
    let (n_tokens_slot, chunk_base_slot, k_ptr_slot, v_ptr_slot) = match slots {
        Some(s) => (Some(s.n_tokens_kv), Some(s.chunk_base), s.k_ptr, s.v_ptr),
        None => (None, None, None, None),
    };
    let crate::AttnBuffers { q, k, v, out } = buffers;
    let crate::AttnSplitkPartials {
        partials_m,
        partials_s,
        partials_o,
    } = partials;
    let crate::AttnSplitkShape {
        n_heads_q,
        n_heads_kv,
        head_dim,
        n_tokens_kv,
        chunk_size,
        chunk_base,
    } = shape;
    let crate::AttnKnobs { scale, window_size, ring_depth } = knobs;
    assert!(
        head_dim == 64 || head_dim == 128 || head_dim == 256 || head_dim == 512,
        "attention_decode_f16_splitk: head_dim {head_dim} not supported"
    );
    assert!(chunk_size > 0);
    assert!(chunk_base <= n_tokens_kv);

    let module = reg.expect_module("attention_decode_f16_splitk")?;
    let k_chunk = B::kernel(module, "flambeau_attention_decode_f16_splitk_chunk")?;
    let k_combine = B::kernel(module, "flambeau_attention_decode_f16_splitk_combine")?;

    let n_chunks = (n_tokens_kv - chunk_base).div_ceil(chunk_size);
    let n_heads_q_i = n_heads_q as i32;
    let n_heads_kv_i = n_heads_kv as i32;
    let head_dim_i = head_dim as i32;
    let n_tokens_i = n_tokens_kv as i32;
    let n_chunks_i = n_chunks as i32;
    let chunk_size_i = chunk_size as i32;
    let chunk_base_i = chunk_base as i32;
    let q_ptr: u64 = q.as_usize() as u64;
    let k_ptr: u64 = k.as_usize() as u64;
    let v_ptr: u64 = v.as_usize() as u64;
    let o_ptr: u64 = out.as_usize() as u64;
    let m_ptr: u64 = partials_m.as_usize() as u64;
    let s_ptr: u64 = partials_s.as_usize() as u64;
    let po_ptr: u64 = partials_o.as_usize() as u64;
    let mut a1 = <B::Args<'_>>::default();
    a1.push_arg(&q_ptr);
    push_scalar_maybe_slot(&mut a1, &k_ptr, k_ptr_slot);
    push_scalar_maybe_slot(&mut a1, &v_ptr, v_ptr_slot);
    a1.push_arg(&m_ptr);
    a1.push_arg(&s_ptr);
    a1.push_arg(&po_ptr);
    a1.push_arg(&n_heads_q_i);
    a1.push_arg(&n_heads_kv_i);
    a1.push_arg(&head_dim_i);
    push_scalar_maybe_slot(&mut a1, &n_tokens_i, n_tokens_slot);
    a1.push_arg(&n_chunks_i);
    a1.push_arg(&chunk_size_i);
    a1.push_arg(&scale);
    a1.push_arg(&window_size);
    a1.push_arg(&ring_depth);
    push_scalar_maybe_slot(&mut a1, &chunk_base_i, chunk_base_slot);
    let cfg1 = LaunchCfg {
        grid: (n_heads_q as u32, n_chunks as u32, 1),
        block: (head_dim as u32, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&k_chunk, stream, cfg1, a1)?;

    let mut a2 = <B::Args<'_>>::default();
    a2.push_arg(&m_ptr);
    a2.push_arg(&s_ptr);
    a2.push_arg(&po_ptr);
    a2.push_arg(&o_ptr);
    a2.push_arg(&n_heads_q_i);
    a2.push_arg(&n_chunks_i);
    a2.push_arg(&head_dim_i);
    let cfg2 = LaunchCfg {
        grid: (n_heads_q as u32, 1, 1),
        block: (head_dim as u32, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&k_combine, stream, cfg2, a2)?;

    Ok(())
}

/// Batched-slots split-K (flash-decoding) decode attention, F16 KV. The
/// per-slot sibling of [`attention_decode_f16_splitk`]: grid.z fans each slot's
/// KV into `n_chunks`, then a combine pass merges per (q_head, slot). CUDA-only
/// (the kernel lives in kernels-cuda); on a backend without the module (gfx906)
/// this falls back to the serial [`attention_decode_f16_batched`]. `n_chunks`
/// derives from the host-side `max_n_tokens_kv`; per-slot device lengths clamp
/// each chunk so shorter slots emit empty partials.
///
/// Partials (caller-provided, f32): `partials_m`/`partials_s` at
/// `n_heads_q * n_slots * n_chunks`, `partials_o` at `* head_dim`.
pub fn attention_decode_f16_batched_splitk<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::AttnBatchedBuffers,
    partials: crate::AttnSplitkPartials,
    shape: crate::AttnDecodeBatchedSplitkShape,
    knobs: crate::AttnKnobs,
) -> Result<()> {
    let crate::AttnDecodeBatchedSplitkShape {
        n_heads_q,
        n_heads_kv,
        head_dim,
        n_slots,
        max_n_tokens_kv,
        chunk_size,
    } = shape;
    // gfx906 has no split-K batched module — keep the serial batched kernel.
    if reg.module("attention_decode_batched_splitk").is_none() {
        return attention_decode_f16_batched(
            reg,
            stream,
            buffers,
            crate::AttnDecodeBatchedShape { n_heads_q, n_heads_kv, head_dim, n_slots },
            knobs,
        );
    }
    assert!(
        head_dim == 64 || head_dim == 128 || head_dim == 256 || head_dim == 512,
        "attention_decode_f16_batched_splitk: head_dim {head_dim} not supported"
    );
    assert!((1..=32).contains(&n_slots));
    assert!(chunk_size > 0);

    let crate::AttnBatchedBuffers {
        q_batched,
        k_cache_ptrs,
        v_cache_ptrs,
        out_batched,
        n_tokens_kv_ptrs,
    } = buffers;
    let crate::AttnSplitkPartials { partials_m, partials_s, partials_o } = partials;
    let crate::AttnKnobs { scale, window_size, ring_depth } = knobs;

    let module = reg.expect_module("attention_decode_batched_splitk")?;
    let k_chunk = B::kernel(module, "flambeau_attention_decode_f16_batched_splitk_chunk")?;
    let k_combine = B::kernel(module, "flambeau_attention_decode_f16_batched_splitk_combine")?;

    let n_chunks = max_n_tokens_kv.div_ceil(chunk_size).max(1);
    let n_heads_q_i = n_heads_q as i32;
    let n_heads_kv_i = n_heads_kv as i32;
    let head_dim_i = head_dim as i32;
    let n_slots_i = n_slots as i32;
    let n_chunks_i = n_chunks as i32;
    let chunk_size_i = chunk_size as i32;
    let q_ptr: u64 = q_batched.as_usize() as u64;
    let k_ptrs_ptr: u64 = k_cache_ptrs.as_usize() as u64;
    let v_ptrs_ptr: u64 = v_cache_ptrs.as_usize() as u64;
    let o_ptr: u64 = out_batched.as_usize() as u64;
    let n_kv_ptr: u64 = n_tokens_kv_ptrs.as_usize() as u64;
    let m_ptr: u64 = partials_m.as_usize() as u64;
    let s_ptr: u64 = partials_s.as_usize() as u64;
    let po_ptr: u64 = partials_o.as_usize() as u64;

    let mut a1 = <B::Args<'_>>::default();
    a1.push_arg(&q_ptr);
    a1.push_arg(&k_ptrs_ptr);
    a1.push_arg(&v_ptrs_ptr);
    a1.push_arg(&m_ptr);
    a1.push_arg(&s_ptr);
    a1.push_arg(&po_ptr);
    a1.push_arg(&n_kv_ptr);
    a1.push_arg(&n_heads_q_i);
    a1.push_arg(&n_heads_kv_i);
    a1.push_arg(&head_dim_i);
    a1.push_arg(&n_slots_i);
    a1.push_arg(&n_chunks_i);
    a1.push_arg(&chunk_size_i);
    a1.push_arg(&scale);
    a1.push_arg(&window_size);
    a1.push_arg(&ring_depth);
    let cfg1 = LaunchCfg {
        grid: (n_heads_q as u32, n_slots as u32, n_chunks as u32),
        block: (head_dim as u32, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&k_chunk, stream, cfg1, a1)?;

    let mut a2 = <B::Args<'_>>::default();
    a2.push_arg(&m_ptr);
    a2.push_arg(&s_ptr);
    a2.push_arg(&po_ptr);
    a2.push_arg(&o_ptr);
    a2.push_arg(&n_heads_q_i);
    a2.push_arg(&n_slots_i);
    a2.push_arg(&n_chunks_i);
    a2.push_arg(&head_dim_i);
    let cfg2 = LaunchCfg {
        grid: (n_heads_q as u32, n_slots as u32, 1),
        block: (head_dim as u32, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&k_combine, stream, cfg2, a2)?;
    Ok(())
}

/// Device-position split-K decode attention, F16 KV. Behaviour matches
/// [`attention_decode_f16_splitk`] for the position the device buffer
/// `pos_ptr` holds, but reads `position` on-device and derives `n_tokens_kv` /
/// `chunk_base` / the K-V cache pointer from the baked [`crate::AttnDposKnobs`]
/// regime — so a captured decode graph rebinds nothing per token (one 8-byte
/// HtoD of the shared `[position, token_id]` context suffices). The chunk kernel
/// is CUDA-only (`attention_decode_f16_splitk_dpos`); the combine pass is
/// position-free and reuses the shared `attention_decode_f16_splitk` module.
/// `q`/`k`/`v` are the UNSLID base pointers (the kernel slides in-band).
pub fn attention_decode_f16_splitk_dpos<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::AttnBuffers,
    partials: crate::AttnSplitkPartials,
    shape: crate::AttnSplitkDposShape,
    knobs: crate::AttnDposKnobs,
    pos_ptr: flambeau_core::DevicePtr,
) -> Result<()> {
    let crate::AttnBuffers { q, k, v, out } = buffers;
    let crate::AttnSplitkPartials { partials_m, partials_s, partials_o } = partials;
    let crate::AttnSplitkDposShape { n_heads_q, n_heads_kv, head_dim, n_chunks, chunk_size } = shape;
    let crate::AttnDposKnobs { scale, window, ring_depth, slide_row_stride } = knobs;
    assert!(
        matches!(head_dim, 64 | 128 | 256 | 512),
        "attention_decode_f16_splitk_dpos: head_dim {head_dim} not supported"
    );
    assert!(chunk_size > 0 && n_chunks > 0);

    let module = reg.expect_module("attention_decode_f16_splitk_dpos")?;
    let k_chunk = B::kernel(module, "flambeau_attention_decode_f16_splitk_dpos_chunk")?;
    let combine_module = reg.expect_module("attention_decode_f16_splitk")?;
    let k_combine = B::kernel(combine_module, "flambeau_attention_decode_f16_splitk_combine")?;

    let n_heads_q_i = n_heads_q as i32;
    let n_heads_kv_i = n_heads_kv as i32;
    let head_dim_i = head_dim as i32;
    let n_chunks_i = n_chunks as i32;
    let chunk_size_i = chunk_size as i32;
    let q_ptr: u64 = q.as_usize() as u64;
    let k_ptr: u64 = k.as_usize() as u64;
    let v_ptr: u64 = v.as_usize() as u64;
    let o_ptr: u64 = out.as_usize() as u64;
    let m_ptr: u64 = partials_m.as_usize() as u64;
    let s_ptr: u64 = partials_s.as_usize() as u64;
    let po_ptr: u64 = partials_o.as_usize() as u64;
    let pos_p: u64 = pos_ptr.as_usize() as u64;

    let mut a1 = <B::Args<'_>>::default();
    a1.push_arg(&q_ptr);
    a1.push_arg(&k_ptr);
    a1.push_arg(&v_ptr);
    a1.push_arg(&m_ptr);
    a1.push_arg(&s_ptr);
    a1.push_arg(&po_ptr);
    a1.push_arg(&n_heads_q_i);
    a1.push_arg(&n_heads_kv_i);
    a1.push_arg(&head_dim_i);
    a1.push_arg(&n_chunks_i);
    a1.push_arg(&chunk_size_i);
    a1.push_arg(&scale);
    a1.push_arg(&window);
    a1.push_arg(&ring_depth);
    a1.push_arg(&slide_row_stride);
    a1.push_arg(&pos_p);
    let cfg1 = LaunchCfg {
        grid: (n_heads_q as u32, n_chunks as u32, 1),
        block: (head_dim as u32, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&k_chunk, stream, cfg1, a1)?;

    let mut a2 = <B::Args<'_>>::default();
    a2.push_arg(&m_ptr);
    a2.push_arg(&s_ptr);
    a2.push_arg(&po_ptr);
    a2.push_arg(&o_ptr);
    a2.push_arg(&n_heads_q_i);
    a2.push_arg(&n_chunks_i);
    a2.push_arg(&head_dim_i);
    let cfg2 = LaunchCfg {
        grid: (n_heads_q as u32, 1, 1),
        block: (head_dim as u32, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&k_combine, stream, cfg2, a2)?;
    Ok(())
}

/// Device-position split-K decode attention, Q8_0 KV. Q8 sibling of
/// [`attention_decode_f16_splitk_dpos`]: reads `position` on-device from
/// `pos_ptr` and derives `n_tokens_kv` / `chunk_base` / the slid K-V cache
/// pointer from the baked [`crate::AttnDposKnobs`] regime, so a captured decode
/// graph rebinds nothing per token. The chunk kernel is CUDA-only
/// (`attention_decode_q8_kv_splitk_dpos`); the position-free combine reuses the
/// `attention_decode_q8_kv_splitk` module. `q`/`k`/`v` are the UNSLID base
/// pointers (the kernel slides in-band using the Q8 row stride in bytes).
pub fn attention_decode_q8_kv_splitk_dpos<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::AttnBuffers,
    partials: crate::AttnSplitkPartials,
    shape: crate::AttnSplitkDposShape,
    knobs: crate::AttnDposKnobs,
    pos_ptr: flambeau_core::DevicePtr,
) -> Result<()> {
    let crate::AttnBuffers { q, k, v, out } = buffers;
    let crate::AttnSplitkPartials { partials_m, partials_s, partials_o } = partials;
    let crate::AttnSplitkDposShape { n_heads_q, n_heads_kv, head_dim, n_chunks, chunk_size } = shape;
    let crate::AttnDposKnobs { scale, window, ring_depth, slide_row_stride } = knobs;
    assert!(
        matches!(head_dim, 64 | 128 | 256 | 512),
        "attention_decode_q8_kv_splitk_dpos: head_dim {head_dim} not supported"
    );
    assert!(chunk_size > 0 && n_chunks > 0);

    let module = reg.expect_module("attention_decode_q8_kv_splitk_dpos")?;
    let k_chunk = B::kernel(module, "flambeau_attention_decode_q8_kv_splitk_dpos_chunk")?;
    let combine_module = reg.expect_module("attention_decode_q8_kv_splitk")?;
    let k_combine = B::kernel(combine_module, "flambeau_attention_decode_q8_kv_splitk_combine")?;

    let n_heads_q_i = n_heads_q as i32;
    let n_heads_kv_i = n_heads_kv as i32;
    let head_dim_i = head_dim as i32;
    let n_chunks_i = n_chunks as i32;
    let chunk_size_i = chunk_size as i32;
    let q_ptr: u64 = q.as_usize() as u64;
    let k_ptr: u64 = k.as_usize() as u64;
    let v_ptr: u64 = v.as_usize() as u64;
    let o_ptr: u64 = out.as_usize() as u64;
    let m_ptr: u64 = partials_m.as_usize() as u64;
    let s_ptr: u64 = partials_s.as_usize() as u64;
    let po_ptr: u64 = partials_o.as_usize() as u64;
    let pos_p: u64 = pos_ptr.as_usize() as u64;

    let mut a1 = <B::Args<'_>>::default();
    a1.push_arg(&q_ptr);
    a1.push_arg(&k_ptr);
    a1.push_arg(&v_ptr);
    a1.push_arg(&m_ptr);
    a1.push_arg(&s_ptr);
    a1.push_arg(&po_ptr);
    a1.push_arg(&n_heads_q_i);
    a1.push_arg(&n_heads_kv_i);
    a1.push_arg(&head_dim_i);
    a1.push_arg(&n_chunks_i);
    a1.push_arg(&chunk_size_i);
    a1.push_arg(&scale);
    a1.push_arg(&window);
    a1.push_arg(&ring_depth);
    a1.push_arg(&slide_row_stride);
    a1.push_arg(&pos_p);
    // Chunk pass: block = head_dim/4 (one thread per int32-packed Q8 quad).
    let cfg1 = LaunchCfg {
        grid: (n_heads_q as u32, n_chunks as u32, 1),
        block: ((head_dim / 4) as u32, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&k_chunk, stream, cfg1, a1)?;

    let mut a2 = <B::Args<'_>>::default();
    a2.push_arg(&m_ptr);
    a2.push_arg(&s_ptr);
    a2.push_arg(&po_ptr);
    a2.push_arg(&o_ptr);
    a2.push_arg(&n_heads_q_i);
    a2.push_arg(&n_chunks_i);
    a2.push_arg(&head_dim_i);
    let cfg2 = LaunchCfg {
        grid: (n_heads_q as u32, 1, 1),
        block: (head_dim as u32, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&k_combine, stream, cfg2, a2)?;
    Ok(())
}

/// KV parts (gridDim.y) of `attention_decode_q8_kv_vec_dpos` per Q head:
/// about 4 blocks per SM of an 82-SM sm_86 for 24 Q heads. One baked grid
/// for every position: x1.65 to x2.7 over the split-K path from a 181- to an
/// 8192-token tail (g5b microbench).
pub const ATTN_VEC_DPOS_PARTS: usize = 14;

/// Device-position Q8_0-KV decode attention with 128-thread blocks
/// (`attention_decode_q8_kv_vec_dpos`, CUDA-only): grid (n_heads_q, n_par).
/// With `n_par == 1` the chunk pass writes `out`; above, it writes split-K
/// partials and the position-free `attention_decode_q8_kv_splitk` combine
/// reduces them.
pub fn attention_decode_q8_kv_vec_dpos<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::AttnBuffers,
    partials: crate::AttnSplitkPartials,
    shape: crate::AttnVecDposShape,
    scale: f32,
    pos_ptr: flambeau_core::DevicePtr,
) -> Result<()> {
    let crate::AttnBuffers { q, k, v, out } = buffers;
    let crate::AttnSplitkPartials { partials_m, partials_s, partials_o } = partials;
    let crate::AttnVecDposShape { n_heads_q, n_heads_kv, head_dim, n_par } = shape;
    let entry = match head_dim {
        128 => "flambeau_attention_decode_q8_kv_vec_dpos_d128",
        256 => "flambeau_attention_decode_q8_kv_vec_dpos_d256",
        _ => anyhow::bail!("attention_decode_q8_kv_vec_dpos: head_dim {head_dim} not supported"),
    };
    anyhow::ensure!(n_par > 0 && n_heads_kv > 0 && n_heads_q % n_heads_kv == 0);
    let module = reg.expect_module("attention_decode_q8_kv_vec_dpos")?;
    let kernel = B::kernel(module, entry)?;

    let q_ptr: u64 = q.as_usize() as u64;
    let k_ptr: u64 = k.as_usize() as u64;
    let v_ptr: u64 = v.as_usize() as u64;
    let o_ptr: u64 = out.as_usize() as u64;
    let m_ptr: u64 = partials_m.as_usize() as u64;
    let s_ptr: u64 = partials_s.as_usize() as u64;
    let po_ptr: u64 = partials_o.as_usize() as u64;
    let pos_p: u64 = pos_ptr.as_usize() as u64;
    let n_heads_q_i = n_heads_q as i32;
    let n_heads_kv_i = n_heads_kv as i32;
    let mut a = <B::Args<'_>>::default();
    a.push_arg(&q_ptr);
    a.push_arg(&k_ptr);
    a.push_arg(&v_ptr);
    a.push_arg(&o_ptr);
    a.push_arg(&m_ptr);
    a.push_arg(&s_ptr);
    a.push_arg(&po_ptr);
    a.push_arg(&n_heads_q_i);
    a.push_arg(&n_heads_kv_i);
    a.push_arg(&scale);
    a.push_arg(&pos_p);
    let cfg = LaunchCfg { grid: (n_heads_q as u32, n_par as u32, 1), block: (128, 1, 1), shared_bytes: 0 };
    B::launch(&kernel, stream, cfg, a)?;
    if n_par == 1 {
        return Ok(());
    }

    let combine_module = reg.expect_module("attention_decode_q8_kv_splitk")?;
    let combine = B::kernel(combine_module, "flambeau_attention_decode_q8_kv_splitk_combine")?;
    let n_par_i = n_par as i32;
    let head_dim_i = head_dim as i32;
    let mut a2 = <B::Args<'_>>::default();
    a2.push_arg(&m_ptr);
    a2.push_arg(&s_ptr);
    a2.push_arg(&po_ptr);
    a2.push_arg(&o_ptr);
    a2.push_arg(&n_heads_q_i);
    a2.push_arg(&n_par_i);
    a2.push_arg(&head_dim_i);
    let cfg2 = LaunchCfg { grid: (n_heads_q as u32, 1, 1), block: (head_dim as u32, 1, 1), shared_bytes: 0 };
    B::launch(&combine, stream, cfg2, a2)?;
    Ok(())
}

/// Half2-packed split-K decode attention, F16 KV. Identical math + partials
/// layout to `attention_decode_f16_splitk`; inner KQ dot and VKQ accumulate
/// use `__hmul2` so gfx906 issues `v_pk_mul_f16` (2 F16 mul/cycle vs the
/// scalar F32 FMA equivalent). Block size halved to `head_dim / 2` —
/// each thread handles a dim pair — which also boosts MI50 occupancy
/// from 1 → 2 blocks/CU at head_dim=256.
///
/// Same partials sizing as `attention_decode_f16_splitk`. Phase 2
/// (`flambeau_attention_decode_f16_splitk_combine`) is shared and
/// unmodified.
pub fn attention_decode_f16_splitk_h2<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::AttnBuffers,
    partials: crate::AttnSplitkPartials,
    shape: crate::AttnSplitkShape,
    knobs: crate::AttnKnobs,
) -> Result<()> {
    let crate::AttnBuffers { q, k, v, out } = buffers;
    let crate::AttnSplitkPartials {
        partials_m,
        partials_s,
        partials_o,
    } = partials;
    let crate::AttnSplitkShape {
        n_heads_q,
        n_heads_kv,
        head_dim,
        n_tokens_kv,
        chunk_size,
        // The h2 chunk kernel chunks from position 0; ring chunk_base is not
        // wired here (this variant is unused — kept for the head_dim=256 lever).
        chunk_base: _,
    } = shape;
    let crate::AttnKnobs { scale, window_size, ring_depth } = knobs;
    assert!(
        head_dim == 128 || head_dim == 256 || head_dim == 512,
        "attention_decode_f16_splitk_h2: head_dim {head_dim} not in {{128, 256, 512}}"
    );
    assert!(chunk_size > 0);
    assert!(head_dim % 2 == 0);

    let module = reg.expect_module("attention_decode_f16_splitk_h2")?;
    let k_chunk = B::kernel(module, "flambeau_attention_decode_f16_splitk_h2_chunk")?;
    let combine_module = reg.expect_module("attention_decode_f16_splitk")?;
    let k_combine = B::kernel(combine_module, "flambeau_attention_decode_f16_splitk_combine")?;

    let n_chunks = n_tokens_kv.div_ceil(chunk_size);
    let n_heads_q_i = n_heads_q as i32;
    let n_heads_kv_i = n_heads_kv as i32;
    let head_dim_i = head_dim as i32;
    let n_tokens_i = n_tokens_kv as i32;
    let n_chunks_i = n_chunks as i32;
    let chunk_size_i = chunk_size as i32;
    let q_ptr: u64 = q.as_usize() as u64;
    let k_ptr: u64 = k.as_usize() as u64;
    let v_ptr: u64 = v.as_usize() as u64;
    let o_ptr: u64 = out.as_usize() as u64;
    let m_ptr: u64 = partials_m.as_usize() as u64;
    let s_ptr: u64 = partials_s.as_usize() as u64;
    let po_ptr: u64 = partials_o.as_usize() as u64;

    let mut a1 = <B::Args<'_>>::default();
    a1.push_arg(&q_ptr);
    a1.push_arg(&k_ptr);
    a1.push_arg(&v_ptr);
    a1.push_arg(&m_ptr);
    a1.push_arg(&s_ptr);
    a1.push_arg(&po_ptr);
    a1.push_arg(&n_heads_q_i);
    a1.push_arg(&n_heads_kv_i);
    a1.push_arg(&head_dim_i);
    a1.push_arg(&n_tokens_i);
    a1.push_arg(&n_chunks_i);
    a1.push_arg(&chunk_size_i);
    a1.push_arg(&scale);
    a1.push_arg(&window_size);
    a1.push_arg(&ring_depth);
    let cfg1 = LaunchCfg {
        grid: (n_heads_q as u32, n_chunks as u32, 1),
        block: ((head_dim / 2) as u32, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&k_chunk, stream, cfg1, a1)?;

    let mut a2 = <B::Args<'_>>::default();
    a2.push_arg(&m_ptr);
    a2.push_arg(&s_ptr);
    a2.push_arg(&po_ptr);
    a2.push_arg(&o_ptr);
    a2.push_arg(&n_heads_q_i);
    a2.push_arg(&n_chunks_i);
    a2.push_arg(&head_dim_i);
    let cfg2 = LaunchCfg {
        grid: (n_heads_q as u32, 1, 1),
        block: (head_dim as u32, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&k_combine, stream, cfg2, a2)?;

    Ok(())
}

/// Decode contexts up to this many KV rows run the single-pass kernel. Longer
/// ones split K: the single-pass grid is one workgroup per Q head (12 on a
/// qwen35 27B TP rank), which leaves most CUs idle even at short context.
pub const SPLITK_MIN_TOKENS_KV: usize = 32;

/// Pick a split-K chunk size that partitions `n_tokens_kv` into at most
/// `max_chunks` chunks of >= 32 tokens each. The chunk kernel launches
/// `n_heads * n_slots * n_chunks` blocks, so the caller sizes `max_chunks` to
/// fill the device's block slots: single-slot decode passes
/// `MAX_SPLITK_CHUNKS` (n_heads * 32 blocks saturates the target GPUs at
/// n_heads=16); batched decode passes `MAX_SPLITK_CHUNKS / n_slots` so the
/// block count stays at that budget instead of over-subscribing by the slot
/// count. Returns the single-pass fallback `chunk_size = n_tokens_kv` up to
/// [`SPLITK_MIN_TOKENS_KV`] tokens.
pub fn splitk_chunk_size(n_tokens_kv: usize, max_chunks: usize) -> usize {
    if n_tokens_kv <= SPLITK_MIN_TOKENS_KV {
        return n_tokens_kv;
    }
    n_tokens_kv.div_ceil(max_chunks.max(1)).max(32).next_multiple_of(4)
}

/// Decode attention with Q8_0-quantised KV. Same args as the F16 variant;
/// `k_cache` / `v_cache` hold `flambeau_block_q8_0` blocks laid out as
/// `[n_tokens_kv, n_heads_kv, head_dim/32]` row-major.
pub fn attention_decode_q8_kv<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::AttnBuffers,
    shape: crate::AttnDecodeShape,
    knobs: crate::AttnKnobs,
) -> Result<()> {
    attention_decode_q8_kv_slots(reg, stream, buffers, shape, knobs, None)
}

/// Graph-captureable variant of [`attention_decode_q8_kv`]. Q8_0-KV sibling of
/// [`attention_decode_f16_slots`]: with `slots=None` behaviour is byte-
/// identical to the wrapper; with `Some`, the `n_tokens_kv` kernel arg is
/// tagged via `KernelArgs::push_slot` for per-replay rebind.
pub fn attention_decode_q8_kv_slots<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::AttnBuffers,
    shape: crate::AttnDecodeShape,
    knobs: crate::AttnKnobs,
    slots: Option<crate::AttnDecodeQ8Slots>,
) -> Result<()> {
    let crate::AttnBuffers { q, k, v, out } = buffers;
    let crate::AttnDecodeShape {
        n_heads_q,
        n_heads_kv,
        head_dim,
        n_tokens_kv,
    } = shape;
    let crate::AttnKnobs { scale, window_size, ring_depth } = knobs;
    // Kernel supports head_dim ∈ {64, 128, 256, 512}. block = head_dim/4
    // threads (16/32/64/128). At d=512 the block is 2 waves and the
    // sum-of-blocks reduction adds a tiny cross-wave LDS rendezvous
    // (`score_parts[2]` + 2 __syncthreads). SWA layers (window_size > 0)
    // are supported via the t_start clamp in the kernel inner loop.
    assert!(
        matches!(head_dim, 64 | 128 | 256 | 512),
        "attention_decode_q8_kv: head_dim {head_dim} not supported (expected 64, 128, 256, or 512)"
    );
    let module = reg.expect_module("attention_decode_q8_kv")?;
    let kernel = B::kernel(module, "flambeau_attention_decode_q8_kv")?;

    let n_heads_q_i = n_heads_q as i32;
    let n_heads_kv_i = n_heads_kv as i32;
    let head_dim_i = head_dim as i32;
    let n_tokens_i = n_tokens_kv as i32;
    let q_ptr: u64 = q.as_usize() as u64;
    let k_ptr: u64 = k.as_usize() as u64;
    let v_ptr: u64 = v.as_usize() as u64;
    let o_ptr: u64 = out.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&q_ptr);
    args.push_arg(&k_ptr);
    args.push_arg(&v_ptr);
    args.push_arg(&o_ptr);
    args.push_arg(&n_heads_q_i);
    args.push_arg(&n_heads_kv_i);
    args.push_arg(&head_dim_i);
    match slots {
        Some(s) => args.push_slot(&n_tokens_i, s.n_tokens_kv),
        None => args.push_arg(&n_tokens_i),
    }
    args.push_arg(&scale);
    args.push_arg(&window_size);
    args.push_arg(&ring_depth);
    // block.x = head_dim/4 (one thread per int32-
    // packed quad). For head_dim=256 that's 64 threads = 1 wavefront.
    let cfg = LaunchCfg::one_d(n_heads_q as u32, (head_dim / 4) as u32);
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Split-K (flash-decoding) decode attention with Q8_0 KV. Same shape as
/// [`attention_decode_f16_splitk`] but K and V come from a `KvCache<Q8Contig>`
/// — dequantised on the fly during the dot product and the V-accumulate.
/// Closes the long-context Q8↔F16 gap (single-pass `attention_decode_q8_kv`
/// is the same shape as the single-pass F16 kernel and pays the same 7.78×
/// occupancy penalty past 256 KV tokens).
pub fn attention_decode_q8_kv_splitk<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::AttnBuffers,
    partials: crate::AttnSplitkPartials,
    shape: crate::AttnSplitkShape,
    knobs: crate::AttnKnobs,
) -> Result<()> {
    attention_decode_q8_kv_splitk_slots(reg, stream, buffers, partials, shape, knobs, None)
}

/// Graph-captureable variant of [`attention_decode_q8_kv_splitk`]. Q8_0-KV
/// sibling of [`attention_decode_f16_splitk_slots`]: with `slots=None`
/// behaviour is byte-identical to the wrapper; with `Some`, the chunk kernel's
/// `n_tokens_kv` + `chunk_base` scalar args are tagged via
/// `KernelArgs::push_slot` for per-replay rebind while `n_chunks` is fixed.
pub fn attention_decode_q8_kv_splitk_slots<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::AttnBuffers,
    partials: crate::AttnSplitkPartials,
    shape: crate::AttnSplitkShape,
    knobs: crate::AttnKnobs,
    slots: Option<crate::AttnDecodeQ8SplitkSlots>,
) -> Result<()> {
    let (n_tokens_slot, chunk_base_slot) = match slots {
        Some(s) => (Some(s.n_tokens_kv), Some(s.chunk_base)),
        None => (None, None),
    };
    let crate::AttnBuffers { q, k, v, out } = buffers;
    let crate::AttnSplitkPartials {
        partials_m,
        partials_s,
        partials_o,
    } = partials;
    let crate::AttnSplitkShape {
        n_heads_q,
        n_heads_kv,
        head_dim,
        n_tokens_kv,
        chunk_size,
        chunk_base,
    } = shape;
    let crate::AttnKnobs { scale, window_size, ring_depth } = knobs;
    // head_dim ∈ {64, 128, 256, 512}. d=512 enables Q8 on gemma4
    // global layers; the kernel adds a cross-wave LDS reduce for that
    // case (see attention_decode_q8_kv comments).
    assert!(
        matches!(head_dim, 64 | 128 | 256 | 512),
        "attention_decode_q8_kv_splitk: head_dim {head_dim} not supported"
    );
    assert!(chunk_size > 0);
    assert!(chunk_base <= n_tokens_kv);

    let module = reg.expect_module("attention_decode_q8_kv_splitk")?;
    let k_chunk = B::kernel(module, "flambeau_attention_decode_q8_kv_splitk_chunk")?;
    let k_combine = B::kernel(module, "flambeau_attention_decode_q8_kv_splitk_combine")?;

    let n_chunks = (n_tokens_kv - chunk_base).div_ceil(chunk_size);
    let n_heads_q_i = n_heads_q as i32;
    let n_heads_kv_i = n_heads_kv as i32;
    let head_dim_i = head_dim as i32;
    let n_tokens_i = n_tokens_kv as i32;
    let n_chunks_i = n_chunks as i32;
    let chunk_size_i = chunk_size as i32;
    let chunk_base_i = chunk_base as i32;
    let q_ptr: u64 = q.as_usize() as u64;
    let k_ptr: u64 = k.as_usize() as u64;
    let v_ptr: u64 = v.as_usize() as u64;
    let o_ptr: u64 = out.as_usize() as u64;
    let m_ptr: u64 = partials_m.as_usize() as u64;
    let s_ptr: u64 = partials_s.as_usize() as u64;
    let po_ptr: u64 = partials_o.as_usize() as u64;

    let mut a1 = <B::Args<'_>>::default();
    a1.push_arg(&q_ptr);
    a1.push_arg(&k_ptr);
    a1.push_arg(&v_ptr);
    a1.push_arg(&m_ptr);
    a1.push_arg(&s_ptr);
    a1.push_arg(&po_ptr);
    a1.push_arg(&n_heads_q_i);
    a1.push_arg(&n_heads_kv_i);
    a1.push_arg(&head_dim_i);
    push_scalar_maybe_slot(&mut a1, &n_tokens_i, n_tokens_slot);
    a1.push_arg(&n_chunks_i);
    a1.push_arg(&chunk_size_i);
    a1.push_arg(&scale);
    a1.push_arg(&window_size);
    a1.push_arg(&ring_depth);
    push_scalar_maybe_slot(&mut a1, &chunk_base_i, chunk_base_slot);
    // chunk pass uses block = head_dim/4 (one
    // thread per int32-packed quad). Combine pass still needs
    // head_dim threads (one per output element).
    let cfg1 = LaunchCfg {
        grid: (n_heads_q as u32, n_chunks as u32, 1),
        block: ((head_dim / 4) as u32, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&k_chunk, stream, cfg1, a1)?;

    let mut a2 = <B::Args<'_>>::default();
    a2.push_arg(&m_ptr);
    a2.push_arg(&s_ptr);
    a2.push_arg(&po_ptr);
    a2.push_arg(&o_ptr);
    a2.push_arg(&n_heads_q_i);
    a2.push_arg(&n_chunks_i);
    a2.push_arg(&head_dim_i);
    let cfg2 = LaunchCfg {
        grid: (n_heads_q as u32, 1, 1),
        block: (head_dim as u32, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&k_combine, stream, cfg2, a2)?;

    Ok(())
}

/// Prefill attention with Q8_0 KV cache. Dispatches on `n_q_tokens`:
/// - `n_q_tokens >= 4` → BR-tiled flash-tile kernel (LDS K/V tile reuse,
///   mirrors the F16 flash-tile path; cooperative load dequants Q8 → F32
///   into LDS, score loop is identical to F16 flash-tile after that).
/// - `n_q_tokens < 4` → oracle kernel (one block per `(q_token, q_head)`,
///   dp4a score loop, V FP-dequant per element).
///   closes the prefill regression where Q8
///   was stuck on the oracle path (~0.90× F16). Used by the batched-Q8-
///   prefill driver — replaces the per-token Q8 prefill fallback with
///   one launch per layer per ubatch chunk.
pub fn attention_prefill_q8_kv<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::AttnBuffers,
    shape: crate::AttnPrefillShape,
    knobs: crate::AttnKnobs,
) -> Result<()> {
    let crate::AttnBuffers { q, k, v, out } = buffers;
    let crate::AttnPrefillShape {
        n_q_tokens,
        n_heads_q,
        n_heads_kv,
        head_dim,
        n_k_tokens,
        q_offset,
    } = shape;
    let crate::AttnKnobs { scale, window_size, ring_depth } = knobs;
    // head_dim ∈ {64, 128, 256, 512}. d=512 routes through the oracle
    // single-pass kernel (no flash_tile template at d=512); d≤256 goes
    // flash_tile when n_q_tokens ≥ 4.
    assert!(
        matches!(head_dim, 64 | 128 | 256 | 512),
        "attention_prefill_q8_kv: head_dim {head_dim} not supported"
    );

    let n_q_i = n_q_tokens as i32;
    let n_heads_q_i = n_heads_q as i32;
    let n_heads_kv_i = n_heads_kv as i32;
    let n_k_i = n_k_tokens as i32;
    let q_off_i = q_offset as i32;
    let q_ptr: u64 = q.as_usize() as u64;
    let k_ptr: u64 = k.as_usize() as u64;
    let v_ptr: u64 = v.as_usize() as u64;
    let o_ptr: u64 = out.as_usize() as u64;

    if n_q_tokens >= 4 && head_dim != 512 {
        // Flash-tile fast path: BR=4 (head_dim ∈ {64,128}) or BR=8
        // (head_dim=256, more Q rows / fewer blocks at high n_q).
        // d=512 has no flash_tile template (template instantiation
        // would push LDS tile size to 32 KB at BC=16); falls back to
        // the oracle single-pass kernel below, which now handles
        // d=512 via cross-wave LDS reduction.
        let module = reg.expect_module("attention_prefill_flash_tile_q8_kv")?;
        let entry = match head_dim {
            64 => "flambeau_attention_prefill_flash_tile_d64_q8_kv",
            128 => "flambeau_attention_prefill_flash_tile_d128_q8_kv",
            256 => "flambeau_attention_prefill_flash_tile_d256_br8_q8_kv",
            _ => unreachable!(),
        };
        let kernel = B::kernel(module, entry)?;
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&q_ptr);
        args.push_arg(&k_ptr);
        args.push_arg(&v_ptr);
        args.push_arg(&o_ptr);
        args.push_arg(&n_q_i);
        args.push_arg(&n_heads_q_i);
        args.push_arg(&n_heads_kv_i);
        args.push_arg(&n_k_i);
        args.push_arg(&q_off_i);
        args.push_arg(&scale);
        args.push_arg(&window_size);
        args.push_arg(&ring_depth);
        let br: u32 = if head_dim == 256 { 8 } else { 4 };
        let cfg = LaunchCfg {
            grid: ((n_q_tokens as u32).div_ceil(br), n_heads_q as u32, 1),
            block: (B::WARP_SIZE, br, 1),
            shared_bytes: 0,
        };
        B::launch(&kernel, stream, cfg, args)?;
        return Ok(());
    }

    // Oracle path for n_q < 4 (very short prompts / edge shapes).
    let module = reg.expect_module("attention_prefill_q8_kv")?;
    let kernel = B::kernel(module, "flambeau_attention_prefill_q8_kv")?;
    let head_dim_i = head_dim as i32;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&q_ptr);
    args.push_arg(&k_ptr);
    args.push_arg(&v_ptr);
    args.push_arg(&o_ptr);
    args.push_arg(&n_q_i);
    args.push_arg(&n_heads_q_i);
    args.push_arg(&n_heads_kv_i);
    args.push_arg(&head_dim_i);
    args.push_arg(&n_k_i);
    args.push_arg(&q_off_i);
    args.push_arg(&scale);
    args.push_arg(&window_size);
    args.push_arg(&ring_depth);
    let cfg = LaunchCfg {
        grid: (n_q_tokens as u32, n_heads_q as u32, 1),
        block: ((head_dim / 4) as u32, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Split the interleaved `(Q, gate)` output of a gated-attention query
/// projection into two contiguous F16 tensors. Used by Qwen3.5/3.6 full-attn
/// layers, where the `attn_q` weight projects to `2 * head_dim` per head —
/// first half is Q, second half is the output gate. Q goes into the
/// attention kernel; gate is held for a silu-multiply after attention.
/// Launch: `(n_tokens, n_heads, ceil(head_dim/128))` × 128 threads.
pub fn split_q_gate_f16<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    bufs: crate::SplitQGateBuffers,
    shape: crate::SplitQGateShape,
) -> Result<()> {
    let crate::SplitQGateBuffers { fused_qg, q_out, gate_out } = bufs;
    let crate::SplitQGateShape { n_tokens, n_heads, head_dim } = shape;
    let module = reg.expect_module("split_q_gate_f16")?;
    let kernel = B::kernel(module, "flambeau_split_q_gate_f16")?;

    let n_tokens_i = n_tokens as i32;
    let n_heads_i = n_heads as i32;
    let head_dim_i = head_dim as i32;
    let f_ptr: u64 = fused_qg.as_usize() as u64;
    let q_ptr: u64 = q_out.as_usize() as u64;
    let g_ptr: u64 = gate_out.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&f_ptr);
    args.push_arg(&q_ptr);
    args.push_arg(&g_ptr);
    args.push_arg(&n_tokens_i);
    args.push_arg(&n_heads_i);
    args.push_arg(&head_dim_i);
    let threads = 128u32;
    let grid_z = (head_dim as u32).div_ceil(threads);
    let cfg = LaunchCfg {
        grid: (n_tokens as u32, n_heads as u32, grid_z),
        block: (threads, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Prefill attention, F16 KV. Computes `n_q_tokens` Q rows against
/// `n_k_tokens` KV rows with causal masking (`q_token_i` attends to
/// `k_token_0..k_token_{q_offset + i}`).
/// Dispatches:
/// - `attention_prefill_flash_tile_f16` (BR=4 LDS-tiled flash-attn v2)
///   when `n_q_tokens >= 4` and head_dim ∈ {64, 128, 256}.
/// - The per-(q_token, q_head) oracle kernel (`attention_prefill_f16`)
///   otherwise. Shape grid covers the n_q < 4 edge that flash-tile's
///   BR=4 coop-load pattern under-utilises.
pub fn attention_prefill_f16<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::AttnBuffers,
    shape: crate::AttnPrefillShape,
    knobs: crate::AttnKnobs,
) -> Result<()> {
    attention_prefill_f16_slots(reg, stream, buffers, shape, knobs, None)
}

/// Graph-captureable variant of [`attention_prefill_f16`].
/// Behaviour matches the wrapper in every non-capture case. When the
/// optional [`ScalarSlot`] handles are `Some`, `push_slot` tags their
/// kernel args so the graph recorder can bind the slots to the
/// resulting kernel node. Callers can then replay the captured exec at
/// a different `start_position` by calling `HipGraphExec::set_slot`.
/// The relevant pos-varying scalars are:
/// - `n_k_tokens` — total K tokens the causal mask stops at (grows each
///   ubatch as the KV cache fills).
/// - `q_offset` — row offset of the Q block into the global causal grid
///   (equals `start_position`).
///   Both kernels (flash_tile + oracle) place these at different arg
///   indices; the slot recorder captures whichever path n_q_tokens
///   selected. Capturing at a particular n_q and replaying at a different
///   n_q is unsupported (different path → different node layout).
///   [`ScalarSlot`]: flambeau_core::ScalarSlot
pub fn attention_prefill_f16_slots<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buffers: crate::AttnBuffers,
    shape: crate::AttnPrefillShape,
    knobs: crate::AttnKnobs,
    slots: Option<crate::AttnPrefillSlots>,
) -> Result<()> {
    let crate::AttnBuffers { q, k, v, out } = buffers;
    let crate::AttnPrefillShape {
        n_q_tokens,
        n_heads_q,
        n_heads_kv,
        head_dim,
        n_k_tokens,
        q_offset,
    } = shape;
    let crate::AttnKnobs { scale, window_size, ring_depth } = knobs;
    let (n_k_slot, q_off_slot) = match slots {
        Some(s) => (Some(s.n_k), Some(s.q_off)),
        None => (None, None),
    };
    assert!(
        head_dim == 64 || head_dim == 128 || head_dim == 256 || head_dim == 512,
        "attention_prefill_f16: head_dim {head_dim} not supported (expected 64, 128, 256, or 512)"
    );

    // flash_tile prefill now SWA-aware (window_size kernel arg lands
    // alongside causal + per-warp swa_min masking). Route through it
    // whenever n_q ≥ 4.
    let use_flash_tile = n_q_tokens >= 4;

    let q_ptr: u64 = q.as_usize() as u64;
    let k_ptr: u64 = k.as_usize() as u64;
    let v_ptr: u64 = v.as_usize() as u64;
    let o_ptr: u64 = out.as_usize() as u64;
    let n_q_i = n_q_tokens as i32;
    let n_heads_q_i = n_heads_q as i32;
    let n_heads_kv_i = n_heads_kv as i32;
    let n_k_i = n_k_tokens as i32;
    let q_off_i = q_offset as i32;

    // Tensor-core (f16 mma) prefill — CUDA-only. Present only when the backend
    // compiled `attention_prefill_mma` (gfx906 has no matrix core, so the module
    // is absent there and this branch is skipped). NW=8 → BR=128 q rows/block; the
    // kernel bounds-clamps a partial final tile, so any n_q ≥ 16 is fine. d512
    // (gemma global layers) uses the DV-split variant; head_dims outside
    // {128,256,512} fall through to flash_tile.
    // Tensor-core prefill when the backend's [[attention_prefill]] table picks
    // it for this width (sm_86 from 16 query rows) at a head_dim it serves.
    let mma_row = matches!(head_dim, 128 | 256 | 512)
        && B::dispatch_attention_prefill(flambeau_core::QDtype::F16, n_q_tokens)
            .is_some_and(|row| row.impl_id == "attention_prefill_mma");
    if mma_row {
        let module = reg.expect_module("attention_prefill_mma")?;
        let entry = match head_dim {
            128 => "flambeau_attention_prefill_mma_d128_f16",
            256 => "flambeau_attention_prefill_mma_d256_f16",
            512 => "flambeau_attention_prefill_mma_d512_f16",
            _ => unreachable!(),
        };
        let kernel = B::kernel(module, entry)?;
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&q_ptr);
        args.push_arg(&k_ptr);
        args.push_arg(&v_ptr);
        args.push_arg(&o_ptr);
        args.push_arg(&n_q_i);
        args.push_arg(&n_heads_q_i);
        args.push_arg(&n_heads_kv_i);
        push_scalar_maybe_slot(&mut args, &n_k_i, n_k_slot);
        push_scalar_maybe_slot(&mut args, &q_off_i, q_off_slot);
        args.push_arg(&scale);
        args.push_arg(&window_size);
        args.push_arg(&ring_depth);
        let cfg = LaunchCfg {
            grid: ((n_q_tokens as u32).div_ceil(128), n_heads_q as u32, 1),
            block: (32, 8, 1),
            shared_bytes: 0,
        };
        B::launch(&kernel, stream, cfg, args)?;
        return Ok(());
    }

    if use_flash_tile {
        // BR=4 LDS-tiled kernel.
        let module = reg.expect_module("attention_prefill_flash_tile_f16")?;
        let entry = match head_dim {
            64 => "flambeau_attention_prefill_flash_tile_d64_f16",
            128 => "flambeau_attention_prefill_flash_tile_d128_f16",
            256 => "flambeau_attention_prefill_flash_tile_d256_br8_f16",
            512 => "flambeau_attention_prefill_flash_tile_d512_f16",
            _ => unreachable!(),
        };
        let kernel = B::kernel(module, entry)?;
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&q_ptr);
        args.push_arg(&k_ptr);
        args.push_arg(&v_ptr);
        args.push_arg(&o_ptr);
        args.push_arg(&n_q_i);
        args.push_arg(&n_heads_q_i);
        args.push_arg(&n_heads_kv_i);
        push_scalar_maybe_slot(&mut args, &n_k_i, n_k_slot);
        push_scalar_maybe_slot(&mut args, &q_off_i, q_off_slot);
        args.push_arg(&scale);
        args.push_arg(&window_size);
        args.push_arg(&ring_depth);
        // BR depends on which variant we dispatch to.
        // d256_br8 uses BR=8 (more Q rows per block, fewer blocks);
        // other head_dims still use BR=4.
        let br: u32 = if head_dim == 256 { 8 } else { 4 };
        let cfg = LaunchCfg {
            grid: ((n_q_tokens as u32).div_ceil(br), n_heads_q as u32, 1),
            block: (B::WARP_SIZE, br, 1),
            shared_bytes: 0,
        };
        B::launch(&kernel, stream, cfg, args)?;
        return Ok(());
    }

    // Oracle path for n_q < 4 (very short prompts / edge shapes) or
    // any SWA call (flash_tile is not SWA-aware yet).
    let module = reg.expect_module("attention_prefill_f16")?;
    let kernel = B::kernel(module, "flambeau_attention_prefill_f16")?;
    let head_dim_i = head_dim as i32;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&q_ptr);
    args.push_arg(&k_ptr);
    args.push_arg(&v_ptr);
    args.push_arg(&o_ptr);
    args.push_arg(&n_q_i);
    args.push_arg(&n_heads_q_i);
    args.push_arg(&n_heads_kv_i);
    args.push_arg(&head_dim_i);
    push_scalar_maybe_slot(&mut args, &n_k_i, n_k_slot);
    push_scalar_maybe_slot(&mut args, &q_off_i, q_off_slot);
    args.push_arg(&scale);
    args.push_arg(&window_size);
    args.push_arg(&ring_depth);
    let cfg = LaunchCfg {
        grid: (n_q_tokens as u32, n_heads_q as u32, 1),
        block: (head_dim as u32, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

#[inline]
fn push_scalar_maybe_slot<'a, A: LaunchArgs<'a>, T: 'a>(
    args: &mut A,
    v: &'a T,
    slot: Option<flambeau_core::ScalarSlot>,
) {
    match slot {
        Some(s) => args.push_slot(v, s),
        None => args.push_arg(v),
    }
}

/// Fused KV-cache append with V unit-RMSNorm. For gemma4 archs that
/// apply RMSNorm V (unit weights) before the cache write. Saves
/// (rmsnorm_f16 + DtoD memcpy back + 2× DtoD memcpy kv_append) → 1
/// kernel launch per layer per token.
///
/// `k_src` / `v_src`: F16 [n_tokens, n_kv_heads, head_dim] in scratch.
/// `k_cache` / `v_cache`: F16 [max_seq, n_kv_heads, head_dim] slot.
/// `write_pos`: starting row offset within the slot.
/// `head_dim` ∈ {64, 128, 256, 512}.
pub fn kv_append_v_unit_norm_f16<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buf: crate::KvAppendBuffers,
    shape: crate::KvAppendVUnitShape,
    write_pos: usize,
    eps: f32,
) -> Result<()> {
    kv_append_v_unit_norm_f16_slots(reg, stream, buf, shape, write_pos, eps, None)
}

/// Graph-captureable variant of [`kv_append_v_unit_norm_f16`]. With
/// `slots=None` behaviour is byte-identical to the wrapper; with `Some`, the
/// `write_pos` cache-row offset (a scalar kernel arg) is tagged via
/// `KernelArgs::push_slot` so a captured graph advances the write cursor per
/// replay.
pub fn kv_append_v_unit_norm_f16_slots<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buf: crate::KvAppendBuffers,
    shape: crate::KvAppendVUnitShape,
    write_pos: usize,
    eps: f32,
    slots: Option<crate::KvAppendSlots>,
) -> Result<()> {
    let entry = match shape.head_dim {
        64 => "flambeau_kv_append_v_unit_norm_f16_d64",
        128 => "flambeau_kv_append_v_unit_norm_f16_d128",
        256 => "flambeau_kv_append_v_unit_norm_f16_d256",
        512 => "flambeau_kv_append_v_unit_norm_f16_d512",
        other => anyhow::bail!(
            "kv_append_v_unit_norm_f16: head_dim {other} not in {{64, 128, 256, 512}}"
        ),
    };
    let module = reg.expect_module("kv_append_v_unit_norm_f16")?;
    let kernel = B::kernel(module, entry)?;

    let n_kv_heads_i = shape.n_kv_heads as i32;
    let write_pos_i = write_pos as i32;
    let ring_depth_i = shape.ring_depth as i32;
    let k_src_p: u64 = buf.k_src.as_usize() as u64;
    let v_src_p: u64 = buf.v_src.as_usize() as u64;
    let k_dst_p: u64 = buf.k_dst.as_usize() as u64;
    let v_dst_p: u64 = buf.v_dst.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&k_src_p);
    args.push_arg(&v_src_p);
    args.push_arg(&k_dst_p);
    args.push_arg(&v_dst_p);
    args.push_arg(&n_kv_heads_i);
    match slots {
        Some(s) => args.push_slot(&write_pos_i, s.write_pos),
        None => args.push_arg(&write_pos_i),
    }
    args.push_arg(&eps);
    args.push_arg(&ring_depth_i);
    let cfg = LaunchCfg {
        grid: (shape.n_tokens as u32, shape.n_kv_heads as u32, 1),
        block: (shape.head_dim as u32, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Device-position variant of [`kv_append_v_unit_norm_f16`]. The write row is
/// read on-device from `pos_ptr[0]` instead of a host `write_pos`
/// scalar, so a captured decode graph advances the KV write cursor with no
/// rebind. CUDA-only kernel (`kv_append_v_unit_norm_f16_dpos`); `shape.ring_depth`
/// and `eps` stay baked constants.
pub fn kv_append_v_unit_norm_f16_dpos<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buf: crate::KvAppendBuffers,
    shape: crate::KvAppendVUnitShape,
    eps: f32,
    pos_ptr: flambeau_core::DevicePtr,
) -> Result<()> {
    let entry = match shape.head_dim {
        64 => "flambeau_kv_append_v_unit_norm_f16_dpos_d64",
        128 => "flambeau_kv_append_v_unit_norm_f16_dpos_d128",
        256 => "flambeau_kv_append_v_unit_norm_f16_dpos_d256",
        512 => "flambeau_kv_append_v_unit_norm_f16_dpos_d512",
        other => anyhow::bail!(
            "kv_append_v_unit_norm_f16_dpos: head_dim {other} not in {{64, 128, 256, 512}}"
        ),
    };
    let module = reg.expect_module("kv_append_v_unit_norm_f16_dpos")?;
    let kernel = B::kernel(module, entry)?;

    let n_kv_heads_i = shape.n_kv_heads as i32;
    let ring_depth_i = shape.ring_depth as i32;
    let k_src_p: u64 = buf.k_src.as_usize() as u64;
    let v_src_p: u64 = buf.v_src.as_usize() as u64;
    let k_dst_p: u64 = buf.k_dst.as_usize() as u64;
    let v_dst_p: u64 = buf.v_dst.as_usize() as u64;
    let pos_p: u64 = pos_ptr.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&k_src_p);
    args.push_arg(&v_src_p);
    args.push_arg(&k_dst_p);
    args.push_arg(&v_dst_p);
    args.push_arg(&n_kv_heads_i);
    args.push_arg(&eps);
    args.push_arg(&ring_depth_i);
    args.push_arg(&pos_p);
    let cfg = LaunchCfg {
        grid: (shape.n_tokens as u32, shape.n_kv_heads as u32, 1),
        block: (shape.head_dim as u32, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Device-position F16 KV append: one K and one V row of `kv_width` written at
/// row `ring_depth > 0 ? position % ring_depth : position` of the unslid slot
/// bases, `position` read on-device from `pos_ptr[0]`. CUDA-only kernel
/// (`kv_append_f16_dpos`).
pub fn kv_append_f16_dpos<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buf: crate::KvAppendBuffers,
    kv_width: usize,
    ring_depth: usize,
    pos_ptr: flambeau_core::DevicePtr,
) -> Result<()> {
    let module = reg.expect_module("kv_append_f16_dpos")?;
    let kernel = B::kernel(module, "flambeau_kv_append_f16_dpos")?;
    let kv_width_i = kv_width as i32;
    let ring_depth_i = ring_depth as i32;
    let k_src_p: u64 = buf.k_src.as_usize() as u64;
    let v_src_p: u64 = buf.v_src.as_usize() as u64;
    let k_dst_p: u64 = buf.k_dst.as_usize() as u64;
    let v_dst_p: u64 = buf.v_dst.as_usize() as u64;
    let pos_p: u64 = pos_ptr.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&k_src_p);
    args.push_arg(&v_src_p);
    args.push_arg(&k_dst_p);
    args.push_arg(&v_dst_p);
    args.push_arg(&kv_width_i);
    args.push_arg(&ring_depth_i);
    args.push_arg(&pos_p);
    let cfg = LaunchCfg {
        grid: (kv_width.div_ceil(256) as u32, 1, 1),
        block: (256, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Device-position embedding-row gather, F16. Copies one hidden-width
/// row `embd[token_id * hidden + j]` → `dst[j]`, reading `token_id` on-device
/// from `ctx_ptr[1]` (the decode-graph `[position, token_id]` context). Used
/// only under decode-graph capture; the non-capture path keeps a plain DtoD
/// memcpy. CUDA-only kernel (`embed_gather_f16_dpos`).
pub fn embed_gather_f16_dpos<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    embd: flambeau_core::DevicePtr,
    dst: flambeau_core::DevicePtr,
    hidden: usize,
    ctx_ptr: flambeau_core::DevicePtr,
) -> Result<()> {
    let module = reg.expect_module("embed_gather_f16_dpos")?;
    let kernel = B::kernel(module, "flambeau_embed_gather_f16_dpos")?;

    let hidden_i = hidden as i32;
    let embd_p: u64 = embd.as_usize() as u64;
    let dst_p: u64 = dst.as_usize() as u64;
    let ctx_p: u64 = ctx_ptr.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&embd_p);
    args.push_arg(&dst_p);
    args.push_arg(&hidden_i);
    args.push_arg(&ctx_p);
    let threads = 256u32;
    let cfg = LaunchCfg::one_d((hidden as u32).div_ceil(threads), threads);
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Q4_K embedding gather, F16 out (`embed_gather_q4_k_f16`): `dst[r]` =
/// dequant(row `ids[r]` of `embd`), bit-identical to the host Q4_K dequant +
/// f16 cast. `hidden` must be a multiple of the 256-element super-block.
pub fn embed_gather_q4_k_f16<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    embd: flambeau_core::DevicePtr,
    dst: flambeau_core::DevicePtr,
    ids: flambeau_core::DevicePtr,
    n_rows: usize,
    hidden: usize,
) -> Result<()> {
    const QK_K: usize = 256;
    if hidden == 0 || hidden % QK_K != 0 {
        anyhow::bail!("embed_gather_q4_k_f16: hidden {hidden} not a multiple of {QK_K}");
    }
    if n_rows == 0 {
        return Ok(());
    }
    let module = reg.expect_module("embed_gather_q4_k_f16")?;
    let kernel = B::kernel(module, "flambeau_embed_gather_q4_k_f16")?;

    let embd_p: u64 = embd.as_usize() as u64;
    let dst_p: u64 = dst.as_usize() as u64;
    let ids_p: u64 = ids.as_usize() as u64;
    let hidden_i = hidden as i32;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&embd_p);
    args.push_arg(&dst_p);
    args.push_arg(&ids_p);
    args.push_arg(&hidden_i);
    let cfg = LaunchCfg {
        grid: (n_rows as u32, (hidden / QK_K) as u32, 1),
        block: (QK_K as u32, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}
