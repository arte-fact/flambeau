//! Attention — decode (F16 KV + Q8_0 KV) and prefill (F16 KV).
//! GQA is handled via the `n_heads_kv` argument; the kernel broadcasts one
//! KV head across `n_heads_q / n_heads_kv` Q heads.

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "op wrapper — every unsafe block is a kernel.launch or memcpy_async over \
              DevicePtrs validated by the caller; the kernel stem + entry are resolved \
              through the validated registry and the ABI matches the kernels extern-C \
              signature."
)]

use anyhow::Result;
use flambeau_backend_hip::{HipStream, KernelArgs, LaunchCfg};
use flambeau_core::DevicePtr;

use super::OpsRegistry;

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
pub fn attention_decode_f16(
    ctx: crate::OpCtx<'_>,
    buffers: crate::AttnBuffers,
    shape: crate::AttnDecodeShape,
    knobs: crate::AttnKnobs,
) -> Result<()> {
    attention_decode_f16_slots(ctx, buffers, shape, knobs, None)
}

/// 7.a-i3 — graph-captureable variant of [`attention_decode_f16`].
/// Identical behaviour for non-capture callers (slots=None). When
/// `slots` is Some, the n_tokens_kv kernel arg is tagged via
/// `KernelArgs::push_slot` so the graph recorder can bind the slot.
/// Caller then updates the slot per replay via `HipGraphExec::set_slot`
/// to track the growing KV cache tail.
pub fn attention_decode_f16_slots(
    ctx: crate::OpCtx<'_>,
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
    let crate::AttnKnobs { scale, window_size } = knobs;
    assert!(
        head_dim == 64 || head_dim == 128 || head_dim == 256 || head_dim == 512,
        "attention_decode_f16: head_dim {head_dim} not supported (expected 64, 128, 256, or 512)"
    );
    let module = ctx.reg.expect_module("attention_decode_f16")?;
    let kernel = module.kernel("flambeau_attention_decode_f16")?;

    let n_heads_q_i = n_heads_q as i32;
    let n_heads_kv_i = n_heads_kv as i32;
    let head_dim_i = head_dim as i32;
    let n_tokens_i = n_tokens_kv as i32;
    let q_ptr: u64 = q.as_usize() as u64;
    let k_ptr: u64 = k.as_usize() as u64;
    let v_ptr: u64 = v.as_usize() as u64;
    let o_ptr: u64 = out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&q_ptr);
    args.push(&k_ptr);
    args.push(&v_ptr);
    args.push(&o_ptr);
    args.push(&n_heads_q_i);
    args.push(&n_heads_kv_i);
    args.push(&head_dim_i);
    match slots {
        Some(s) => args.push_slot(&n_tokens_i, s.n_tokens_kv),
        None => args.push(&n_tokens_i),
    }
    args.push(&scale);
    args.push(&window_size);
    let cfg = LaunchCfg::one_d(n_heads_q as u32, head_dim as u32);
    unsafe { kernel.launch(ctx.stream, cfg, args)? };
    Ok(())
}

/// **#266b** — single-launch batched-decode attention over `n_slots`
/// (Q-row, per-slot KV-cache) pairs. Replaces the per-slot loop in
/// `forward_full_attn_layer_decode_batched_*`, which structurally caps
/// hybrid throughput at ~1.0× (per #267 cert).
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
///   [`attention_decode_f16_slots`] for the same inputs (regression guard
///   for the wiring task #266c).
/// # Safety
/// All device pointers must outlive the kernel launch and remain valid
/// on the stream's device. `k_cache_ptrs` / `v_cache_ptrs` / `n_tokens_kv`
/// must each point at device buffers of length ≥ `n_slots`. Each
/// `k_cache_ptrs[s]` and `v_cache_ptrs[s]` must point at ≥ `n_tokens_kv[s] *
/// n_heads_kv * head_dim` F16 elements. `q` / `out` must each point at
/// ≥ `n_slots * n_heads_q * head_dim` F16 elements.
pub fn attention_decode_f16_batched(
    ctx: crate::OpCtx<'_>,
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
    let crate::AttnKnobs { scale, window_size } = knobs;
    assert!(
        head_dim == 64 || head_dim == 128 || head_dim == 256 || head_dim == 512,
        "attention_decode_f16_batched: head_dim {head_dim} not supported (expected 64, 128, 256, or 512)"
    );
    assert!(
        (1..=32).contains(&n_slots),
        "attention_decode_f16_batched: n_slots {n_slots} out of supported range [1, 32]"
    );
    let module = ctx.reg.expect_module("attention_decode_f16_batched")?;
    let kernel = module.kernel("flambeau_attention_decode_f16_batched")?;

    let n_heads_q_i = n_heads_q as i32;
    let n_heads_kv_i = n_heads_kv as i32;
    let head_dim_i = head_dim as i32;
    let n_slots_i = n_slots as i32;
    let q_ptr: u64 = q_batched.as_usize() as u64;
    let k_ptrs_ptr: u64 = k_cache_ptrs.as_usize() as u64;
    let v_ptrs_ptr: u64 = v_cache_ptrs.as_usize() as u64;
    let o_ptr: u64 = out_batched.as_usize() as u64;
    let n_kv_ptr: u64 = n_tokens_kv.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&q_ptr);
    args.push(&k_ptrs_ptr);
    args.push(&v_ptrs_ptr);
    args.push(&o_ptr);
    args.push(&n_kv_ptr);
    args.push(&n_heads_q_i);
    args.push(&n_heads_kv_i);
    args.push(&head_dim_i);
    args.push(&n_slots_i);
    args.push(&scale);
    args.push(&window_size);
    let cfg = LaunchCfg {
        grid: (n_heads_q as u32, n_slots as u32, 1),
        block: (head_dim as u32, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(ctx.stream, cfg, args)? };
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
pub fn attention_decode_f16_paged(
    ctx: crate::OpCtx<'_>,
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
    let module = ctx.reg.expect_module("attention_decode_f16_paged")?;
    let kernel = module.kernel("flambeau_attention_decode_f16_paged")?;

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
    let mut args = KernelArgs::new();
    args.push(&q_ptr);
    args.push(&k_pool_ptr);
    args.push(&v_pool_ptr);
    args.push(&bt_ptr);
    args.push(&o_ptr);
    args.push(&n_kv_ptr);
    args.push(&n_heads_q_i);
    args.push(&n_heads_kv_i);
    args.push(&head_dim_i);
    args.push(&n_slots_i);
    args.push(&page_size_i);
    args.push(&max_pps_i);
    args.push(&scale);
    let cfg = LaunchCfg {
        grid: (n_heads_q as u32, n_slots as u32, 1),
        block: (head_dim as u32, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(ctx.stream, cfg, args)? };
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
pub fn kv_append_f16_paged_slots(
    reg: &OpsRegistry,
    stream: &HipStream,
    k_src: DevicePtr,
    v_src: DevicePtr,
    k_pool: DevicePtr,
    v_pool: DevicePtr,
    block_tables: DevicePtr,
    slot_write_pos: DevicePtr,
    n_slots: usize,
    kv_width: usize,
    page_size: usize,
    max_pages_per_slot: usize,
) -> Result<()> {
    assert!(
        page_size > 0 && page_size.is_power_of_two(),
        "kv_append_f16_paged_slots: page_size {page_size} must be a positive power of two"
    );
    assert!(
        max_pages_per_slot >= 1,
        "kv_append_f16_paged_slots: max_pages_per_slot must be >= 1"
    );
    let module = reg.expect_module("kv_append_f16_paged_slots")?;
    let kernel = module.kernel("flambeau_kv_append_f16_paged_slots")?;

    let n_slots_i = n_slots as i32;
    let kv_width_i = kv_width as i32;
    let page_size_i = page_size as i32;
    let max_pps_i = max_pages_per_slot as i32;
    let k_src_ptr: u64 = k_src.as_usize() as u64;
    let v_src_ptr: u64 = v_src.as_usize() as u64;
    let k_pool_ptr: u64 = k_pool.as_usize() as u64;
    let v_pool_ptr: u64 = v_pool.as_usize() as u64;
    let bt_ptr: u64 = block_tables.as_usize() as u64;
    let wpos_ptr: u64 = slot_write_pos.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&k_src_ptr);
    args.push(&v_src_ptr);
    args.push(&k_pool_ptr);
    args.push(&v_pool_ptr);
    args.push(&bt_ptr);
    args.push(&wpos_ptr);
    args.push(&n_slots_i);
    args.push(&kv_width_i);
    args.push(&page_size_i);
    args.push(&max_pps_i);
    let block_threads: u32 = kv_width.min(128) as u32;
    let cfg = LaunchCfg {
        grid: (n_slots as u32, 1, 1),
        block: (block_threads.max(1), 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
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
pub fn attention_prefill_f16_paged(
    reg: &OpsRegistry,
    stream: &HipStream,
    q: DevicePtr,
    k_pool: DevicePtr,
    v_pool: DevicePtr,
    block_table: DevicePtr,
    out: DevicePtr,
    n_q_tokens: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    n_k_tokens: usize,
    q_offset: usize,
    page_size: usize,
    scale: f32,
    window_size: i32,
) -> Result<()> {
    assert!(
        head_dim == 64 || head_dim == 128 || head_dim == 256 || head_dim == 512,
        "attention_prefill_f16_paged: head_dim {head_dim} not supported (expected 64, 128, 256, or 512)"
    );
    assert!(
        page_size > 0 && page_size.is_power_of_two(),
        "attention_prefill_f16_paged: page_size {page_size} must be a positive power of two"
    );
    let module = reg.expect_module("attention_prefill_f16_paged")?;
    let kernel = module.kernel("flambeau_attention_prefill_f16_paged")?;

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
    let mut args = KernelArgs::new();
    args.push(&q_ptr);
    args.push(&k_ptr);
    args.push(&v_ptr);
    args.push(&bt_ptr);
    args.push(&o_ptr);
    args.push(&n_q_i);
    args.push(&n_heads_q_i);
    args.push(&n_heads_kv_i);
    args.push(&head_dim_i);
    args.push(&n_k_i);
    args.push(&q_off_i);
    args.push(&page_size_i);
    args.push(&scale);
    args.push(&window_size);
    let cfg = LaunchCfg {
        grid: (n_q_tokens as u32, n_heads_q as u32, 1),
        block: (head_dim as u32, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
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
pub fn kv_append_f16_paged_prefill(
    reg: &OpsRegistry,
    stream: &HipStream,
    k_src: DevicePtr,
    v_src: DevicePtr,
    k_pool: DevicePtr,
    v_pool: DevicePtr,
    block_table: DevicePtr,
    n_tokens: usize,
    kv_width: usize,
    start_pos: usize,
    page_size: usize,
) -> Result<()> {
    assert!(
        page_size > 0 && page_size.is_power_of_two(),
        "kv_append_f16_paged_prefill: page_size {page_size} must be a positive power of two"
    );
    let module = reg.expect_module("kv_append_f16_paged_prefill")?;
    let kernel = module.kernel("flambeau_kv_append_f16_paged_prefill")?;

    let n_tokens_i = n_tokens as i32;
    let kv_width_i = kv_width as i32;
    let start_pos_i = start_pos as i32;
    let page_size_i = page_size as i32;
    let k_src_ptr: u64 = k_src.as_usize() as u64;
    let v_src_ptr: u64 = v_src.as_usize() as u64;
    let k_pool_ptr: u64 = k_pool.as_usize() as u64;
    let v_pool_ptr: u64 = v_pool.as_usize() as u64;
    let bt_ptr: u64 = block_table.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&k_src_ptr);
    args.push(&v_src_ptr);
    args.push(&k_pool_ptr);
    args.push(&v_pool_ptr);
    args.push(&bt_ptr);
    args.push(&n_tokens_i);
    args.push(&kv_width_i);
    args.push(&start_pos_i);
    args.push(&page_size_i);
    let block_threads: u32 = kv_width.min(128) as u32;
    let cfg = LaunchCfg {
        grid: (n_tokens as u32, 1, 1),
        block: (block_threads.max(1), 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
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
pub fn kv_append_f16_batched_slots(
    reg: &OpsRegistry,
    stream: &HipStream,
    k_src: DevicePtr,
    v_src: DevicePtr,
    slot_k_dst_ptrs: DevicePtr,
    slot_v_dst_ptrs: DevicePtr,
    slot_write_pos: DevicePtr,
    n_slots: usize,
    kv_width: usize,
) -> Result<()> {
    let module = reg.expect_module("kv_append_f16_batched_slots")?;
    let kernel = module.kernel("flambeau_kv_append_f16_batched_slots")?;

    let n_slots_i = n_slots as i32;
    let kv_width_i = kv_width as i32;
    let k_src_ptr: u64 = k_src.as_usize() as u64;
    let v_src_ptr: u64 = v_src.as_usize() as u64;
    let k_dst_arr: u64 = slot_k_dst_ptrs.as_usize() as u64;
    let v_dst_arr: u64 = slot_v_dst_ptrs.as_usize() as u64;
    let wpos_ptr: u64 = slot_write_pos.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&k_src_ptr);
    args.push(&v_src_ptr);
    args.push(&k_dst_arr);
    args.push(&v_dst_arr);
    args.push(&wpos_ptr);
    args.push(&n_slots_i);
    args.push(&kv_width_i);
    let block_threads: u32 = kv_width.min(128) as u32;
    let cfg = LaunchCfg {
        grid: (n_slots as u32, 1, 1),
        block: (block_threads.max(1), 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// 9.b — split-K (flash-decoding) decode attention, F16 KV. Same math
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
pub fn attention_decode_f16_splitk(
    ctx: crate::OpCtx<'_>,
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
    } = shape;
    let crate::AttnKnobs { scale, window_size } = knobs;
    assert!(
        head_dim == 64 || head_dim == 128 || head_dim == 256 || head_dim == 512,
        "attention_decode_f16_splitk: head_dim {head_dim} not supported"
    );
    assert!(chunk_size > 0);

    let module = ctx.reg.expect_module("attention_decode_f16_splitk")?;
    let k_chunk = module.kernel("flambeau_attention_decode_f16_splitk_chunk")?;
    let k_combine = module.kernel("flambeau_attention_decode_f16_splitk_combine")?;

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
    let mut a1 = KernelArgs::new();
    a1.push(&q_ptr);
    a1.push(&k_ptr);
    a1.push(&v_ptr);
    a1.push(&m_ptr);
    a1.push(&s_ptr);
    a1.push(&po_ptr);
    a1.push(&n_heads_q_i);
    a1.push(&n_heads_kv_i);
    a1.push(&head_dim_i);
    a1.push(&n_tokens_i);
    a1.push(&n_chunks_i);
    a1.push(&chunk_size_i);
    a1.push(&scale);
    a1.push(&window_size);
    let cfg1 = LaunchCfg {
        grid: (n_heads_q as u32, n_chunks as u32, 1),
        block: (head_dim as u32, 1, 1),
        shared_bytes: 0,
    };
    unsafe { k_chunk.launch(ctx.stream, cfg1, a1)? };

    let mut a2 = KernelArgs::new();
    a2.push(&m_ptr);
    a2.push(&s_ptr);
    a2.push(&po_ptr);
    a2.push(&o_ptr);
    a2.push(&n_heads_q_i);
    a2.push(&n_chunks_i);
    a2.push(&head_dim_i);
    let cfg2 = LaunchCfg {
        grid: (n_heads_q as u32, 1, 1),
        block: (head_dim as u32, 1, 1),
        shared_bytes: 0,
    };
    unsafe { k_combine.launch(ctx.stream, cfg2, a2)? };

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
pub fn attention_decode_f16_splitk_h2(
    ctx: crate::OpCtx<'_>,
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
    } = shape;
    let crate::AttnKnobs { scale, window_size } = knobs;
    assert!(
        head_dim == 128 || head_dim == 256 || head_dim == 512,
        "attention_decode_f16_splitk_h2: head_dim {head_dim} not in {{128, 256, 512}}"
    );
    assert!(chunk_size > 0);
    assert!(head_dim % 2 == 0);

    let module = ctx.reg.expect_module("attention_decode_f16_splitk_h2")?;
    let k_chunk = module.kernel("flambeau_attention_decode_f16_splitk_h2_chunk")?;
    let combine_module = ctx.reg.expect_module("attention_decode_f16_splitk")?;
    let k_combine = combine_module.kernel("flambeau_attention_decode_f16_splitk_combine")?;

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

    let mut a1 = KernelArgs::new();
    a1.push(&q_ptr);
    a1.push(&k_ptr);
    a1.push(&v_ptr);
    a1.push(&m_ptr);
    a1.push(&s_ptr);
    a1.push(&po_ptr);
    a1.push(&n_heads_q_i);
    a1.push(&n_heads_kv_i);
    a1.push(&head_dim_i);
    a1.push(&n_tokens_i);
    a1.push(&n_chunks_i);
    a1.push(&chunk_size_i);
    a1.push(&scale);
    a1.push(&window_size);
    let cfg1 = LaunchCfg {
        grid: (n_heads_q as u32, n_chunks as u32, 1),
        block: ((head_dim / 2) as u32, 1, 1),
        shared_bytes: 0,
    };
    unsafe { k_chunk.launch(ctx.stream, cfg1, a1)? };

    let mut a2 = KernelArgs::new();
    a2.push(&m_ptr);
    a2.push(&s_ptr);
    a2.push(&po_ptr);
    a2.push(&o_ptr);
    a2.push(&n_heads_q_i);
    a2.push(&n_chunks_i);
    a2.push(&head_dim_i);
    let cfg2 = LaunchCfg {
        grid: (n_heads_q as u32, 1, 1),
        block: (head_dim as u32, 1, 1),
        shared_bytes: 0,
    };
    unsafe { k_combine.launch(ctx.stream, cfg2, a2)? };

    Ok(())
}

/// Pick a reasonable split-K chunk size given `n_tokens`. Target n_chunks in
/// [4, 16] so we land 16 heads × n_chunks = 64–256 blocks on 60 CUs
/// (1–4× saturation, more than enough to hide the per-block serial loop).
/// Returns the single-pass fallback `chunk_size = n_tokens` when the
/// context is short enough that split-K overhead (the combine kernel +
/// partials write/read) costs more than the occupancy win.
pub fn splitk_chunk_size(n_tokens_kv: usize) -> usize {
    // Threshold tuned per the 9.b A/B: at n_tokens=128 split-K already
    // ties the single-pass (1.10×) and every larger shape wins hard, so
    // default to split-K whenever it gives ≥ 4 chunks.
    if n_tokens_kv <= 256 {
        n_tokens_kv // one chunk; caller should just use single-pass
    } else if n_tokens_kv <= 1024 {
        128
    } else if n_tokens_kv <= 2048 {
        256
    } else {
        512
    }
}

/// Decode attention with Q8_0-quantised KV. Same args as the F16 variant;
/// `k_cache` / `v_cache` hold `flambeau_block_q8_0` blocks laid out as
/// `[n_tokens_kv, n_heads_kv, head_dim/32]` row-major.
pub fn attention_decode_q8_kv(
    ctx: crate::OpCtx<'_>,
    buffers: crate::AttnBuffers,
    shape: crate::AttnDecodeShape,
    knobs: crate::AttnKnobs,
) -> Result<()> {
    let crate::AttnBuffers { q, k, v, out } = buffers;
    let crate::AttnDecodeShape {
        n_heads_q,
        n_heads_kv,
        head_dim,
        n_tokens_kv,
    } = shape;
    let crate::AttnKnobs { scale, window_size } = knobs;
    // Kernel supports head_dim ∈ {64, 128, 256, 512}. block = head_dim/4
    // threads (16/32/64/128). At d=512 the block is 2 waves and the
    // sum-of-blocks reduction adds a tiny cross-wave LDS rendezvous
    // (`score_parts[2]` + 2 __syncthreads). SWA layers (window_size > 0)
    // are supported via the t_start clamp in the kernel inner loop.
    assert!(
        matches!(head_dim, 64 | 128 | 256 | 512),
        "attention_decode_q8_kv: head_dim {head_dim} not supported (expected 64, 128, 256, or 512)"
    );
    let module = ctx.reg.expect_module("attention_decode_q8_kv")?;
    let kernel = module.kernel("flambeau_attention_decode_q8_kv")?;

    let n_heads_q_i = n_heads_q as i32;
    let n_heads_kv_i = n_heads_kv as i32;
    let head_dim_i = head_dim as i32;
    let n_tokens_i = n_tokens_kv as i32;
    let q_ptr: u64 = q.as_usize() as u64;
    let k_ptr: u64 = k.as_usize() as u64;
    let v_ptr: u64 = v.as_usize() as u64;
    let o_ptr: u64 = out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&q_ptr);
    args.push(&k_ptr);
    args.push(&v_ptr);
    args.push(&o_ptr);
    args.push(&n_heads_q_i);
    args.push(&n_heads_kv_i);
    args.push(&head_dim_i);
    args.push(&n_tokens_i);
    args.push(&scale);
    args.push(&window_size);
    // block.x = head_dim/4 (one thread per int32-
    // packed quad). For head_dim=256 that's 64 threads = 1 wavefront.
    let cfg = LaunchCfg::one_d(n_heads_q as u32, (head_dim / 4) as u32);
    unsafe { kernel.launch(ctx.stream, cfg, args)? };
    Ok(())
}

/// Split-K (flash-decoding) decode attention with Q8_0 KV. Same shape as
/// [`attention_decode_f16_splitk`] but K and V come from a `KvCache<Q8Contig>`
/// — dequantised on the fly during the dot product and the V-accumulate.
/// Closes the long-context Q8↔F16 gap (single-pass `attention_decode_q8_kv`
/// is the same shape as the single-pass F16 kernel and pays the same 7.78×
/// occupancy penalty past 256 KV tokens).
pub fn attention_decode_q8_kv_splitk(
    ctx: crate::OpCtx<'_>,
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
    } = shape;
    let crate::AttnKnobs { scale, window_size } = knobs;
    // head_dim ∈ {64, 128, 256, 512}. d=512 enables Q8 on gemma4
    // global layers; the kernel adds a cross-wave LDS reduce for that
    // case (see attention_decode_q8_kv comments).
    assert!(
        matches!(head_dim, 64 | 128 | 256 | 512),
        "attention_decode_q8_kv_splitk: head_dim {head_dim} not supported"
    );
    assert!(chunk_size > 0);

    let module = ctx.reg.expect_module("attention_decode_q8_kv_splitk")?;
    let k_chunk = module.kernel("flambeau_attention_decode_q8_kv_splitk_chunk")?;
    let k_combine = module.kernel("flambeau_attention_decode_q8_kv_splitk_combine")?;

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

    let mut a1 = KernelArgs::new();
    a1.push(&q_ptr);
    a1.push(&k_ptr);
    a1.push(&v_ptr);
    a1.push(&m_ptr);
    a1.push(&s_ptr);
    a1.push(&po_ptr);
    a1.push(&n_heads_q_i);
    a1.push(&n_heads_kv_i);
    a1.push(&head_dim_i);
    a1.push(&n_tokens_i);
    a1.push(&n_chunks_i);
    a1.push(&chunk_size_i);
    a1.push(&scale);
    a1.push(&window_size);
    // chunk pass uses block = head_dim/4 (one
    // thread per int32-packed quad). Combine pass still needs
    // head_dim threads (one per output element).
    let cfg1 = LaunchCfg {
        grid: (n_heads_q as u32, n_chunks as u32, 1),
        block: ((head_dim / 4) as u32, 1, 1),
        shared_bytes: 0,
    };
    unsafe { k_chunk.launch(ctx.stream, cfg1, a1)? };

    let mut a2 = KernelArgs::new();
    a2.push(&m_ptr);
    a2.push(&s_ptr);
    a2.push(&po_ptr);
    a2.push(&o_ptr);
    a2.push(&n_heads_q_i);
    a2.push(&n_chunks_i);
    a2.push(&head_dim_i);
    let cfg2 = LaunchCfg {
        grid: (n_heads_q as u32, 1, 1),
        block: (head_dim as u32, 1, 1),
        shared_bytes: 0,
    };
    unsafe { k_combine.launch(ctx.stream, cfg2, a2)? };

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
pub fn attention_prefill_q8_kv(
    reg: &OpsRegistry,
    stream: &HipStream,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    out: DevicePtr,
    n_q_tokens: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    n_k_tokens: usize,
    q_offset: usize,
    scale: f32,
    window_size: i32,
) -> Result<()> {
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
    let scale_f = scale;
    let window_i = window_size;
    let q_ptr: u64 = q.as_usize() as u64;
    let k_ptr: u64 = k_cache.as_usize() as u64;
    let v_ptr: u64 = v_cache.as_usize() as u64;
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
        let kernel = module.kernel(entry)?;
        let mut args = KernelArgs::new();
        args.push(&q_ptr);
        args.push(&k_ptr);
        args.push(&v_ptr);
        args.push(&o_ptr);
        args.push(&n_q_i);
        args.push(&n_heads_q_i);
        args.push(&n_heads_kv_i);
        args.push(&n_k_i);
        args.push(&q_off_i);
        args.push(&scale_f);
        args.push(&window_i);
        let br: u32 = if head_dim == 256 { 8 } else { 4 };
        const WARP: u32 = 64;
        let cfg = LaunchCfg {
            grid: ((n_q_tokens as u32).div_ceil(br), n_heads_q as u32, 1),
            block: (WARP, br, 1),
            shared_bytes: 0,
        };
        unsafe { kernel.launch(stream, cfg, args)? };
        return Ok(());
    }

    // Oracle path for n_q < 4 (very short prompts / edge shapes).
    let module = reg.expect_module("attention_prefill_q8_kv")?;
    let kernel = module.kernel("flambeau_attention_prefill_q8_kv")?;
    let head_dim_i = head_dim as i32;
    let mut args = KernelArgs::new();
    args.push(&q_ptr);
    args.push(&k_ptr);
    args.push(&v_ptr);
    args.push(&o_ptr);
    args.push(&n_q_i);
    args.push(&n_heads_q_i);
    args.push(&n_heads_kv_i);
    args.push(&head_dim_i);
    args.push(&n_k_i);
    args.push(&q_off_i);
    args.push(&scale_f);
    args.push(&window_i);
    let cfg = LaunchCfg {
        grid: (n_q_tokens as u32, n_heads_q as u32, 1),
        block: ((head_dim / 4) as u32, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Split the interleaved `(Q, gate)` output of a gated-attention query
/// projection into two contiguous F16 tensors. Used by Qwen3.5/3.6 full-attn
/// layers, where the `attn_q` weight projects to `2 * head_dim` per head —
/// first half is Q, second half is the output gate. Q goes into the
/// attention kernel; gate is held for a silu-multiply after attention.
/// Launch: `(n_tokens, n_heads, ceil(head_dim/128))` × 128 threads.
pub fn split_q_gate_f16(
    reg: &OpsRegistry,
    stream: &HipStream,
    fused_qg: DevicePtr,
    q_out: DevicePtr,
    gate_out: DevicePtr,
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
) -> Result<()> {
    let module = reg.expect_module("split_q_gate_f16")?;
    let kernel = module.kernel("flambeau_split_q_gate_f16")?;

    let n_tokens_i = n_tokens as i32;
    let n_heads_i = n_heads as i32;
    let head_dim_i = head_dim as i32;
    let f_ptr: u64 = fused_qg.as_usize() as u64;
    let q_ptr: u64 = q_out.as_usize() as u64;
    let g_ptr: u64 = gate_out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&f_ptr);
    args.push(&q_ptr);
    args.push(&g_ptr);
    args.push(&n_tokens_i);
    args.push(&n_heads_i);
    args.push(&head_dim_i);
    let threads = 128u32;
    let grid_z = (head_dim as u32).div_ceil(threads);
    let cfg = LaunchCfg {
        grid: (n_tokens as u32, n_heads as u32, grid_z),
        block: (threads, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Prefill attention, F16 KV. Computes `n_q_tokens` Q rows against
/// `n_k_tokens` KV rows with causal masking (`q_token_i` attends to
/// `k_token_0..k_token_{q_offset + i}`).
/// Dispatches:
/// - `attention_prefill_flash_tile_f16` (candle port —
///   BR=4 LDS-tiled flash-attn v2) when `n_q_tokens >= 4` and head_dim
///   ∈ {64, 128, 256}. Per-call time on gfx906 is ~5× the previous
///   oracle kernel at pp512 (measured 2026-04-22: 2994 µs → target
///   ≤800 µs).
/// - The per-(q_token, q_head) oracle kernel (`attention_prefill_f16`)
///   otherwise. Shape grid covers the n_q < 4 edge that flash-tile's
///   BR=4 coop-load pattern under-utilises.
pub fn attention_prefill_f16(
    reg: &OpsRegistry,
    stream: &HipStream,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    out: DevicePtr,
    n_q_tokens: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    n_k_tokens: usize,
    q_offset: usize,
    scale: f32,
    window_size: i32,
) -> Result<()> {
    attention_prefill_f16_slots(
        reg,
        stream,
        q,
        k_cache,
        v_cache,
        out,
        n_q_tokens,
        n_heads_q,
        n_heads_kv,
        head_dim,
        n_k_tokens,
        q_offset,
        scale,
        window_size,
        None,
        None,
    )
}

/// 6.a-i4 — graph-captureable variant of [`attention_prefill_f16`].
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
///   [`ScalarSlot`]: flambeau_backend_hip::ScalarSlot
pub fn attention_prefill_f16_slots(
    reg: &OpsRegistry,
    stream: &HipStream,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    out: DevicePtr,
    n_q_tokens: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    n_k_tokens: usize,
    q_offset: usize,
    scale: f32,
    window_size: i32,
    n_k_slot: Option<flambeau_backend_hip::ScalarSlot>,
    q_off_slot: Option<flambeau_backend_hip::ScalarSlot>,
) -> Result<()> {
    assert!(
        head_dim == 64 || head_dim == 128 || head_dim == 256 || head_dim == 512,
        "attention_prefill_f16: head_dim {head_dim} not supported (expected 64, 128, 256, or 512)"
    );

    // flash_tile prefill now SWA-aware (window_size kernel arg lands
    // alongside causal + per-warp swa_min masking). Route through it
    // whenever n_q ≥ 4.
    let use_flash_tile = n_q_tokens >= 4;

    let q_ptr: u64 = q.as_usize() as u64;
    let k_ptr: u64 = k_cache.as_usize() as u64;
    let v_ptr: u64 = v_cache.as_usize() as u64;
    let o_ptr: u64 = out.as_usize() as u64;
    let n_q_i = n_q_tokens as i32;
    let n_heads_q_i = n_heads_q as i32;
    let n_heads_kv_i = n_heads_kv as i32;
    let n_k_i = n_k_tokens as i32;
    let q_off_i = q_offset as i32;
    let scale_f = scale;

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
        let kernel = module.kernel(entry)?;
        let window_i = window_size;
        let mut args = KernelArgs::new();
        args.push(&q_ptr);
        args.push(&k_ptr);
        args.push(&v_ptr);
        args.push(&o_ptr);
        args.push(&n_q_i);
        args.push(&n_heads_q_i);
        args.push(&n_heads_kv_i);
        push_scalar_maybe_slot(&mut args, &n_k_i, n_k_slot);
        push_scalar_maybe_slot(&mut args, &q_off_i, q_off_slot);
        args.push(&scale_f);
        args.push(&window_i);
        // 9.b — BR depends on which variant we dispatch to.
        // d256_br8 uses BR=8 (more Q rows per block, fewer blocks);
        // other head_dims still use BR=4.
        let br: u32 = if head_dim == 256 { 8 } else { 4 };
        const WARP: u32 = 64;
        let cfg = LaunchCfg {
            grid: ((n_q_tokens as u32).div_ceil(br), n_heads_q as u32, 1),
            block: (WARP, br, 1),
            shared_bytes: 0,
        };
        unsafe { kernel.launch(stream, cfg, args)? };
        return Ok(());
    }

    // Oracle path for n_q < 4 (very short prompts / edge shapes) or
    // any SWA call (flash_tile is not SWA-aware yet).
    let module = reg.expect_module("attention_prefill_f16")?;
    let kernel = module.kernel("flambeau_attention_prefill_f16")?;
    let head_dim_i = head_dim as i32;
    let window_i = window_size;
    let mut args = KernelArgs::new();
    args.push(&q_ptr);
    args.push(&k_ptr);
    args.push(&v_ptr);
    args.push(&o_ptr);
    args.push(&n_q_i);
    args.push(&n_heads_q_i);
    args.push(&n_heads_kv_i);
    args.push(&head_dim_i);
    push_scalar_maybe_slot(&mut args, &n_k_i, n_k_slot);
    push_scalar_maybe_slot(&mut args, &q_off_i, q_off_slot);
    args.push(&scale_f);
    args.push(&window_i);
    let cfg = LaunchCfg {
        grid: (n_q_tokens as u32, n_heads_q as u32, 1),
        block: (head_dim as u32, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

#[inline]
fn push_scalar_maybe_slot<'a, T: 'a>(
    args: &mut KernelArgs<'a>,
    v: &'a T,
    slot: Option<flambeau_backend_hip::ScalarSlot>,
) {
    match slot {
        Some(s) => args.push_slot(v, s),
        None => args.push(v),
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
pub fn kv_append_v_unit_norm_f16(
    reg: &OpsRegistry,
    stream: &HipStream,
    k_src: DevicePtr,
    v_src: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    n_tokens: usize,
    n_kv_heads: usize,
    head_dim: usize,
    write_pos: usize,
    eps: f32,
) -> Result<()> {
    let entry = match head_dim {
        64 => "flambeau_kv_append_v_unit_norm_f16_d64",
        128 => "flambeau_kv_append_v_unit_norm_f16_d128",
        256 => "flambeau_kv_append_v_unit_norm_f16_d256",
        512 => "flambeau_kv_append_v_unit_norm_f16_d512",
        other => anyhow::bail!(
            "kv_append_v_unit_norm_f16: head_dim {other} not in {{64, 128, 256, 512}}"
        ),
    };
    let module = reg.expect_module("kv_append_v_unit_norm_f16")?;
    let kernel = module.kernel(entry)?;

    let n_kv_heads_i = n_kv_heads as i32;
    let write_pos_i = write_pos as i32;
    let eps_f = eps;
    let k_src_p: u64 = k_src.as_usize() as u64;
    let v_src_p: u64 = v_src.as_usize() as u64;
    let k_dst_p: u64 = k_cache.as_usize() as u64;
    let v_dst_p: u64 = v_cache.as_usize() as u64;
    let mut args = flambeau_backend_hip::KernelArgs::new();
    args.push(&k_src_p);
    args.push(&v_src_p);
    args.push(&k_dst_p);
    args.push(&v_dst_p);
    args.push(&n_kv_heads_i);
    args.push(&write_pos_i);
    args.push(&eps_f);
    let cfg = flambeau_backend_hip::LaunchCfg {
        grid: (n_tokens as u32, n_kv_heads as u32, 1),
        block: (head_dim as u32, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}
