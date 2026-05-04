//! Attention — decode (F16 KV + Q8_0 KV) and prefill (F16 KV).
//!
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
///
/// Shapes:
/// - `q[n_heads_q, head_dim]` F16
/// - `k_cache[n_tokens_kv, n_heads_kv, head_dim]` F16 contiguous
/// - `v_cache` same shape as K
/// - `out[n_heads_q, head_dim]` F16
///
/// Launch: one block per Q head, `head_dim` threads/block (one thread per
/// output lane). Kernel supports `head_dim ∈ {128, 256}` — both
/// Qwen3.5 (GQA-32/4, head_dim=128) and Qwen3.6 (GQA-16/2, head_dim=256).
pub fn attention_decode_f16(
    reg: &OpsRegistry,
    stream: &HipStream,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    out: DevicePtr,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    n_tokens_kv: usize,
    scale: f32,
) -> Result<()> {
    attention_decode_f16_slots(
        reg, stream, q, k_cache, v_cache, out, n_heads_q, n_heads_kv, head_dim,
        n_tokens_kv, scale, None,
    )
}

/// V2.27.a-i3 — graph-captureable variant of [`attention_decode_f16`].
///
/// Identical behaviour for non-capture callers (slot=None). When
/// `n_tokens_kv_slot` is Some, the n_tokens_kv kernel arg is tagged
/// via `KernelArgs::push_slot` so the graph recorder can bind the
/// slot. Caller then updates the slot per replay via
/// `HipGraphExec::set_slot` to track the growing KV cache tail.
#[allow(clippy::too_many_arguments)]
pub fn attention_decode_f16_slots(
    reg: &OpsRegistry,
    stream: &HipStream,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    out: DevicePtr,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    n_tokens_kv: usize,
    scale: f32,
    n_tokens_kv_slot: Option<flambeau_backend_hip::ScalarSlot>,
) -> Result<()> {
    assert!(
        head_dim == 64 || head_dim == 128 || head_dim == 256,
        "attention_decode_f16: head_dim {head_dim} not supported (expected 64, 128, or 256)"
    );
    let module = reg.expect_module("attention_decode_f16")?;
    let kernel = module.kernel("flambeau_attention_decode_f16")?;

    let n_heads_q_i = n_heads_q as i32;
    let n_heads_kv_i = n_heads_kv as i32;
    let head_dim_i = head_dim as i32;
    let n_tokens_i = n_tokens_kv as i32;
    let scale_f = scale;
    let q_ptr: u64 = q.as_usize() as u64;
    let k_ptr: u64 = k_cache.as_usize() as u64;
    let v_ptr: u64 = v_cache.as_usize() as u64;
    let o_ptr: u64 = out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&q_ptr);
    args.push(&k_ptr);
    args.push(&v_ptr);
    args.push(&o_ptr);
    args.push(&n_heads_q_i);
    args.push(&n_heads_kv_i);
    args.push(&head_dim_i);
    match n_tokens_kv_slot {
        Some(s) => args.push_slot(&n_tokens_i, s),
        None => args.push(&n_tokens_i),
    }
    args.push(&scale_f);
    let cfg = LaunchCfg::one_d(n_heads_q as u32, head_dim as u32);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// **#266b** — single-launch batched-decode attention over `n_slots`
/// (Q-row, per-slot KV-cache) pairs. Replaces the per-slot loop in
/// `forward_full_attn_layer_decode_batched_*`, which structurally caps
/// hybrid throughput at ~1.0× (per #267 cert).
///
/// Each grid block owns one `(q_head, slot)` pair and runs the same
/// flash-attn-v2 online-softmax body as [`attention_decode_f16`]. Slot
/// addressing comes from device-side tables built once per call by the
/// dispatcher:
///
/// - `k_cache_ptrs[n_slots]`: device addresses of each slot's K cache.
/// - `v_cache_ptrs[n_slots]`: device addresses of each slot's V cache.
/// - `n_tokens_kv[n_slots]`: per-slot KV-tail length (post-append).
///
/// Q and out are contiguous batched layouts:
///   `q[n_slots, n_heads_q, head_dim]` F16
///   `out[n_slots, n_heads_q, head_dim]` F16
///
/// Per-slot K/V layouts (pointed-to memory) match
/// [`attention_decode_f16`]: `[n_tokens, n_heads_kv, head_dim]` F16.
///
/// Launch: `gridDim = (n_heads_q, n_slots)`, `blockDim = (head_dim,)`.
/// At `n_slots = 1` the kernel produces output bit-identical to
/// [`attention_decode_f16_slots`] for the same inputs (regression guard
/// for the wiring task #266c).
///
/// # Safety
/// All device pointers must outlive the kernel launch and remain valid
/// on the stream's device. `k_cache_ptrs` / `v_cache_ptrs` / `n_tokens_kv`
/// must each point at device buffers of length ≥ `n_slots`. Each
/// `k_cache_ptrs[s]` and `v_cache_ptrs[s]` must point at ≥ `n_tokens_kv[s] *
/// n_heads_kv * head_dim` F16 elements. `q` / `out` must each point at
/// ≥ `n_slots * n_heads_q * head_dim` F16 elements.
#[allow(clippy::too_many_arguments)]
pub fn attention_decode_f16_batched(
    reg: &OpsRegistry,
    stream: &HipStream,
    q_batched: DevicePtr,
    k_cache_ptrs: DevicePtr,
    v_cache_ptrs: DevicePtr,
    out_batched: DevicePtr,
    n_tokens_kv: DevicePtr,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    n_slots: usize,
    scale: f32,
) -> Result<()> {
    assert!(
        head_dim == 64 || head_dim == 128 || head_dim == 256,
        "attention_decode_f16_batched: head_dim {head_dim} not supported (expected 64, 128, or 256)"
    );
    assert!(
        n_slots >= 1 && n_slots <= 32,
        "attention_decode_f16_batched: n_slots {n_slots} out of supported range [1, 32]"
    );
    let module = reg.expect_module("attention_decode_f16_batched")?;
    let kernel = module.kernel("flambeau_attention_decode_f16_batched")?;

    let n_heads_q_i = n_heads_q as i32;
    let n_heads_kv_i = n_heads_kv as i32;
    let head_dim_i = head_dim as i32;
    let n_slots_i = n_slots as i32;
    let scale_f = scale;
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
    args.push(&scale_f);
    let cfg = LaunchCfg {
        grid: (n_heads_q as u32, n_slots as u32, 1),
        block: (head_dim as u32, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// MTP-4-C-4 — BF16 sibling of [`attention_decode_f16`]. Same math
/// (online flash-attn-v2 softmax in F32), BF16 storage throughout.
/// Used by the MTP BF16 forward path.
///
/// Shapes (all BF16 unless noted):
///   q[n_heads_q, head_dim]
///   k_cache[n_tokens_kv, n_heads_kv, head_dim]
///   v_cache same as k
///   out[n_heads_q, head_dim]
///
/// Launch: one block per Q head, `head_dim` threads/block. Supports
/// `head_dim ∈ {64, 128, 256}`; MTP uses 256.
#[allow(clippy::too_many_arguments)]
pub fn attention_decode_bf16(
    reg: &OpsRegistry,
    stream: &HipStream,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    out: DevicePtr,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    n_tokens_kv: usize,
    scale: f32,
) -> Result<()> {
    assert!(
        head_dim == 64 || head_dim == 128 || head_dim == 256,
        "attention_decode_bf16: head_dim {head_dim} not supported (expected 64, 128, or 256)"
    );
    let module = reg.expect_module("attention_decode_bf16")?;
    let kernel = module.kernel("flambeau_attention_decode_bf16")?;

    let n_heads_q_i = n_heads_q as i32;
    let n_heads_kv_i = n_heads_kv as i32;
    let head_dim_i = head_dim as i32;
    let n_tokens_i = n_tokens_kv as i32;
    let scale_f = scale;
    let q_ptr: u64 = q.as_usize() as u64;
    let k_ptr: u64 = k_cache.as_usize() as u64;
    let v_ptr: u64 = v_cache.as_usize() as u64;
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
    args.push(&scale_f);
    let cfg = LaunchCfg::one_d(n_heads_q as u32, head_dim as u32);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// V2.19.b — split-K (flash-decoding) decode attention, F16 KV. Same math
/// as [`attention_decode_f16`] but partitions the context across grid.y to
/// attack the single-pass kernel's occupancy starvation on Qwen3.6
/// (16 heads × 1 block = 27 % of 60 CUs at head_dim=256).
///
/// Two passes:
///   1. `flambeau_attention_decode_f16_splitk_chunk` — grid
///      `(n_heads_q, n_chunks)`, each block owns one (q_head, chunk) pair
///      and emits `(m_c, s_c, o_c[head_dim])` into `partials_*` scratch.
///   2. `flambeau_attention_decode_f16_splitk_combine` — grid
///      `(n_heads_q,)`, merges the `n_chunks` partials per head into the
///      final output using online-softmax rescaling.
///
/// Scratch sizing (caller-provided, f32):
///   * `partials_m` / `partials_s`: `n_heads_q * n_chunks` floats each
///   * `partials_o`: `n_heads_q * n_chunks * head_dim` floats
///
/// Measured (Qwen3.6 shape, head_dim=256, 16/2, MI50):
///   * n_tokens=2048: single-pass 2647 µs → split-K 340 µs = **7.78×**
///   * n_tokens=4096: single-pass 5210 µs → split-K 662 µs = **7.87×**
pub fn attention_decode_f16_splitk(
    reg: &OpsRegistry,
    stream: &HipStream,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    out: DevicePtr,
    partials_m: DevicePtr,
    partials_s: DevicePtr,
    partials_o: DevicePtr,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    n_tokens_kv: usize,
    chunk_size: usize,
    scale: f32,
) -> Result<()> {
    assert!(
        head_dim == 64 || head_dim == 128 || head_dim == 256,
        "attention_decode_f16_splitk: head_dim {head_dim} not supported"
    );
    assert!(chunk_size > 0);

    let module = reg.expect_module("attention_decode_f16_splitk")?;
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
    let k_ptr: u64 = k_cache.as_usize() as u64;
    let v_ptr: u64 = v_cache.as_usize() as u64;
    let o_ptr: u64 = out.as_usize() as u64;
    let m_ptr: u64 = partials_m.as_usize() as u64;
    let s_ptr: u64 = partials_s.as_usize() as u64;
    let po_ptr: u64 = partials_o.as_usize() as u64;
    let scale_f = scale;

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
    a1.push(&scale_f);
    let cfg1 = LaunchCfg {
        grid: (n_heads_q as u32, n_chunks as u32, 1),
        block: (head_dim as u32, 1, 1),
        shared_bytes: 0,
    };
    unsafe { k_chunk.launch(stream, cfg1, a1)? };

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
    unsafe { k_combine.launch(stream, cfg2, a2)? };

    Ok(())
}

/// Pick a reasonable split-K chunk size given `n_tokens`. Target n_chunks in
/// [4, 16] so we land 16 heads × n_chunks = 64–256 blocks on 60 CUs
/// (1–4× saturation, more than enough to hide the per-block serial loop).
///
/// Returns the single-pass fallback `chunk_size = n_tokens` when the
/// context is short enough that split-K overhead (the combine kernel +
/// partials write/read) costs more than the occupancy win.
pub fn splitk_chunk_size(n_tokens_kv: usize) -> usize {
    // Threshold tuned per the V2.19.b A/B: at n_tokens=128 split-K already
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
    reg: &OpsRegistry,
    stream: &HipStream,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    out: DevicePtr,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    n_tokens_kv: usize,
    scale: f32,
) -> Result<()> {
    // Kernel supports head_dim ∈ {64, 128, 256}; block = head_dim so each
    // thread owns one output lane + one int8 within a Q8_0 block.
    assert!(
        head_dim == 64 || head_dim == 128 || head_dim == 256,
        "attention_decode_q8_kv: head_dim {head_dim} not supported (expected 64, 128, or 256)"
    );
    let module = reg.expect_module("attention_decode_q8_kv")?;
    let kernel = module.kernel("flambeau_attention_decode_q8_kv")?;

    let n_heads_q_i = n_heads_q as i32;
    let n_heads_kv_i = n_heads_kv as i32;
    let head_dim_i = head_dim as i32;
    let n_tokens_i = n_tokens_kv as i32;
    let scale_f = scale;
    let q_ptr: u64 = q.as_usize() as u64;
    let k_ptr: u64 = k_cache.as_usize() as u64;
    let v_ptr: u64 = v_cache.as_usize() as u64;
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
    args.push(&scale_f);
    let cfg = LaunchCfg::one_d(n_heads_q as u32, head_dim as u32);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Split the interleaved `(Q, gate)` output of a gated-attention query
/// projection into two contiguous F16 tensors. Used by Qwen3.5/3.6 full-attn
/// layers, where the `attn_q` weight projects to `2 * head_dim` per head —
/// first half is Q, second half is the output gate. Q goes into the
/// attention kernel; gate is held for a silu-multiply after attention.
///
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

/// MTP-4-C-5: BF16 sibling of [`split_q_gate_f16`]. Same launch shape;
/// no arithmetic, just strided copy.
pub fn split_q_gate_bf16(
    reg: &OpsRegistry,
    stream: &HipStream,
    fused_qg: DevicePtr,
    q_out: DevicePtr,
    gate_out: DevicePtr,
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
) -> Result<()> {
    let module = reg.expect_module("split_q_gate_bf16")?;
    let kernel = module.kernel("flambeau_split_q_gate_bf16")?;

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
///
/// Dispatches:
///   - `attention_prefill_flash_tile_f16` (V2.2.d fix 4 candle port —
///     BR=4 LDS-tiled flash-attn v2) when `n_q_tokens >= 4` and head_dim
///     ∈ {64, 128, 256}. Per-call time on gfx906 is ~5× the previous
///     oracle kernel at pp512 (measured 2026-04-22: 2994 µs → target
///     ≤800 µs).
///   - The per-(q_token, q_head) oracle kernel (`attention_prefill_f16`)
///     otherwise. Shape grid covers the n_q < 4 edge that flash-tile's
///     BR=4 coop-load pattern under-utilises.
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
) -> Result<()> {
    attention_prefill_f16_slots(
        reg, stream, q, k_cache, v_cache, out, n_q_tokens, n_heads_q, n_heads_kv,
        head_dim, n_k_tokens, q_offset, scale, None, None,
    )
}

/// V2.26.a-i4 — graph-captureable variant of [`attention_prefill_f16`].
///
/// Behaviour matches the wrapper in every non-capture case. When the
/// optional [`ScalarSlot`] handles are `Some`, `push_slot` tags their
/// kernel args so the graph recorder can bind the slots to the
/// resulting kernel node. Callers can then replay the captured exec at
/// a different `start_position` by calling `HipGraphExec::set_slot`.
///
/// The relevant pos-varying scalars are:
/// - `n_k_tokens` — total K tokens the causal mask stops at (grows each
///   ubatch as the KV cache fills).
/// - `q_offset` — row offset of the Q block into the global causal grid
///   (equals `start_position`).
///
/// Both kernels (flash_tile + oracle) place these at different arg
/// indices; the slot recorder captures whichever path n_q_tokens
/// selected. Capturing at a particular n_q and replaying at a different
/// n_q is unsupported (different path → different node layout).
///
/// [`ScalarSlot`]: flambeau_backend_hip::ScalarSlot
#[allow(clippy::too_many_arguments)]
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
    n_k_slot: Option<flambeau_backend_hip::ScalarSlot>,
    q_off_slot: Option<flambeau_backend_hip::ScalarSlot>,
) -> Result<()> {
    assert!(
        head_dim == 64 || head_dim == 128 || head_dim == 256,
        "attention_prefill_f16: head_dim {head_dim} not supported (expected 64, 128, or 256)"
    );

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
        push_scalar_maybe_slot(&mut args, &n_k_i, n_k_slot);
        push_scalar_maybe_slot(&mut args, &q_off_i, q_off_slot);
        args.push(&scale_f);
        // V2.29.b — BR depends on which variant we dispatch to.
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

    // Oracle path for n_q < 4 (very short prompts / edge shapes).
    let module = reg.expect_module("attention_prefill_f16")?;
    let kernel = module.kernel("flambeau_attention_prefill_f16")?;
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
    push_scalar_maybe_slot(&mut args, &n_k_i, n_k_slot);
    push_scalar_maybe_slot(&mut args, &q_off_i, q_off_slot);
    args.push(&scale_f);
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
