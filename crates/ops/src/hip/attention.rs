//! Attention — decode (F16 KV + Q8_0 KV) and prefill (F16 KV).
//!
//! GQA is handled via the `n_heads_kv` argument; the kernel broadcasts one
//! KV head across `n_heads_q / n_heads_kv` Q heads.

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
    // Kernel-side shared memory is sized to head_dim=256, block = head_dim.
    // Any multiple of wave64 ≤ 256 works (1 / 2 / 4 warps cover 64 / 128 /
    // 256). Larger head_dims need a kernel-side rework (wider block or
    // per-thread strides) — add the cert shape first, then relax this guard.
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
    args.push(&n_tokens_i);
    args.push(&scale_f);
    let cfg = LaunchCfg::one_d(n_heads_q as u32, head_dim as u32);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
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

/// Prefill attention, F16 KV. Computes `n_q_tokens` Q rows against
/// `n_k_tokens` KV rows with causal masking (`q_token_i` attends to
/// `k_token_0..k_token_{q_offset + i}`).
///
/// Launch: `(n_q_tokens, n_heads_q)`, 128 threads/block.
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
    // Kernel supports head_dim ∈ {64, 128, 256}; block = head_dim.
    assert!(
        head_dim == 64 || head_dim == 128 || head_dim == 256,
        "attention_prefill_f16: head_dim {head_dim} not supported (expected 64, 128, or 256)"
    );
    let module = reg.expect_module("attention_prefill_f16")?;
    let kernel = module.kernel("flambeau_attention_prefill_f16")?;

    let n_q_i = n_q_tokens as i32;
    let n_heads_q_i = n_heads_q as i32;
    let n_heads_kv_i = n_heads_kv as i32;
    let head_dim_i = head_dim as i32;
    let n_k_i = n_k_tokens as i32;
    let q_off_i = q_offset as i32;
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
    args.push(&n_q_i);
    args.push(&n_heads_q_i);
    args.push(&n_heads_kv_i);
    args.push(&head_dim_i);
    args.push(&n_k_i);
    args.push(&q_off_i);
    args.push(&scale_f);
    let cfg = LaunchCfg {
        grid: (n_q_tokens as u32, n_heads_q as u32, 1),
        block: (head_dim as u32, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}
