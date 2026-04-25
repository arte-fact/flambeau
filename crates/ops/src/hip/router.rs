//! MoE router — dense F32 GEMV producing per-expert logits.
//!
//! Only one kernel today. The router weight (`ffn_gate_inp.weight`) is F32
//! in every Qwen3.x GGUF we target, so we don't pay for a quantised path
//! here. Output goes to `moe::topk_f32`.

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

/// Dense GEMV: `y[n] = Σ_k w[n, k] * (float) x[k]`. Weight is F32
/// `[n_rows, k]` (outermost-first, contiguous innermost k); activation is
/// F16 `[k]`; output is F32 `[n_rows]`.
///
/// Launch: one block per output row, 256 threads (4 wave64).
pub fn dense_gemv_f32_f16(
    reg: &OpsRegistry,
    stream: &HipStream,
    w: DevicePtr,
    x: DevicePtr,
    y: DevicePtr,
    n_rows: usize,
    k: usize,
) -> Result<()> {
    let module = reg.expect_module("dense_gemv_f32_f16")?;
    let kernel = module.kernel("flambeau_dense_gemv_f32_f16")?;
    let n_rows_i = n_rows as i32;
    let k_i = k as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let x_ptr: u64 = x.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&x_ptr);
    args.push(&y_ptr);
    args.push(&n_rows_i);
    args.push(&k_i);
    let cfg = LaunchCfg::one_d(n_rows as u32, 256);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// V2.31.g — batched dense GEMV: `y[t, n] = Σ_k w[n, k] * (float) x[t, k]`.
/// Weight F32 `[n_rows, k]`; activation F16 `[n_tokens, k]`; output F32
/// `[n_tokens, n_rows]`.
///
/// Launch: `gridDim = (n_rows, n_tokens)`, one block per (row, token) pair.
/// Collapses a caller-side `for t in 0..n_tokens { gemv_single }` loop
/// into a single kernel launch — saves ~L µs launch overhead per call at
/// the cost of a 2D grid.
pub fn dense_gemv_f32_f16_batched(
    reg: &OpsRegistry,
    stream: &HipStream,
    w: DevicePtr,
    x: DevicePtr,
    y: DevicePtr,
    n_rows: usize,
    k: usize,
    n_tokens: usize,
) -> Result<()> {
    let module = reg.expect_module("dense_gemv_f32_f16_batched")?;
    let kernel = module.kernel("flambeau_dense_gemv_f32_f16_batched")?;
    let n_rows_i = n_rows as i32;
    let k_i = k as i32;
    let n_tokens_i = n_tokens as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let x_ptr: u64 = x.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&x_ptr);
    args.push(&y_ptr);
    args.push(&n_rows_i);
    args.push(&k_i);
    args.push(&n_tokens_i);
    let cfg = LaunchCfg {
        grid: (n_rows as u32, n_tokens as u32, 1),
        block: (256, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}
