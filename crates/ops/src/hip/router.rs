//! MoE router — dense F32 GEMV producing per-expert logits.
//! Only one kernel today. The router weight (`ffn_gate_inp.weight`) is F32
//! in every Qwen3.x GGUF we target, so we don't pay for a quantised path
//! here. Output goes to `moe::topk_f32`.

use anyhow::Result;
use flambeau_core::{LaunchArgs, LaunchCfg};
use flambeau_core::DevicePtr;


/// Dense GEMV: `y[n] = Σ_k w[n, k] * (float) x[k]`. Weight is F32
/// `[n_rows, k]` (outermost-first, contiguous innermost k); activation is
/// F16 `[k]`; output is F32 `[n_rows]`.
/// Launch: one block per output row, 256 threads (4 wave64).
pub fn dense_gemv_f32_f16<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    w: DevicePtr,
    x: DevicePtr,
    y: DevicePtr,
    n_rows: usize,
    k: usize,
) -> Result<()> {
    let module = reg.expect_module("dense_gemv_f32_f16")?;
    let kernel = B::kernel(module, "flambeau_dense_gemv_f32_f16")?;
    let n_rows_i = n_rows as i32;
    let k_i = k as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let x_ptr: u64 = x.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&x_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&k_i);
    let cfg = LaunchCfg::one_d(n_rows as u32, 256);
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// F16-weight variant of [`dense_gemv_f32_f16`].
/// Halves HBM weight bandwidth (2 B/elem vs 4 B/elem). Used for the MoE
/// router on models where the F32→F16 router-weight conversion is
/// applied at load time (see `sharded.rs::up_router_f16` / equivalent).
/// Same launch shape and reduction semantics as the F32 variant.
pub fn dense_gemv_f16_f16<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    w: DevicePtr,
    x: DevicePtr,
    y: DevicePtr,
    n_rows: usize,
    k: usize,
) -> Result<()> {
    let module = reg.expect_module("dense_gemv_f16_f16")?;
    let kernel = B::kernel(module, "flambeau_dense_gemv_f16_f16")?;
    let n_rows_i = n_rows as i32;
    let k_i = k as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let x_ptr: u64 = x.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&x_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&k_i);
    let cfg = LaunchCfg::one_d(n_rows as u32, 256);
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// F16-weight variant of [`dense_gemv_f32_f16_batched`].
pub fn dense_gemv_f16_f16_batched<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    bufs: crate::DenseGemvBatchedBuffers,
    shape: crate::DenseGemvBatchedShape,
) -> Result<()> {
    let crate::DenseGemvBatchedBuffers { w, x, y } = bufs;
    let crate::DenseGemvBatchedShape { n_rows, k, n_tokens } = shape;
    let module = reg.expect_module("dense_gemv_f16_f16_batched")?;
    let kernel = B::kernel(module, "flambeau_dense_gemv_f16_f16_batched")?;
    let n_rows_i = n_rows as i32;
    let k_i = k as i32;
    let n_tokens_i = n_tokens as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let x_ptr: u64 = x.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&x_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&k_i);
    args.push_arg(&n_tokens_i);
    let cfg = LaunchCfg {
        grid: (n_rows as u32, n_tokens as u32, 1),
        block: (256, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Batched dense GEMV: `y[t, n] = Σ_k w[n, k] * (float) x[t, k]`.
/// Weight F32 `[n_rows, k]`; activation F16 `[n_tokens, k]`; output F32
/// `[n_tokens, n_rows]`.
/// Launch: `gridDim = (n_rows, n_tokens)`, one block per (row, token) pair.
/// Collapses a caller-side `for t in 0..n_tokens { gemv_single }` loop
/// into a single kernel launch — saves ~L µs launch overhead per call at
/// the cost of a 2D grid.
pub fn dense_gemv_f32_f16_batched<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    bufs: crate::DenseGemvBatchedBuffers,
    shape: crate::DenseGemvBatchedShape,
) -> Result<()> {
    let crate::DenseGemvBatchedBuffers { w, x, y } = bufs;
    let crate::DenseGemvBatchedShape { n_rows, k, n_tokens } = shape;
    let module = reg.expect_module("dense_gemv_f32_f16_batched")?;
    let kernel = B::kernel(module, "flambeau_dense_gemv_f32_f16_batched")?;
    let n_rows_i = n_rows as i32;
    let k_i = k as i32;
    let n_tokens_i = n_tokens as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let x_ptr: u64 = x.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&w_ptr);
    args.push_arg(&x_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&n_rows_i);
    args.push_arg(&k_i);
    args.push_arg(&n_tokens_i);
    let cfg = LaunchCfg {
        grid: (n_rows as u32, n_tokens as u32, 1),
        block: (256, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}
