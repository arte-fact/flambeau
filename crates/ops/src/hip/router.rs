//! MoE router — dense F32 GEMV producing per-expert logits.
//!
//! Only one kernel today. The router weight (`ffn_gate_inp.weight`) is F32
//! in every Qwen3.x GGUF we target, so we don't pay for a quantised path
//! here. Output goes to `moe::topk_f32`.

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
