//! 1D convolutions used inside recurrent layers (GDN).
//! Only one kernel today: depthwise causal conv1d. Gated-Delta-Net uses it
//! between the QKV input projection and the silu activation.

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "op wrapper — every unsafe block is a kernel.launch or memcpy_async over \
              DevicePtrs validated by the caller; the kernel stem + entry are resolved \
              through the validated registry and the ABI matches the kernels extern-C \
              signature."
)]

use anyhow::Result;
use flambeau_backend_hip::{KernelArgs, LaunchCfg};


/// Depthwise causal conv1d. `conv_input[n_total, conv_channels]` is the
/// caller-prepared concat of `conv_kernel - 1` history tokens and `n_new`
/// new tokens, so `n_total = (conv_kernel - 1) + n_new`. Output is
/// `y[n_new, conv_channels]` F32. Each channel runs independently.
/// Launch: `(ceil(conv_channels/256), n_new)` blocks × 256 threads.
pub fn causal_conv1d_f32(
    ctx: crate::OpCtx<'_>,
    bufs: crate::ConvCausal1dBuffers,
    shape: crate::ConvCausal1dShape,
) -> Result<()> {
    let crate::OpCtx { reg, stream } = ctx;
    let crate::ConvCausal1dBuffers { conv_input, weight, y } = bufs;
    let crate::ConvCausal1dShape { n_new, conv_channels, conv_kernel } = shape;
    let module = reg.expect_module("causal_conv1d_f32")?;
    let kernel = module.kernel("flambeau_causal_conv1d_f32")?;
    let n_new_i = n_new as i32;
    let cc_i = conv_channels as i32;
    let ck_i = conv_kernel as i32;
    let x_ptr: u64 = conv_input.as_usize() as u64;
    let w_ptr: u64 = weight.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&x_ptr);
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&n_new_i);
    args.push(&cc_i);
    args.push(&ck_i);
    let threads = 256u32;
    let grid_x = (conv_channels as u32).div_ceil(threads);
    let cfg = LaunchCfg {
        grid: (grid_x, n_new as u32, 1),
        block: (threads, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}
