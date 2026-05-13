//! Final-logit softcap — `y = tanh(x / cap) * cap`. Used by Gemma4 on
//! the LM-head logits (hparams.f_final_logit_softcapping = 30). In-place
//! safe: `x` and `y` may alias.

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "op wrapper — the single unsafe block is a kernel.launch over device pointers \
              validated by the caller; the kernel stem + entry resolve through the registry."
)]

use anyhow::Result;
use flambeau_backend_hip::{HipStream, KernelArgs, LaunchCfg};
use flambeau_core::DevicePtr;

use super::OpsRegistry;

const SOFTCAP_THREADS: u32 = 256;

pub fn apply_softcap_f32(
    reg: &OpsRegistry,
    stream: &HipStream,
    x: DevicePtr,
    y: DevicePtr,
    n: usize,
    cap: f32,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let module = reg.expect_module("apply_softcap_f32")?;
    let kernel = module.kernel("flambeau_apply_softcap_f32")?;

    let n_i = n as i32;
    let cap_f = cap;
    let x_ptr: u64 = x.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&x_ptr);
    args.push(&y_ptr);
    args.push(&n_i);
    args.push(&cap_f);

    let grid = (n as u32).div_ceil(SOFTCAP_THREADS);
    let cfg = LaunchCfg::one_d(grid, SOFTCAP_THREADS);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}
