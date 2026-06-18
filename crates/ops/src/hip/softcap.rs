//! Final-logit softcap — `y = tanh(x / cap) * cap`. Used by Gemma4 on
//! the LM-head logits (hparams.f_final_logit_softcapping = 30). In-place
//! safe: `x` and `y` may alias.

use anyhow::Result;
use flambeau_core::{LaunchArgs, LaunchCfg};
use flambeau_core::DevicePtr;


const SOFTCAP_THREADS: u32 = 256;

pub fn apply_softcap_f32<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    x: DevicePtr,
    y: DevicePtr,
    n: usize,
    cap: f32,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let module = reg.expect_module("apply_softcap_f32")?;
    let kernel = B::kernel(module, "flambeau_apply_softcap_f32")?;

    let n_i = n as i32;
    let cap_f = cap;
    let x_ptr: u64 = x.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&x_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&n_i);
    args.push_arg(&cap_f);

    let grid = (n as u32).div_ceil(SOFTCAP_THREADS);
    let cfg = LaunchCfg::one_d(grid, SOFTCAP_THREADS);
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}
