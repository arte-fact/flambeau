//! Recurrent-layer kernels (Gated-Delta-Net).
//!
//! V1.7.2.F: fused per-(b,h) autoregressive + prefill step kernel. One launch
//! runs the recurrence loop over `L` tokens with the state held in registers,
//! so decode (L=1) and prefill (L>1) share the same dispatch path.

use anyhow::Result;
use flambeau_backend_hip::{HipStream, KernelArgs, LaunchCfg};
use flambeau_core::DevicePtr;

use super::OpsRegistry;

/// Fused GDN step. S_v = 128 (Qwen3.6-35B: `head_k_dim = head_v_dim = 128`).
///
/// Shapes (row-major, contiguous, F32):
/// - `q, k`:      `(B, H_kv, L, 128)` where `H_kv = H / n_rep`
/// - `v`:         `(B, H,    L, 128)`
/// - `gate, beta`: `(B, H,    L)` — one scalar per (b, h, t)
/// - `state_in`:  `(B, H, 128, 128)` stored col-outer (see kernel).
/// - `state_out`: same shape and layout; may alias `state_in` (the kernel
///   loads state_in into registers at entry and writes state_out at exit).
/// - `attn_out`:  `(B, H, L, 128)`
///
/// `n_rep = H_v / H_kv` drives implicit GQA broadcast for Q/K (no caller-side
/// expand). `n_rep = 1` reduces to the no-GQA path.
///
/// Launch: grid `(H, B, ceil(S_v / warps_per_block))`, block
/// `(WARP_SIZE=64, 4, 1)` — 4 warps per block, each warp owns one output
/// column. `warps_per_block = 4` ⇒ grid_z = `S_v / 4 = 32` at S_v=128.
#[allow(clippy::too_many_arguments)]
pub fn gdn_state_step_f32_s128(
    reg: &OpsRegistry,
    stream: &HipStream,
    q: DevicePtr,
    k: DevicePtr,
    v: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    state_in: DevicePtr,
    state_out: DevicePtr,
    attn_out: DevicePtr,
    b: usize,
    h_v: usize,
    l: usize,
    n_rep: usize,
) -> Result<()> {
    const S_V: u32 = 128;
    const WARP_SIZE: u32 = 64;
    const WARPS_PER_BLOCK: u32 = 4;

    let module = reg.expect_module("gdn_state_step_f32")?;
    let kernel = module.kernel("flambeau_gdn_state_step_f32_s128")?;

    let b_i = b as i32;
    let h_i = h_v as i32;
    let l_i = l as i32;
    let n_rep_i = n_rep as i32;
    let q_ptr: u64 = q.as_usize() as u64;
    let k_ptr: u64 = k.as_usize() as u64;
    let v_ptr: u64 = v.as_usize() as u64;
    let gate_ptr: u64 = gate.as_usize() as u64;
    let beta_ptr: u64 = beta.as_usize() as u64;
    let sin_ptr: u64 = state_in.as_usize() as u64;
    let sout_ptr: u64 = state_out.as_usize() as u64;
    let ao_ptr: u64 = attn_out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&q_ptr);
    args.push(&k_ptr);
    args.push(&v_ptr);
    args.push(&gate_ptr);
    args.push(&beta_ptr);
    args.push(&sin_ptr);
    args.push(&sout_ptr);
    args.push(&ao_ptr);
    args.push(&b_i);
    args.push(&h_i);
    args.push(&l_i);
    args.push(&n_rep_i);

    let grid_z = S_V / WARPS_PER_BLOCK;
    let cfg = LaunchCfg {
        grid: (h_v as u32, b as u32, grid_z),
        block: (WARP_SIZE, WARPS_PER_BLOCK, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Fused GDN α/β/gate compute over `[num_v_heads]` F32 elements:
///   gate_out[i] = softplus(alpha_in[i] + ssm_dt_bias[i]) * ssm_a[i]
///   beta_out[i] = sigmoid(beta_in[i])
///
/// Replaces V1.7.3-c2's host-roundtrip placeholder. One block of 64 threads
/// covers Qwen3.6's `num_v_heads = 32`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_alpha_beta_f32(
    reg: &OpsRegistry,
    stream: &HipStream,
    alpha_in: DevicePtr,
    beta_in: DevicePtr,
    ssm_dt_bias: DevicePtr,
    ssm_a: DevicePtr,
    gate_out: DevicePtr,
    beta_out: DevicePtr,
    n: usize,
) -> Result<()> {
    let module = reg.expect_module("gdn_alpha_beta_f32")?;
    let kernel = module.kernel("flambeau_gdn_alpha_beta_f32")?;
    let n_i = n as i32;
    let a_ptr: u64 = alpha_in.as_usize() as u64;
    let b_ptr: u64 = beta_in.as_usize() as u64;
    let dt_ptr: u64 = ssm_dt_bias.as_usize() as u64;
    let sa_ptr: u64 = ssm_a.as_usize() as u64;
    let g_ptr: u64 = gate_out.as_usize() as u64;
    let bo_ptr: u64 = beta_out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&a_ptr);
    args.push(&b_ptr);
    args.push(&dt_ptr);
    args.push(&sa_ptr);
    args.push(&g_ptr);
    args.push(&bo_ptr);
    args.push(&n_i);
    let cfg = LaunchCfg::one_d((n as u32).div_ceil(64), 64);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}
