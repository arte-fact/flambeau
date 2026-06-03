//! Positional encoding — RoPE (interleaved-pair, F16, in-place).

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

/// Apply interleaved-pair RoPE in-place to a `(n_tokens, n_heads, head_dim)`
/// F16 tensor. Pairs are `(x[2i], x[2i+1])` — used by dense Qwen3 / Gemma-4.
/// For Qwen3.5/3.6 full-attention layers (NeoX-split partial), use
/// [`rope_neox_partial_f16`] instead.
/// Grid: `(n_tokens, n_heads)`. Block: `head_dim / 2` threads.
pub fn rope_f16(
    reg: &OpsRegistry,
    stream: &HipStream,
    x: DevicePtr,
    positions: DevicePtr,
    theta_base: f32,
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
) -> Result<()> {
    assert_eq!(head_dim % 2, 0, "rope_f16 expects head_dim % 2 == 0");
    let module = reg.expect_module("rope_f16")?;
    let kernel = module.kernel("flambeau_rope_f16")?;

    let n_heads_i = n_heads as i32;
    let head_dim_i = head_dim as i32;
    let theta = theta_base;
    let x_ptr: u64 = x.as_usize() as u64;
    let p_ptr: u64 = positions.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&x_ptr);
    args.push(&p_ptr);
    args.push(&theta);
    args.push(&n_heads_i);
    args.push(&head_dim_i);
    let cfg = LaunchCfg {
        grid: (n_tokens as u32, n_heads as u32, 1),
        block: ((head_dim / 2) as u32, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// NeoX-style partial RoPE, in-place. Rotates only the first `rotated_dims`
/// of each `head_dim`; dims `rotated_dims..head_dim` pass through unchanged.
/// Pair layout is split — `(x[i], x[i + rotated_dims/2])` — matching
/// llama.cpp's `rope_multi` path for text-only MROPE (all sections pointing
/// at the same position ID).
/// Used by Qwen3.5 / Qwen3.6 / Qwen3-Next full-attention layers.
/// `rotated_dims` must be even and ≤ `head_dim`.
/// Grid: `(n_tokens, n_heads)`. Block: `rotated_dims / 2` threads.
pub fn rope_neox_partial_f16(
    reg: &OpsRegistry,
    stream: &HipStream,
    x: DevicePtr,
    positions: DevicePtr,
    theta_base: f32,
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    rotated_dims: usize,
) -> Result<()> {
    assert_eq!(
        rotated_dims % 2,
        0,
        "rope_neox_partial_f16 expects rotated_dims % 2 == 0"
    );
    assert!(
        rotated_dims <= head_dim,
        "rotated_dims ({rotated_dims}) must fit in head_dim ({head_dim})"
    );
    let module = reg.expect_module("rope_neox_partial_f16")?;
    let kernel = module.kernel("flambeau_rope_neox_partial_f16")?;

    let n_heads_i = n_heads as i32;
    let head_dim_i = head_dim as i32;
    let rotated_dims_i = rotated_dims as i32;
    let theta = theta_base;
    let x_ptr: u64 = x.as_usize() as u64;
    let p_ptr: u64 = positions.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&x_ptr);
    args.push(&p_ptr);
    args.push(&theta);
    args.push(&n_heads_i);
    args.push(&head_dim_i);
    args.push(&rotated_dims_i);
    let cfg = LaunchCfg {
        grid: (n_tokens as u32, n_heads as u32, 1),
        block: ((rotated_dims / 2) as u32, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Fused per-head rmsnorm + partial NeoX RoPE, F16 in-place. Replaces
/// (`rmsnorm_f16` → DtoD memcpy back → `rope_neox_partial_f16`) for
/// gemma4 Q-norm and K-norm paths. Saves 2 launches + 1 DtoD memcpy
/// per call. `head_dim` must be 64 / 128 / 256 / 512 (one wrapper per
/// supported width); `rotated_dims` ≤ `head_dim`.
pub fn rmsnorm_rope_neox_partial_f16(
    reg: &OpsRegistry,
    stream: &HipStream,
    x: DevicePtr,
    norm_w: DevicePtr,
    positions: DevicePtr,
    theta_base: f32,
    eps: f32,
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    rotated_dims: usize,
) -> Result<()> {
    assert_eq!(
        rotated_dims % 2,
        0,
        "rmsnorm_rope_neox_partial_f16 expects rotated_dims % 2 == 0"
    );
    assert!(
        rotated_dims <= head_dim,
        "rotated_dims ({rotated_dims}) must fit in head_dim ({head_dim})"
    );
    let entry = match head_dim {
        64 => "flambeau_rmsnorm_rope_neox_partial_f16_d64",
        128 => "flambeau_rmsnorm_rope_neox_partial_f16_d128",
        256 => "flambeau_rmsnorm_rope_neox_partial_f16_d256",
        512 => "flambeau_rmsnorm_rope_neox_partial_f16_d512",
        other => anyhow::bail!(
            "rmsnorm_rope_neox_partial_f16: head_dim {other} not in {{64, 128, 256, 512}}"
        ),
    };
    let module = reg.expect_module("rmsnorm_rope_neox_partial_f16")?;
    let kernel = module.kernel(entry)?;

    let n_heads_i = n_heads as i32;
    let rotated_dims_i = rotated_dims as i32;
    let theta = theta_base;
    let eps_f = eps;
    let x_ptr: u64 = x.as_usize() as u64;
    let w_ptr: u64 = norm_w.as_usize() as u64;
    let p_ptr: u64 = positions.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&x_ptr);
    args.push(&w_ptr);
    args.push(&p_ptr);
    args.push(&n_heads_i);
    args.push(&rotated_dims_i);
    args.push(&theta);
    args.push(&eps_f);
    let cfg = LaunchCfg {
        grid: (n_tokens as u32, n_heads as u32, 1),
        block: (head_dim as u32, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}
