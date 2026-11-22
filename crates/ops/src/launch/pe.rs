//! Positional encoding — RoPE (interleaved-pair, F16, in-place).

use anyhow::Result;
use flambeau_core::{LaunchArgs, LaunchCfg};

use crate::sig::{RopeBuffers, RopeFusedBuffers, RopePartialShape, RopeShape};

/// Apply interleaved-pair RoPE in-place to a `(n_tokens, n_heads, head_dim)`
/// F16 tensor. Pairs are `(x[2i], x[2i+1])` — used by dense Qwen3 / Gemma-4.
/// For Qwen3.5/3.6 full-attention layers (NeoX-split partial), use
/// [`rope_neox_partial_f16`] instead.
/// Grid: `(n_tokens, n_heads)`. Block: `head_dim / 2` threads.
pub fn rope_f16<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buf: RopeBuffers,
    shape: RopeShape,
    theta_base: f32,
) -> Result<()> {
    assert_eq!(shape.head_dim % 2, 0, "rope_f16 expects head_dim % 2 == 0");
    let module = reg.expect_module("rope_f16")?;
    let kernel = B::kernel(module, "flambeau_rope_f16")?;

    let n_heads_i = shape.n_heads as i32;
    let head_dim_i = shape.head_dim as i32;
    let x_ptr: u64 = buf.x.as_usize() as u64;
    let p_ptr: u64 = buf.positions.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&x_ptr);
    args.push_arg(&p_ptr);
    args.push_arg(&theta_base);
    args.push_arg(&n_heads_i);
    args.push_arg(&head_dim_i);
    let cfg = LaunchCfg {
        grid: (shape.n_tokens as u32, shape.n_heads as u32, 1),
        block: ((shape.head_dim / 2) as u32, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// NeoX-style partial RoPE, in-place. Rotates only the first
/// `shape.rotated_dims` of each `shape.head_dim`; dims
/// `rotated_dims..head_dim` pass through unchanged. Pair layout is split —
/// `(x[i], x[i + rotated_dims/2])` — matching llama.cpp's `rope_multi`
/// path for text-only MROPE (all sections pointing at the same position
/// ID). Used by Qwen3.5 / Qwen3.6 / Qwen3-Next full-attention layers.
/// `rotated_dims` must be even and ≤ `head_dim`.
/// Grid: `(n_tokens, n_heads)`. Block: `rotated_dims / 2` threads.
pub fn rope_neox_partial_f16<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buf: RopeBuffers,
    shape: RopePartialShape,
    theta_base: f32,
) -> Result<()> {
    assert_eq!(
        shape.rotated_dims % 2,
        0,
        "rope_neox_partial_f16 expects rotated_dims % 2 == 0"
    );
    assert!(
        shape.rotated_dims <= shape.head_dim,
        "rotated_dims ({}) must fit in head_dim ({})",
        shape.rotated_dims,
        shape.head_dim
    );
    let module = reg.expect_module("rope_neox_partial_f16")?;
    let kernel = B::kernel(module, "flambeau_rope_neox_partial_f16")?;

    let n_heads_i = shape.n_heads as i32;
    let head_dim_i = shape.head_dim as i32;
    let rotated_dims_i = shape.rotated_dims as i32;
    let x_ptr: u64 = buf.x.as_usize() as u64;
    let p_ptr: u64 = buf.positions.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&x_ptr);
    args.push_arg(&p_ptr);
    args.push_arg(&theta_base);
    args.push_arg(&n_heads_i);
    args.push_arg(&head_dim_i);
    args.push_arg(&rotated_dims_i);
    let cfg = LaunchCfg {
        grid: (shape.n_tokens as u32, shape.n_heads as u32, 1),
        block: ((shape.rotated_dims / 2) as u32, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Fused per-head rmsnorm + partial NeoX RoPE, F16 in-place. Replaces
/// (`rmsnorm_f16` → DtoD memcpy back → `rope_neox_partial_f16`) for
/// gemma4 Q-norm and K-norm paths. Saves 2 launches + 1 DtoD memcpy
/// per call. `shape.head_dim` must be 64 / 128 / 256 / 512 (one wrapper
/// per supported width); `shape.rotated_dims` ≤ `shape.head_dim`.
pub fn rmsnorm_rope_neox_partial_f16<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    buf: RopeFusedBuffers,
    shape: RopePartialShape,
    theta_base: f32,
    eps: f32,
) -> Result<()> {
    assert_eq!(
        shape.rotated_dims % 2,
        0,
        "rmsnorm_rope_neox_partial_f16 expects rotated_dims % 2 == 0"
    );
    assert!(
        shape.rotated_dims <= shape.head_dim,
        "rotated_dims ({}) must fit in head_dim ({})",
        shape.rotated_dims,
        shape.head_dim
    );
    let entry = match shape.head_dim {
        64 => "flambeau_rmsnorm_rope_neox_partial_f16_d64",
        128 => "flambeau_rmsnorm_rope_neox_partial_f16_d128",
        256 => "flambeau_rmsnorm_rope_neox_partial_f16_d256",
        512 => "flambeau_rmsnorm_rope_neox_partial_f16_d512",
        other => anyhow::bail!(
            "rmsnorm_rope_neox_partial_f16: head_dim {other} not in {{64, 128, 256, 512}}"
        ),
    };
    let module = reg.expect_module("rmsnorm_rope_neox_partial_f16")?;
    let kernel = B::kernel(module, entry)?;

    let n_heads_i = shape.n_heads as i32;
    let rotated_dims_i = shape.rotated_dims as i32;
    let x_ptr: u64 = buf.x.as_usize() as u64;
    let w_ptr: u64 = buf.norm_w.as_usize() as u64;
    let p_ptr: u64 = buf.positions.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&x_ptr);
    args.push_arg(&w_ptr);
    args.push_arg(&p_ptr);
    args.push_arg(&n_heads_i);
    args.push_arg(&rotated_dims_i);
    args.push_arg(&theta_base);
    args.push_arg(&eps);
    let cfg = LaunchCfg {
        grid: (shape.n_tokens as u32, shape.n_heads as u32, 1),
        block: (shape.head_dim as u32, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}
