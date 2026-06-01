//! RoPE correctness sweep.
//! Two-part cert:
//! 1. **Round-trip**: rotate with positions `p`, then rotate again with
//! positions `-p` — should land back at the input modulo F16 round-off.
//! This catches axis-flip bugs, wrong pair grouping, etc.
//! 2. **Fixed-angle oracle**: positions = 1, theta_base = 10000, compare
//! against a CPU F32 reference. This catches the actual angle formula.
//! Shapes: Qwen3.6's head_dim = 128, n_heads_q = 32, n_heads_kv = 4.
//! Sequence lengths from decode (1) through short prefill (128).

#![cfg(feature = "hip")]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "sweep harness — every unsafe block is a kernel launch or a memcpy_async \
              over buffers allocated locally in the same function and freed before \
              return; invariant is uniform across all sites."
)]

use std::path::Path;

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::{
    device_count, FuncAttributes, HipDevice, HipKernel, HipModule, KernelArgs, LaunchCfg,
};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;
use half::f16;

use crate::cert::{now_utc_iso8601, Cert, PmcSnapshot, ShapeResult, SCHEMA_VERSION};
use crate::harness::{alloc_and_upload, max_rel_err_with_floor, rig, seeded_f32_range};

pub fn run_sweep(repo_root: &Path) -> Result<Cert> {
    let n = device_count().context("hipGetDeviceCount")?;
    if n < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;

    let kb = kernels::hsaco("rope_f16").ok_or_else(|| anyhow::anyhow!("rope_f16 not compiled"))?;
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_rope_f16")?;
    let attrs: FuncAttributes = kernel.attributes()?;

    // Qwen3.6 shapes: head_dim = 128. Seq × heads:
    // (1, 32) — decode, Q.
    // (1, 4) — decode, KV.
    // (128, 32) — short prefill, Q.
    // (128, 4) — short prefill, KV.
    let shapes = [(1usize, 32usize), (1, 4), (128, 32), (128, 4)];
    let head_dim = 128usize;
    let theta_base = 10000.0f32;

    let mut results = Vec::new();
    for (n_tokens, n_heads) in shapes {
        let seed = 0xC0FFEE ^ ((n_tokens as u64) * 1009 + (n_heads as u64) * 31);
        let max_rel_err =
            run_shape_cert(&dev, &kernel, n_tokens, n_heads, head_dim, theta_base, seed)?;
        // Two-step cert (round-trip + angle-1 oracle); both should agree to
        // within F16 round-off which is ~1e-3 for well-bounded inputs.
        let tol = 5e-3;
        results.push(ShapeResult {
            m: n_tokens,
            k: n_heads,
            n: head_dim,
            seed,
            max_rel_err,
            tolerance: tol,
            pass: max_rel_err <= tol,
        });
        tracing::info!(
            target: "flambeau_bench::sweep_rope",
            n_tokens, n_heads, head_dim, max_rel_err, tol,
            "rope shape"
        );
    }

    let pass = results.iter().all(|r| r.pass);
    let rig = rig();
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "rope_f16_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "rope".to_string(),
        dtype_weight: "F16".to_string(),
        dtype_activation: "F16".to_string(),
        tolerance_formula: "|err| <= 5e-3 * max(|ref|, 1)  (F16 round-off floor)".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig,
        pmc: Some(PmcSnapshot {
            vgpr_count: Some(attrs.num_regs),
            sgpr_count: None,
            waves_per_simd: Some(attrs.gfx906_waves_per_simd()),
            mem_busy_pct: None,
            valu_busy_pct: None,
        }),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

fn run_shape_cert(
    dev: &HipDevice,
    kernel: &HipKernel<'_>,
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    theta_base: f32,
    seed: u64,
) -> Result<f32> {
    assert_eq!(head_dim % 2, 0);

    // Generate random F16 input, in-range for F16 precision.
    let total = n_tokens * n_heads * head_dim;
    let x0_f32 = seeded_f32_range(seed, total, -0.5, 0.5);
    let x0_f16: Vec<f16> = x0_f32.iter().map(|v| f16::from_f32(*v)).collect();

    // Positions: sequential, one per token.
    let positions: Vec<i32> = (0..n_tokens as i32).collect();

    // --- Oracle: CPU reference at positions ---
    let mut ref_rotated = vec![0.0f32; total];
    for (t, &pos) in positions.iter().enumerate().take(n_tokens) {
        for h in 0..n_heads {
            for pair in 0..(head_dim / 2) {
                let base = (t * n_heads + h) * head_dim + 2 * pair;
                let x0 = x0_f16[base].to_f32();
                let x1 = x0_f16[base + 1].to_f32();
                let inv_freq = 1.0f32 / theta_base.powf(2.0 * (pair as f32) / (head_dim as f32));
                let angle = (pos as f32) * inv_freq;
                let c = angle.cos();
                let s = angle.sin();
                ref_rotated[base] = f16::from_f32(x0 * c - x1 * s).to_f32();
                ref_rotated[base + 1] = f16::from_f32(x0 * s + x1 * c).to_f32();
            }
        }
    }

    // --- GPU path 1: rotate ---
    let d_x = alloc_and_upload(dev, &x0_f16);
    let d_pos = alloc_and_upload(dev, &positions);

    launch_rope(
        dev, kernel, d_x, d_pos, theta_base, n_tokens, n_heads, head_dim,
    )?;

    // Copy out the rotated tensor for the oracle check.
    let mut got1 = vec![f16::from_f32(0.0); total];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got1.as_mut_ptr() as usize),
            d_x,
            total * 2,
        )?;
    }
    dev.default_stream().synchronize()?;
    let got1_f32: Vec<f32> = got1.iter().map(|v| v.to_f32()).collect();
    let oracle_err = max_rel_err_with_floor(&got1_f32, &ref_rotated, 1.0);

    // --- GPU path 2: round-trip (rotate by negative positions, should
    // return to x0 modulo F16 noise).
    let neg_positions: Vec<i32> = positions.iter().map(|p| -p).collect();
    let d_neg_pos = alloc_and_upload(dev, &neg_positions);
    launch_rope(
        dev, kernel, d_x, d_neg_pos, theta_base, n_tokens, n_heads, head_dim,
    )?;

    let mut got2 = vec![f16::from_f32(0.0); total];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got2.as_mut_ptr() as usize),
            d_x,
            total * 2,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_x, total * 2)?;
        dev.dealloc(d_pos, n_tokens * 4)?;
        dev.dealloc(d_neg_pos, n_tokens * 4)?;
    }

    let got2_f32: Vec<f32> = got2.iter().map(|v| v.to_f32()).collect();
    let roundtrip_err = max_rel_err_with_floor(&got2_f32, &x0_f32, 1.0);

    // Report the worse of the two so one cert row catches both failure modes.
    Ok(oracle_err.max(roundtrip_err))
}

fn launch_rope(
    dev: &HipDevice,
    kernel: &HipKernel<'_>,
    d_x: DevicePtr,
    d_pos: DevicePtr,
    theta_base: f32,
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
) -> Result<()> {
    let stream = dev.default_stream();
    let theta = theta_base;
    let n_heads_i = n_heads as i32;
    let head_dim_i = head_dim as i32;
    let d_x_ptr: u64 = d_x.as_usize() as u64;
    let d_pos_ptr: u64 = d_pos.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&d_x_ptr);
    args.push(&d_pos_ptr);
    args.push(&theta);
    args.push(&n_heads_i);
    args.push(&head_dim_i);
    let cfg = LaunchCfg {
        grid: (n_tokens as u32, n_heads as u32, 1),
        block: ((head_dim / 2) as u32, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    stream.synchronize()?;
    Ok(())
}
