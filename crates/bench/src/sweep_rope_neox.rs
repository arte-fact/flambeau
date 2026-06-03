//! RoPE NeoX-partial correctness sweep.
//! Two-part cert (same shape as `sweep_rope`):
//! 1. Round-trip: rotate with positions `p`, then with `-p` — modulo F16 noise.
//! 2. Fixed-angle oracle: compare to CPU F32 reference at sequential positions.
//! Shapes cover the Qwen3.6 full-attention layer:
//! head_dim=256, rotated_dims=64, theta_base=1e7, n_heads_q=16, n_heads_kv=2.

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

    let kb = kernels::hsaco("rope_neox_partial_f16")
        .ok_or_else(|| anyhow::anyhow!("rope_neox_partial_f16 not compiled"))?;
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_rope_neox_partial_f16")?;
    let attrs: FuncAttributes = kernel.attributes()?;

    // (n_tokens, n_heads). Qwen3.6: 16 Q heads, 2 KV heads; head_dim=256,
    // rotated_dims=64, theta_base=1e7.
    let shapes = [(1usize, 16usize), (1, 2), (128, 16), (128, 2)];
    let head_dim = 256usize;
    let rotated_dims = 64usize;
    let theta_base = 10_000_000.0f32;

    let mut results = Vec::new();
    for (n_tokens, n_heads) in shapes {
        let seed = 0xC0FFEE ^ ((n_tokens as u64) * 1013 + (n_heads as u64) * 37);
        let max_rel_err = run_shape(
            &dev,
            &kernel,
            RopeNeoxShape { n_tokens, n_heads, head_dim, rotated_dims },
            theta_base,
            seed,
        )?;
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
    }

    let pass = results.iter().all(|r| r.pass);
    let rig = rig();
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "rope_neox_partial_f16_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "rope_neox_partial".to_string(),
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

/// Shared shape for `sweep_rope_neox`'s `run_shape` + `launch` helpers.
#[derive(Copy, Clone, Debug)]
struct RopeNeoxShape {
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    rotated_dims: usize,
}

fn run_shape(
    dev: &HipDevice,
    kernel: &HipKernel<'_>,
    shape: RopeNeoxShape,
    theta_base: f32,
    seed: u64,
) -> Result<f32> {
    let RopeNeoxShape { n_tokens, n_heads, head_dim, rotated_dims } = shape;
    assert_eq!(rotated_dims % 2, 0);
    assert!(rotated_dims <= head_dim);

    let total = n_tokens * n_heads * head_dim;
    let x0_f32 = seeded_f32_range(seed, total, -0.5, 0.5);
    let x0_f16: Vec<f16> = x0_f32.iter().map(|v| f16::from_f32(*v)).collect();
    let positions: Vec<i32> = (0..n_tokens as i32).collect();

    // CPU reference: rotate only dims [0, rotated_dims), pass the rest through.
    let half = rotated_dims / 2;
    let mut reference = vec![0.0f32; total];
    for (t, &pos) in positions.iter().enumerate().take(n_tokens) {
        for h in 0..n_heads {
            let base = (t * n_heads + h) * head_dim;
            // Pass-through block.
            for d in rotated_dims..head_dim {
                reference[base + d] = x0_f16[base + d].to_f32();
            }
            // NeoX-split rotated block.
            for pair_i in 0..half {
                let lo = base + pair_i;
                let hi = base + pair_i + half;
                let x0 = x0_f16[lo].to_f32();
                let x1 = x0_f16[hi].to_f32();
                let inv_freq =
                    1.0f32 / theta_base.powf(2.0 * (pair_i as f32) / (rotated_dims as f32));
                let angle = (pos as f32) * inv_freq;
                let c = angle.cos();
                let s = angle.sin();
                reference[lo] = f16::from_f32(x0 * c - x1 * s).to_f32();
                reference[hi] = f16::from_f32(x0 * s + x1 * c).to_f32();
            }
        }
    }

    // GPU rotate.
    let d_x = alloc_and_upload(dev, &x0_f16);
    let d_pos = alloc_and_upload(dev, &positions);
    launch(
        dev,
        kernel,
        d_x,
        d_pos,
        theta_base,
        RopeNeoxShape { n_tokens, n_heads, head_dim, rotated_dims },
    )?;

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
    let oracle_err = max_rel_err_with_floor(&got1_f32, &reference, 1.0);

    // Round-trip: rotate by -positions; should return to original modulo F16.
    let neg_positions: Vec<i32> = positions.iter().map(|p| -p).collect();
    let d_neg_pos = alloc_and_upload(dev, &neg_positions);
    launch(
        dev,
        kernel,
        d_x,
        d_neg_pos,
        theta_base,
        RopeNeoxShape { n_tokens, n_heads, head_dim, rotated_dims },
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

    Ok(oracle_err.max(roundtrip_err))
}

fn launch(
    dev: &HipDevice,
    kernel: &HipKernel<'_>,
    d_x: DevicePtr,
    d_pos: DevicePtr,
    theta_base: f32,
    shape: RopeNeoxShape,
) -> Result<()> {
    let RopeNeoxShape { n_tokens, n_heads, head_dim, rotated_dims } = shape;
    let stream = dev.default_stream();
    let theta = theta_base;
    let n_heads_i = n_heads as i32;
    let head_dim_i = head_dim as i32;
    let rotated_dims_i = rotated_dims as i32;
    let d_x_ptr: u64 = d_x.as_usize() as u64;
    let d_pos_ptr: u64 = d_pos.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&d_x_ptr);
    args.push(&d_pos_ptr);
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
    stream.synchronize()?;
    Ok(())
}
