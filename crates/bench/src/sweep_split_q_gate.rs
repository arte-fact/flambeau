//! split-q-gate correctness sweep.

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
use crate::harness::{alloc_and_upload, rig, seeded_f32_range};

pub fn run_sweep(repo_root: &Path) -> Result<Cert> {
    let n = device_count().context("hipGetDeviceCount")?;
    if n < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;

    let kb = kernels::hsaco("split_q_gate_f16")
        .ok_or_else(|| anyhow::anyhow!("split_q_gate_f16 not compiled"))?;
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_split_q_gate_f16")?;
    let attrs: FuncAttributes = kernel.attributes()?;

    // (n_tokens, n_head, head_dim). Qwen3.6: 16 Q heads, head_dim=256.
    let shapes = [(1usize, 16usize, 256usize), (128, 16, 256), (1, 2, 256)];
    let mut results = Vec::new();
    for (n_tokens, n_head, head_dim) in shapes {
        let seed = 0xC0FFEE
            ^ ((n_tokens as u64) * 1049 + (n_head as u64) * 53 + (head_dim as u64) * 11);
        let max_rel_err = run_shape(&dev, &kernel, n_tokens, n_head, head_dim, seed)?;
        let tol = 0.0;
        results.push(ShapeResult {
            m: n_tokens,
            k: n_head,
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
        impl_id: "split_q_gate_f16_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "split_q_gate".to_string(),
        dtype_weight: "F16".to_string(),
        dtype_activation: "F16".to_string(),
        tolerance_formula: "exact match (strided copy, no arithmetic)".to_string(),
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

fn run_shape(
    dev: &HipDevice,
    kernel: &HipKernel<'_>,
    n_tokens: usize,
    n_head: usize,
    head_dim: usize,
    seed: u64,
) -> Result<f32> {
    let total_fused = n_tokens * n_head * 2 * head_dim;
    let total_split = n_tokens * n_head * head_dim;
    let fused_f32 = seeded_f32_range(seed, total_fused, -0.5, 0.5);
    let fused_f16: Vec<f16> = fused_f32.iter().map(|v| f16::from_f32(*v)).collect();

    // CPU reference split.
    let mut ref_q = vec![f16::from_f32(0.0); total_split];
    let mut ref_g = vec![f16::from_f32(0.0); total_split];
    for t in 0..n_tokens {
        for h in 0..n_head {
            for d in 0..head_dim {
                let fb = (t * n_head + h) * (2 * head_dim);
                let sb = (t * n_head + h) * head_dim;
                ref_q[sb + d] = fused_f16[fb + d];
                ref_g[sb + d] = fused_f16[fb + head_dim + d];
            }
        }
    }

    let d_f = alloc_and_upload(dev, &fused_f16);
    let d_q = dev.alloc(total_split * 2)?;
    let d_g = dev.alloc(total_split * 2)?;
    {
        let stream = dev.default_stream();
        let n_tokens_i = n_tokens as i32;
        let n_head_i = n_head as i32;
        let head_dim_i = head_dim as i32;
        let f_ptr: u64 = d_f.as_usize() as u64;
        let q_ptr: u64 = d_q.as_usize() as u64;
        let g_ptr: u64 = d_g.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&f_ptr);
        args.push(&q_ptr);
        args.push(&g_ptr);
        args.push(&n_tokens_i);
        args.push(&n_head_i);
        args.push(&head_dim_i);
        let threads = 128u32;
        let grid_z = (head_dim as u32).div_ceil(threads);
        let cfg = LaunchCfg {
            grid: (n_tokens as u32, n_head as u32, grid_z),
            block: (threads, 1, 1),
            shared_bytes: 0,
        };
        unsafe { kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }
    let mut got_q = vec![f16::from_f32(0.0); total_split];
    let mut got_g = vec![f16::from_f32(0.0); total_split];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got_q.as_mut_ptr() as usize),
            d_q,
            total_split * 2,
        )?;
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got_g.as_mut_ptr() as usize),
            d_g,
            total_split * 2,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_f, total_fused * 2)?;
        dev.dealloc(d_q, total_split * 2)?;
        dev.dealloc(d_g, total_split * 2)?;
    }
    // Exact match expected — pure strided copy.
    let mut err = 0.0f32;
    for (a, b) in got_q.iter().zip(&ref_q) {
        if a.to_bits() != b.to_bits() { err = err.max(1.0); }
    }
    for (a, b) in got_g.iter().zip(&ref_g) {
        if a.to_bits() != b.to_bits() { err = err.max(1.0); }
    }
    Ok(err)
}

