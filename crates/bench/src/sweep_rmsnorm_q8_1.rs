//! V1.6 RMSNorm + Q8_1 fused correctness sweep.
//!
//! Compares the fused kernel's Q8_1 output against the 2-kernel oracle
//! (`rmsnorm_f16` → `quantize_row_q8_1`). Both paths should produce
//! bit-equivalent Q8_1 blocks modulo the round-off difference between
//! F16-temporary storage and direct F32-in-register propagation. The
//! fused kernel keeps `normed` in F32 all the way through the quant step,
//! so it may be *more* accurate than the oracle; we test the dequantised
//! F32 projection of each Q8_1 block against the "true" F32 reference.

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
use flambeau_quant::{BlockQ8_1, QK8_0};
use half::f16;

use crate::cert::{now_utc_iso8601, Cert, PmcSnapshot, ShapeResult, SCHEMA_VERSION};
use crate::harness::{alloc_and_upload, max_rel_err_with_floor, rig, seeded_f32_range};

const QK8: usize = QK8_0;

pub fn run_sweep(repo_root: &Path) -> Result<Cert> {
    let n = device_count().context("hipGetDeviceCount")?;
    if n < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;

    let kb = kernels::hsaco("rmsnorm_q8_1_fused")
        .ok_or_else(|| anyhow::anyhow!("rmsnorm_q8_1_fused not compiled"))?;
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_rmsnorm_q8_1_fused")?;
    let attrs: FuncAttributes = kernel.attributes()?;

    // Qwen3.6 hidden sizes that are multiples of 32 (QK8_1) — all of them.
    let shapes = [
        (1usize, 2048usize),
        (1, 5120),
        (1, 15360),
        (8, 5120),
        (128, 5120),
    ];
    let eps = 1e-6f32;
    let seed = 0xDECADEu64;

    let mut results = Vec::new();
    for (m, k) in shapes {
        let (got, reference) = run_shape(&dev, &kernel, m, k, eps, seed)?;
        let max_rel = max_rel_err_with_floor(&got, &reference, (k as f32).sqrt() * 0.01);
        let tol = 2e-2; // Q8_1 quant noise sets the bar
        results.push(ShapeResult {
            m,
            k,
            n: k,
            seed,
            max_rel_err: max_rel,
            tolerance: tol,
            pass: max_rel <= tol,
        });
        tracing::info!(
            target: "flambeau_bench::sweep_rmsnorm_q8_1",
            m, k, max_rel, tol,
            "rmsnorm_q8_1 shape"
        );
    }

    let pass = results.iter().all(|r| r.pass);
    let rig = rig();
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "rmsnorm_q8_1_fused_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "rmsnorm_q8_1_fused".to_string(),
        dtype_weight: "F16".to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula: "|err| <= 2e-2 * max(|ref|, sqrt(k) * 0.01)".to_string(),
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
    m: usize,
    k: usize,
    eps: f32,
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>)> {
    assert_eq!(k % QK8, 0);
    let nb_per_row = k / QK8;

    // Random inputs in [-0.5, 0.5] — keeps RMS in a nominal range and
    // makes the per-block quant scale close to the empirical max.
    let x_f32 = seeded_f32_range(seed, m * k, -0.5, 0.5);
    let w_f32 = seeded_f32_range(seed.wrapping_add(0x5A5A5A5A), k, -0.5, 0.5);
    let x_f16: Vec<f16> = x_f32.iter().map(|v| f16::from_f32(*v)).collect();
    let w_f16: Vec<f16> = w_f32.iter().map(|v| f16::from_f32(*v)).collect();

    let d_x = alloc_and_upload(dev, &x_f16);
    let d_w = alloc_and_upload(dev, &w_f16);
    let d_y = dev.alloc(m * nb_per_row * std::mem::size_of::<BlockQ8_1>())?;

    {
        let stream = dev.default_stream();
        let n_rows_i = m as i32;
        let k_i = k as i32;
        let eps_f = eps;
        let d_x_ptr: u64 = d_x.as_usize() as u64;
        let d_w_ptr: u64 = d_w.as_usize() as u64;
        let d_y_ptr: u64 = d_y.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_x_ptr);
        args.push(&d_w_ptr);
        args.push(&d_y_ptr);
        args.push(&n_rows_i);
        args.push(&k_i);
        args.push(&eps_f);
        let cfg = LaunchCfg::one_d(m as u32, 256);
        unsafe { kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    // Download Q8_1 blocks and dequantise to F32 for comparison.
    let blocks_total = m * nb_per_row;
    let mut blocks: Vec<BlockQ8_1> = vec![
        BlockQ8_1 {
            d: f16::from_f32(0.0),
            s: f16::from_f32(0.0),
            qs: [0i8; 32],
        };
        blocks_total
    ];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(blocks.as_mut_ptr() as usize),
            d_y,
            blocks_total * std::mem::size_of::<BlockQ8_1>(),
        )?;
    }
    dev.default_stream().synchronize()?;

    unsafe {
        dev.dealloc(d_x, x_f16.len() * 2)?;
        dev.dealloc(d_w, w_f16.len() * 2)?;
        dev.dealloc(d_y, blocks_total * std::mem::size_of::<BlockQ8_1>())?;
    }

    let mut got = vec![0.0f32; m * k];
    for (bi, block) in blocks.iter().enumerate() {
        let d = block.d.to_f32();
        for j in 0..QK8 {
            got[bi * QK8 + j] = (block.qs[j] as f32) * d;
        }
    }

    // Reference: RMSNorm in F32 (with F16 inputs, same as the kernel's first
    // read), then Q8_1 quantise per block, then dequantise for comparison.
    // This is exactly what the fused kernel computes, so the two paths
    // should agree to within F16-input round-off.
    let mut reference = vec![0.0f32; m * k];
    let x_inputs: Vec<f32> = x_f16.iter().map(|v| v.to_f32()).collect();
    let w_inputs: Vec<f32> = w_f16.iter().map(|v| v.to_f32()).collect();
    for row in 0..m {
        let xr = &x_inputs[row * k..(row + 1) * k];
        let mut ss = 0.0f64;
        for v in xr {
            ss += (*v as f64) * (*v as f64);
        }
        let rsqrt = 1.0f32 / ((ss as f32 / k as f32) + eps).sqrt();
        // Per-block quantise — same arithmetic as the kernel.
        for b in 0..(k / QK8) {
            let start = b * QK8;
            let mut normed = [0.0f32; 32];
            let mut amax = 0.0f32;
            for j in 0..QK8 {
                normed[j] = xr[start + j] * rsqrt * w_inputs[start + j];
                let a = normed[j].abs();
                if a > amax {
                    amax = a;
                }
            }
            let d = amax / 127.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            for j in 0..QK8 {
                let qi = (normed[j] * id).round().clamp(-127.0, 127.0) as i32;
                reference[row * k + start + j] = (qi as f32) * d;
            }
        }
    }

    Ok((got, reference))
}

