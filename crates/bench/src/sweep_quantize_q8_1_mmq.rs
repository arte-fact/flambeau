//! V2.2.d.P4 — round-trip cert for the `quantize_q8_1_mmq` kernel.
//!
//! Generates seeded F32 input `[total_b, ncols]`, runs the MMQ Q8_1
//! prequantiser on device, reads the 144-B blocks back, dequantises
//! each block on CPU using the same (d, Σxi) pair the GPU wrote, and
//! compares to the input.
//!
//! Passes when `max_rel_err ≤ 1e-2`. Q8_1's 8-bit granularity is
//! ~1/127 ≈ 0.8 %, so the cert tolerance is 1.25× that floor to cover
//! the sub-block sum path that the Q4_1 vec_dot consumes.

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
    device_count, HipDevice, HipKernel, HipModule, KernelArgs, LaunchCfg,
};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;
use flambeau_quant::BlockQ8_1Mmq;

use crate::cert::{now_utc_iso8601, Cert, ShapeResult, SCHEMA_VERSION};
use crate::harness::{rig, seeded_f32_range};

pub fn run_sweep(repo_root: &Path) -> Result<Cert> {
    let n = device_count().context("hipGetDeviceCount")?;
    if n < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;

    let bytes = kernels::hsaco("quantize_q8_1_mmq")
        .ok_or_else(|| anyhow::anyhow!("quantize_q8_1_mmq not compiled"))?;
    let module = HipModule::load(dev.id(), bytes)?;
    let kernel: HipKernel<'_> =
        module.kernel("flambeau_quantize_q8_1_mmq")?;

    // (total_b = batch rows, ncols = K). ncols must be a multiple of 128.
    let shapes = [
        (1usize, 2048usize),
        (1, 5120),
        (8, 2048),
        (128, 2048),
        (128, 5120),
        (512, 2048),
    ];
    let seed = 0xB10Cu64;

    let mut results = Vec::new();
    for (total_b, ncols) in shapes {
        let (max_rel_err, pass) = run_shape(&dev, &kernel, total_b, ncols, seed)?;
        let tol = 1e-2;
        results.push(ShapeResult {
            m: total_b,
            k: ncols,
            n: 1,
            seed,
            max_rel_err,
            tolerance: tol,
            pass: pass && max_rel_err <= tol,
        });
        tracing::info!(
            target: "flambeau_bench::sweep_quantize_q8_1_mmq",
            total_b, ncols, max_rel_err, tol,
            "quantize_q8_1_mmq shape"
        );
    }

    let pass = results.iter().all(|r| r.pass);
    let rig = rig();

    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "quantize_q8_1_mmq_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "quantize".to_string(),
        dtype_weight: "F32".to_string(),
        dtype_activation: "Q8_1_MMQ".to_string(),
        tolerance_formula: "|max_rel_err| <= 1e-2 (~1.25× Q8_1 noise floor)".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig,
        pmc: None,
    };

    let written = cert.write_to_disk(repo_root)?;
    tracing::info!(
        target: "flambeau_bench::sweep_quantize_q8_1_mmq",
        cert = %written.display(),
        pass = cert.pass,
        "cert written"
    );
    Ok(cert)
}

fn run_shape(
    dev: &HipDevice,
    kernel: &HipKernel<'_>,
    total_b: usize,
    ncols: usize,
    seed: u64,
) -> Result<(f32, bool)> {
    assert_eq!(ncols % 128, 0, "ncols must be a multiple of QK8_1_MMQ=128");
    let n_big_blocks = ncols / 128;

    // Seeded F32 input [total_b, ncols] row-major.
    let input = seeded_f32_range(seed, total_b * ncols, -1.0, 1.0);
    let d_x = dev.alloc(total_b * ncols * 4)?;
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d_x,
            DevicePtr(input.as_ptr() as usize),
            total_b * ncols * 4,
        )?;
    }
    dev.default_stream().synchronize()?;

    // Output buffer: n_big_blocks * total_b × 144 B.
    let block_bytes = std::mem::size_of::<BlockQ8_1Mmq>();
    assert_eq!(block_bytes, 144);
    let d_y = dev.alloc(n_big_blocks * total_b * block_bytes)?;

    // Launch: grid = (n_big_blocks, total_b), block = 128.
    let ncols_i = ncols as i32;
    let total_b_i = total_b as i32;
    let d_x_ptr: u64 = d_x.as_usize() as u64;
    let d_y_ptr: u64 = d_y.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&d_x_ptr);
    args.push(&d_y_ptr);
    args.push(&ncols_i);
    args.push(&total_b_i);
    let cfg = LaunchCfg {
        grid: (n_big_blocks as u32, total_b as u32, 1),
        block: (128, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(dev.default_stream(), cfg, args)? };
    dev.default_stream().synchronize()?;

    // Download blocks.
    let mut blocks_raw = vec![0u8; n_big_blocks * total_b * block_bytes];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(blocks_raw.as_mut_ptr() as usize),
            d_y,
            blocks_raw.len(),
        )?;
    }
    dev.default_stream().synchronize()?;

    unsafe {
        dev.dealloc(d_x, total_b * ncols * 4)?;
        dev.dealloc(d_y, n_big_blocks * total_b * block_bytes)?;
    }

    // CPU dequantise + compare. Blocks are laid out (big_block, col) row-major:
    //   block[b, c] at offset (b * total_b + c) * 144.
    let blocks: &[BlockQ8_1Mmq] = bytemuck::cast_slice(&blocks_raw);

    // Per-block error metric: the Q8_1 quant noise floor is d/2 per element,
    // where d = amax/127. Reporting |got - orig| / block_amax gives the
    // quantisation noise as a fraction of the block's dynamic range — a
    // tight bound (≤ 0.5/127 ≈ 0.004 absent bugs).
    let mut max_rel_err = 0.0f32;
    let mut any_nan = false;
    let mut ssum_max_err = 0.0f32;

    for c in 0..total_b {
        for b in 0..n_big_blocks {
            let bk = &blocks[b * total_b + c];
            for sub in 0..4 {
                let d = bk.ds[sub * 2].to_f32();
                let ssum = bk.ds[sub * 2 + 1].to_f32();

                // Compute the block's amax for the noise-floor denominator.
                let mut block_amax = 0.0f32;
                let mut block_ssum = 0.0f32;
                for lane in 0..32 {
                    let orig_k = b * 128 + sub * 32 + lane;
                    let orig = input[c * ncols + orig_k];
                    block_amax = block_amax.max(orig.abs());
                    block_ssum += orig;
                }
                let denom = block_amax.max(1.0 / 127.0);

                for lane in 0..32 {
                    let q = bk.qs[sub * 32 + lane];
                    let got = q as f32 * d;
                    let orig_k = b * 128 + sub * 32 + lane;
                    let orig = input[c * ncols + orig_k];
                    let err = (got - orig).abs() / denom;
                    if err.is_nan() {
                        any_nan = true;
                    }
                    if err > max_rel_err {
                        max_rel_err = err;
                    }
                }

                // Also verify the stored ssum matches the true Σ xi within a
                // reasonable tolerance (half-precision round-trip).
                let ssum_err = (ssum - block_ssum).abs() / block_amax.max(1.0);
                if ssum_err > ssum_max_err {
                    ssum_max_err = ssum_err;
                }
            }
        }
    }

    // Flag ssum mismatch as a separate issue — if the stored Σ xi differs
    // materially, Q4_1's bias term will be wrong.
    if ssum_max_err > 1e-2 {
        tracing::warn!(
            target: "flambeau_bench::sweep_quantize_q8_1_mmq",
            total_b, ncols, ssum_max_err,
            "ssum round-trip error exceeds 1e-2"
        );
    }

    Ok((max_rel_err.max(ssum_max_err), !any_nan))
}

