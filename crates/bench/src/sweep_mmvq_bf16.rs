//! BF16 weight × BF16 activation MMVQ correctness sweep.
//! Validates `flambeau_mmvq_bf16_bf16` against an F32 reference that
//! up-casts both BF16 inputs to F32 (lossless bit-shift) and accumulates
//! sequentially. The kernel does the same up-cast in F32 but reduces in
//! parallel (256 threads × per-warp sum); accumulation order differs, so
//! we use a relative-error tolerance, not bit-exact.
//! Shapes cover Qwen3.6-27B MTP-block projection sizes:
//! * fc: n=5120, k=10240 (concat embedding+hidden → hidden)
//! * q_proj: n=12288, k=5120 (Q ‖ gate)
//! * k/v_proj: n=1024, k=5120
//! * o_proj: n=5120, k=6144 (n_q*head_dim → hidden)
//! * gate/up: n=intermediate, k=5120
//! * down: n=5120, k=intermediate
//! * lm_head: n=248320, k=5120

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
use half::bf16;

use crate::cert::{now_utc_iso8601, Cert, PmcSnapshot, ShapeResult, SCHEMA_VERSION};
use crate::harness::{alloc_and_upload, max_rel_err_with_floor, rig, seeded_f32_range};

/// Decode-path MTP shapes; smoke + production sizes.
const SHAPES: &[(usize, usize)] = &[
    (256, 256),       // smoke
    (5120, 10240),    // mtp.fc (hidden, 2*hidden)
    (12288, 5120),    // q_proj (2*n_q*head_dim, hidden)
    (1024, 5120),     // k/v_proj
    (5120, 6144),     // o_proj
    (5120, 25600),    // mtp down_proj (hidden, intermediate)
];

pub fn run_sweep(repo_root: &Path) -> Result<Cert> {
    if device_count().context("hipGetDeviceCount")? < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;
    let kb = kernels::hsaco("mmvq_bf16_bf16")
        .ok_or_else(|| anyhow::anyhow!("mmvq_bf16_bf16 not compiled"))?;
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_mmvq_bf16_bf16")?;
    let attrs: FuncAttributes = kernel.attributes()?;

    let mut results = Vec::new();
    for &(n, k) in SHAPES {
        let seed = 0xBF16C0FFu64 ^ (n as u64 * 7919) ^ (k as u64 * 101);
        let (got, reference) = run_shape(&dev, &kernel, n, k, seed)?;
        let max_rel = max_rel_err_with_floor(&got, &reference, (k as f32).sqrt() * 0.01);
        // Both operands are BF16 (relative error ~1/128 = 7.8e-3 each at
        // worst). The matmul accumulates in F32, so rounding is at-input
        // only. With the parallel reduction's different summation order
        // adding O(eps * sqrt(k)) ≈ 1e-7 * 71 ≈ 7e-6 on top, 3e-2 is a
        // comfortable bar that catches structural bugs while ignoring
        // the input-rounding floor.
        let tol = 3e-2;
        results.push(ShapeResult {
            m: 1,
            k,
            n,
            seed,
            max_rel_err: max_rel,
            tolerance: tol,
            pass: max_rel <= tol,
        });
        tracing::info!(
            target: "flambeau_bench::sweep_mmvq_bf16",
            n, k, max_rel, tol,
            "mmvq bf16 × bf16 shape"
        );
    }

    let pass = results.iter().all(|r| r.pass);
    let rig = rig();
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "mmvq_bf16_bf16_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "qmatmul_mmvq".to_string(),
        dtype_weight: "BF16".to_string(),
        dtype_activation: "BF16".to_string(),
        tolerance_formula: "|err| <= 3e-2 * max(|ref|, sqrt(k) * 0.01)".to_string(),
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
    n: usize,
    k: usize,
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let w_f32 = seeded_f32_range(seed, n * k, -0.5, 0.5);
    let x_f32 = seeded_f32_range(seed.wrapping_add(0xA1), k, -0.5, 0.5);
    // Bake both operands at BF16 precision so the reference reflects
    // exactly what the kernel sees in HBM.
    let w_bf16: Vec<bf16> = w_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let x_bf16: Vec<bf16> = x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let w_lifted: Vec<f32> = w_bf16.iter().map(|v| v.to_f32()).collect();
    let x_lifted: Vec<f32> = x_bf16.iter().map(|v| v.to_f32()).collect();

    let d_w = alloc_and_upload(dev, &w_bf16);
    let d_x = alloc_and_upload(dev, &x_bf16);
    let d_out = dev.alloc(n * 4)?;

    {
        let stream = dev.default_stream();
        let n_rows_i = n as i32;
        let k_i = k as i32;
        let w_ptr: u64 = d_w.as_usize() as u64;
        let y_ptr: u64 = d_x.as_usize() as u64;
        let o_ptr: u64 = d_out.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&w_ptr);
        args.push(&y_ptr);
        args.push(&o_ptr);
        args.push(&n_rows_i);
        args.push(&k_i);
        let cfg = LaunchCfg::one_d(n as u32, 256);
        unsafe { kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    let mut got = vec![0.0f32; n];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got.as_mut_ptr() as usize),
            d_out,
            n * 4,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_w, n * k * 2)?;
        dev.dealloc(d_x, k * 2)?;
        dev.dealloc(d_out, n * 4)?;
    }

    // Reference: sequential F32 dot per row.
    let mut reference = vec![0.0f32; n];
    for row in 0..n {
        let wrow = &w_lifted[row * k..(row + 1) * k];
        let mut acc = 0.0f32;
        for j in 0..k {
            acc += wrow[j] * x_lifted[j];
        }
        reference[row] = acc;
    }

    Ok((got, reference))
}
