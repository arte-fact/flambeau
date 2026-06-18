//! Cross-backend (BenchBackend-generic) correctness sweeps for the Stream F
//! fixed pointwise kernels. The legacy `sweep_f32_pointwise` is HIP-only
//! (`open_kernel` + hardcoded `_gfx906` certs); this module is the P3 spine —
//! the same sweeps over `BenchBackend` so each kernel certs on HIP gfx906 AND
//! CUDA sm_86 from one driver. Pilot: `add_f16`.
#![cfg(any(feature = "hip", feature = "cuda"))]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "sweep harness — every unsafe block is a memcpy_async over buffers \
              allocated locally in the same function and freed before return."
)]

use std::path::Path;

use anyhow::{anyhow, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr, LaunchCfg, Stream};
use half::f16;

use crate::backend::{BenchBackend, LaunchArgs};
use crate::cert::{now_utc_iso8601, Cert, ShapeResult, SCHEMA_VERSION};
use crate::harness::{alloc_and_upload, rig_for, seeded_f32_range};

/// `add_f16` — pointwise `y[i] = (f32)a[i] + (f32)b[i] → f16`. Bit-exact vs the
/// F32-add-then-F16-cast host reference.
pub fn run_add_f16_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let image = B::image("add_f16").ok_or_else(|| anyhow!("add_f16 not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, "flambeau_add_f16")?;
    let pmc = B::attributes(&kernel)?;

    let shapes = [64usize, 256, 2048, 4096, 8192];
    let mut results = Vec::new();
    for n in shapes {
        let seed = 0xC0FFEE ^ ((n as u64) * 83 + 23);
        let a_f32 = seeded_f32_range(seed, n, -0.5, 0.5);
        let b_f32 = seeded_f32_range(seed ^ 0xBE, n, -0.5, 0.5);
        let a_f16: Vec<f16> = a_f32.iter().map(|v| f16::from_f32(*v)).collect();
        let b_f16: Vec<f16> = b_f32.iter().map(|v| f16::from_f32(*v)).collect();
        let reference: Vec<f16> = a_f16
            .iter()
            .zip(&b_f16)
            .map(|(a, b)| f16::from_f32(a.to_f32() + b.to_f32()))
            .collect();

        let d_a = alloc_and_upload(&dev, &a_f16);
        let d_b = alloc_and_upload(&dev, &b_f16);
        let d_y = dev.alloc(n * 2)?;
        {
            let stream = dev.default_stream();
            let n_i = n as i32;
            let a_ptr: u64 = d_a.as_usize() as u64;
            let b_ptr: u64 = d_b.as_usize() as u64;
            let y_ptr: u64 = d_y.as_usize() as u64;
            let mut args = <B::Args<'_>>::default();
            args.push_arg(&a_ptr);
            args.push_arg(&b_ptr);
            args.push_arg(&y_ptr);
            args.push_arg(&n_i);
            B::launch(&kernel, stream, LaunchCfg::one_d((n as u32).div_ceil(256), 256), args)?;
            stream.synchronize()?;
        }
        let mut got = vec![f16::from_f32(0.0); n];
        unsafe {
            dev.memcpy_async(
                dev.default_stream(),
                CopyDirection::DeviceToHost,
                DevicePtr(got.as_mut_ptr() as usize),
                d_y,
                n * 2,
            )?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_a, n * 2)?;
            dev.dealloc(d_b, n * 2)?;
            dev.dealloc(d_y, n * 2)?;
        }

        let mismatches = got
            .iter()
            .zip(&reference)
            .filter(|(g, r)| g.to_bits() != r.to_bits())
            .count();
        let err = mismatches as f32;
        results.push(ShapeResult {
            m: n,
            k: 1,
            n: 1,
            seed,
            max_rel_err: err,
            tolerance: 0.0,
            pass: err == 0.0,
        });
    }

    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "add_f16".to_string(),
        backend: dev.backend().to_string(),
        arch: arch.to_string(),
        op: "add_f16".to_string(),
        dtype_weight: "F16".to_string(),
        dtype_activation: "F16".to_string(),
        tolerance_formula: "exact bit match vs (f32) a + (f32) b → f16".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_for(arch),
        pmc: Some(pmc),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}
