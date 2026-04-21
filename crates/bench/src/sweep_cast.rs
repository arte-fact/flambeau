//! V1.7.3-b cast_f32_f16 correctness sweep — single-kernel pointwise cast.
//!
//! Reference: host F32→f16 round-trip (`half::f16::from_f32`). We expect
//! bit-exact match on every element: both the device and the host cast
//! go through round-to-nearest-even with identical rounding modes on the
//! values produced by `seeded_f32`.

#![cfg(feature = "hip")]

use std::path::Path;

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::{
    device_count, FuncAttributes, HipDevice, HipKernel, HipModule, KernelArgs, LaunchCfg,
};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;
use half::f16;

use crate::cert::{now_utc_iso8601, Cert, PmcSnapshot, ShapeResult, SCHEMA_VERSION};

pub fn run_sweep(repo_root: &Path) -> Result<Cert> {
    let n = device_count().context("hipGetDeviceCount")?;
    if n < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;

    let kb = kernels::hsaco("cast_f32_f16")
        .ok_or_else(|| anyhow::anyhow!("cast_f32_f16 not compiled"))?;
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_cast_f32_f16")?;
    let attrs: FuncAttributes = kernel.attributes()?;

    // Shapes covering Qwen3.6 decode-path tensor sizes:
    // - 2048: hidden
    // - 4096: per-head projection × n_heads (post-split gate size)
    // - 8192: fused (Q, gate) output width
    // - 64:   odd small, checks tail-handling
    // - 1024: covers shapes that don't divide 256 cleanly
    let shapes = [64usize, 1024, 2048, 4096, 8192];
    let mut results = Vec::new();
    for n_elems in shapes {
        let seed = 0xC0FFEE ^ ((n_elems as u64) * 31 + 7);
        let err = run_shape(&dev, &kernel, n_elems, seed)?;
        results.push(ShapeResult {
            m: n_elems,
            k: 1,
            n: 1,
            seed,
            max_rel_err: err,
            tolerance: 0.0,
            pass: err == 0.0,
        });
    }

    let pass = results.iter().all(|r| r.pass);
    let rig = format!("{}-gfx906", hostname().unwrap_or_else(|| "unknown".into()));
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "cast_f32_f16_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "cast_f32_f16".to_string(),
        dtype_weight: "F32".to_string(),
        dtype_activation: "F16".to_string(),
        tolerance_formula: "exact bit match vs half::f16::from_f32".to_string(),
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

fn run_shape(dev: &HipDevice, kernel: &HipKernel<'_>, n_elems: usize, seed: u64) -> Result<f32> {
    let x = seeded_f32(seed, n_elems);
    let reference: Vec<f16> = x.iter().map(|v| f16::from_f32(*v)).collect();

    let d_x = alloc_and_upload(dev, &x);
    let d_y = dev.alloc(n_elems * 2)?;
    {
        let stream = dev.default_stream();
        let n_i = n_elems as i32;
        let x_ptr: u64 = d_x.as_usize() as u64;
        let y_ptr: u64 = d_y.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&x_ptr);
        args.push(&y_ptr);
        args.push(&n_i);
        let cfg = LaunchCfg::one_d((n_elems as u32).div_ceil(256), 256);
        unsafe { kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }
    let mut got = vec![f16::from_f32(0.0); n_elems];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got.as_mut_ptr() as usize),
            d_y,
            n_elems * 2,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_x, n_elems * 4)?;
        dev.dealloc(d_y, n_elems * 2)?;
    }

    // Bit-exact comparison — any mismatch lifts the score above 0.
    let mut diff_bits = 0u32;
    for (a, b) in got.iter().zip(&reference) {
        if a.to_bits() != b.to_bits() {
            diff_bits += 1;
        }
    }
    Ok(diff_bits as f32)
}

fn alloc_and_upload<T: Copy>(dev: &HipDevice, data: &[T]) -> DevicePtr {
    let bytes = std::mem::size_of_val(data);
    let d = dev.alloc(bytes).unwrap();
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d,
            DevicePtr(data.as_ptr() as usize),
            bytes,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    d
}

fn seeded_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (s >> 32) as u32;
            (u as f32 / u32::MAX as f32) - 0.5
        })
        .collect()
}

fn hostname() -> Option<String> {
    std::env::var("HOSTNAME").ok().or_else(|| {
        let mut buf = vec![0u8; 256];
        let rv = unsafe { libc_gethostname(buf.as_mut_ptr() as *mut _, buf.len()) };
        if rv != 0 {
            return None;
        }
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        buf.truncate(end);
        String::from_utf8(buf).ok()
    })
}

extern "C" {
    #[link_name = "gethostname"]
    fn libc_gethostname(name: *mut std::os::raw::c_char, len: usize) -> i32;
}
