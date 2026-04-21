//! V1.7.2.B L2-norm correctness sweep.
//!
//! One block per row, 256 threads. Compared to CPU F32 reference at several
//! shapes matching the GDN Q/K layout (`num_k_heads × head_k_dim = 16 × 128`
//! for Qwen3.6).

#![cfg(feature = "hip")]

use std::path::Path;

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::{
    device_count, FuncAttributes, HipDevice, HipKernel, HipModule, KernelArgs, LaunchCfg,
};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;

use crate::cert::{now_utc_iso8601, Cert, PmcSnapshot, ShapeResult, SCHEMA_VERSION};

pub fn run_sweep(repo_root: &Path) -> Result<Cert> {
    let n = device_count().context("hipGetDeviceCount")?;
    if n < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;

    let kb = kernels::hsaco("l2_norm_f32")
        .ok_or_else(|| anyhow::anyhow!("l2_norm_f32 not compiled"))?;
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_l2_norm_f32")?;
    let attrs: FuncAttributes = kernel.attributes()?;

    // (n_rows, k). Qwen3.6 GDN: rows = n_tokens × num_k_heads (= 16), k = head_k_dim (= 128).
    // Decode: n_tokens=1 → 16 rows. Prefill 128 tokens → 2048 rows.
    // Also include a larger k to exercise the strided loop.
    let shapes = [
        (16usize, 128usize),
        (2048, 128),
        (16, 256),
        (1, 512),
    ];
    let eps = 1e-6f32;

    let mut results = Vec::new();
    for (n_rows, k) in shapes {
        let seed = 0xC0FFEE ^ ((n_rows as u64) * 1031 + (k as u64) * 41);
        let max_rel_err = run_shape(&dev, &kernel, n_rows, k, eps, seed)?;
        let tol = 1e-5;
        results.push(ShapeResult {
            m: n_rows,
            k,
            n: 1,
            seed,
            max_rel_err,
            tolerance: tol,
            pass: max_rel_err <= tol,
        });
    }

    let pass = results.iter().all(|r| r.pass);
    let rig = format!("{}-gfx906", hostname().unwrap_or_else(|| "unknown".into()));
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "l2_norm_f32_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "l2_norm".to_string(),
        dtype_weight: "F32".to_string(),
        dtype_activation: "F32".to_string(),
        tolerance_formula: "|err| <= 1e-5 * max(|ref|, 1)".to_string(),
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
    n_rows: usize,
    k: usize,
    eps: f32,
    seed: u64,
) -> Result<f32> {
    let total = n_rows * k;
    let x = seeded_f32(seed, total);

    // CPU reference: y[i] = x[i] / sqrt(sum(x[row]^2) + eps).
    let mut reference = vec![0.0f32; total];
    for r in 0..n_rows {
        let row = &x[r * k..(r + 1) * k];
        let sum_sq: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let inv = 1.0 / ((sum_sq + eps as f64).sqrt());
        for (i, &v) in row.iter().enumerate() {
            reference[r * k + i] = (v as f64 * inv) as f32;
        }
    }

    let d_x = alloc_and_upload(dev, &x);
    let d_y = dev.alloc(total * 4)?;
    {
        let stream = dev.default_stream();
        let n_rows_i = n_rows as i32;
        let k_i = k as i32;
        let eps_f = eps;
        let x_ptr: u64 = d_x.as_usize() as u64;
        let y_ptr: u64 = d_y.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&x_ptr);
        args.push(&y_ptr);
        args.push(&n_rows_i);
        args.push(&k_i);
        args.push(&eps_f);
        let cfg = LaunchCfg::one_d(n_rows as u32, 256);
        unsafe { kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    let mut got = vec![0.0f32; total];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got.as_mut_ptr() as usize),
            d_y,
            total * 4,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_x, x.len() * 4)?;
        dev.dealloc(d_y, total * 4)?;
    }

    Ok(max_rel_err(&got, &reference))
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
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (s >> 32) as u32;
            (u as f32 / u32::MAX as f32) - 0.5
        })
        .collect()
}

fn max_rel_err(got: &[f32], reference: &[f32]) -> f32 {
    got.iter()
        .zip(reference)
        .map(|(g, r)| (g - r).abs() / r.abs().max(1.0))
        .fold(0.0f32, f32::max)
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
