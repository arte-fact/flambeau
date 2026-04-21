//! V1.7.2.C causal depthwise Conv1d correctness sweep.
//!
//! Cert compares the GPU output against a CPU F32 reference at shapes matching
//! Qwen3.6 GDN inner conv (`conv_kernel = 4`, `conv_channels = 8192`). Also
//! exercises a small width for edge-case coverage.

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

    let kb = kernels::hsaco("causal_conv1d_f32")
        .ok_or_else(|| anyhow::anyhow!("causal_conv1d_f32 not compiled"))?;
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_causal_conv1d_f32")?;
    let attrs: FuncAttributes = kernel.attributes()?;

    // (n_new, conv_channels, conv_kernel). Qwen3.6 GDN: conv_channels = 8192,
    // kernel = 4. Decode step: n_new = 1 + 3 history = 4 rows in, 1 out.
    // Prefill: n_new arbitrary, conv_input is n_new + 3.
    let shapes = [
        (1usize, 8192usize, 4usize),
        (128, 8192, 4),
        (1, 512, 4),
        (32, 2048, 4),
    ];

    let mut results = Vec::new();
    for (n_new, conv_channels, conv_kernel) in shapes {
        let seed = 0xC0FFEE
            ^ ((n_new as u64) * 1033 + (conv_channels as u64) * 43 + (conv_kernel as u64) * 7);
        let max_rel_err =
            run_shape(&dev, &kernel, n_new, conv_channels, conv_kernel, seed)?;
        let tol = 5e-5;
        results.push(ShapeResult {
            m: n_new,
            k: conv_channels,
            n: conv_kernel,
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
        impl_id: "causal_conv1d_f32_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "causal_conv1d".to_string(),
        dtype_weight: "F32".to_string(),
        dtype_activation: "F32".to_string(),
        tolerance_formula: "|err| <= 5e-5 * max(|ref|, 1)".to_string(),
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
    n_new: usize,
    conv_channels: usize,
    conv_kernel: usize,
    seed: u64,
) -> Result<f32> {
    let n_total = n_new + conv_kernel - 1;
    let x = seeded_f32(seed, n_total * conv_channels);
    let w = seeded_f32(seed.wrapping_add(0xA1), conv_kernel * conv_channels);

    // CPU reference. Weight layout matches GGUF on-disk order for
    // `ssm_conv1d.weight`: `[conv_channels, conv_kernel]` with the
    // kernel-tap axis innermost (offset = c * conv_kernel + k).
    let mut reference = vec![0.0f32; n_new * conv_channels];
    for t in 0..n_new {
        for c in 0..conv_channels {
            let mut acc = 0.0f64;
            for k in 0..conv_kernel {
                let xv = x[(t + k) * conv_channels + c] as f64;
                let wv = w[c * conv_kernel + k] as f64;
                acc += xv * wv;
            }
            reference[t * conv_channels + c] = acc as f32;
        }
    }

    let d_x = alloc_and_upload(dev, &x);
    let d_w = alloc_and_upload(dev, &w);
    let d_y = dev.alloc(n_new * conv_channels * 4)?;
    {
        let stream = dev.default_stream();
        let n_new_i = n_new as i32;
        let cc_i = conv_channels as i32;
        let ck_i = conv_kernel as i32;
        let x_ptr: u64 = d_x.as_usize() as u64;
        let w_ptr: u64 = d_w.as_usize() as u64;
        let y_ptr: u64 = d_y.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&x_ptr);
        args.push(&w_ptr);
        args.push(&y_ptr);
        args.push(&n_new_i);
        args.push(&cc_i);
        args.push(&ck_i);
        let threads = 256u32;
        let grid_x = (conv_channels as u32).div_ceil(threads);
        let cfg = LaunchCfg {
            grid: (grid_x, n_new as u32, 1),
            block: (threads, 1, 1),
            shared_bytes: 0,
        };
        unsafe { kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    let mut got = vec![0.0f32; n_new * conv_channels];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got.as_mut_ptr() as usize),
            d_y,
            n_new * conv_channels * 4,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_x, x.len() * 4)?;
        dev.dealloc(d_w, w.len() * 4)?;
        dev.dealloc(d_y, n_new * conv_channels * 4)?;
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
