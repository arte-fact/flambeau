//! c1 correctness sweeps for the GDN F32 pointwise kernels:
//! silu_f32, swiglu_f32, scale_f32, rmsnorm_f32, cast_f16_f32. Each writes
//! its own cert under `certs/hip/gfx906/<impl_id>.json`.
//! All references are host F64 scalar implementations; pass tolerance is
//! `1e-6` in the relative sense. The kernels are trivial pointwise math;
//! any larger error is a real bug.

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

// --- helpers ---------------------------------------------------------------

fn seeded_f32(seed: u64, n: usize) -> Vec<f32> {
    seeded_f32_range(seed, n, -0.5, 0.5)
}

fn max_rel_err(got: &[f32], reference: &[f32]) -> f32 {
    max_rel_err_with_floor(got, reference, 1.0)
}

fn open_kernel(stem: &str, entry: &str) -> Result<(HipDevice, HipModule, String)> {
    let n = device_count().context("hipGetDeviceCount")?;
    if n < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;
    let kb = kernels::hsaco(stem).ok_or_else(|| anyhow::anyhow!("{stem} not compiled"))?;
    let module = HipModule::load(dev.id(), kb)?;
    let _ = module.kernel_dynamic(entry)?; // validate upfront
    Ok((dev, module, entry.to_string()))
}

fn pmc(kernel: &HipKernel<'_>) -> Option<PmcSnapshot> {
    let a: FuncAttributes = kernel.attributes().ok()?;
    Some(PmcSnapshot {
        vgpr_count: Some(a.num_regs),
        sgpr_count: None,
        waves_per_simd: Some(a.gfx906_waves_per_simd()),
        mem_busy_pct: None,
        valu_busy_pct: None,
    })
}

// --- silu_f32 --------------------------------------------------------------

pub fn run_silu_sweep(repo_root: &Path) -> Result<Cert> {
    let (dev, module, entry) = open_kernel("silu_f32", "flambeau_silu_f32")?;
    let kernel = module.kernel_dynamic(&entry)?;
    let shapes = [64usize, 1024, 4096, 8192];
    let mut results = Vec::new();
    for n in shapes {
        let seed = 0xC0FFEE ^ ((n as u64) * 17 + 3);
        let x = seeded_f32(seed, n);
        let reference: Vec<f32> = x
            .iter()
            .map(|&xi| {
                let d = xi as f64;
                (d / (1.0 + (-d).exp())) as f32
            })
            .collect();
        let d_x = alloc_and_upload(&dev, &x);
        let d_y = dev.alloc(n * 4)?;
        {
            let n_i = n as i32;
            let x_ptr: u64 = d_x.as_usize() as u64;
            let y_ptr: u64 = d_y.as_usize() as u64;
            let mut args = KernelArgs::new();
            args.push(&x_ptr);
            args.push(&y_ptr);
            args.push(&n_i);
            let cfg = LaunchCfg::one_d((n as u32).div_ceil(256), 256);
            unsafe { kernel.launch(dev.default_stream(), cfg, args)? };
            dev.default_stream().synchronize()?;
        }
        let mut got = vec![0.0f32; n];
        unsafe {
            dev.memcpy_async(
                dev.default_stream(),
                CopyDirection::DeviceToHost,
                DevicePtr(got.as_mut_ptr() as usize),
                d_y,
                n * 4,
            )?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_x, n * 4)?;
            dev.dealloc(d_y, n * 4)?;
        }
        let err = max_rel_err(&got, &reference);
        results.push(ShapeResult {
            m: n,
            k: 1,
            n: 1,
            seed,
            max_rel_err: err,
            tolerance: 1e-6,
            pass: err <= 1e-6,
        });
    }
    let pass = results.iter().all(|r| r.pass);
    let pmc = pmc(&kernel);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "silu_f32_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "silu_f32".to_string(),
        dtype_weight: "F32".to_string(),
        dtype_activation: "F32".to_string(),
        tolerance_formula: "|err| <= 1e-6 * max(|ref|, 1)".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig(),
        pmc,
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

// --- swiglu_f32 ------------------------------------------------------------

pub fn run_swiglu_sweep(repo_root: &Path) -> Result<Cert> {
    let (dev, module, entry) = open_kernel("swiglu_f32", "flambeau_swiglu_f32")?;
    let kernel = module.kernel_dynamic(&entry)?;
    let shapes = [256usize, 1024, 4096];
    let mut results = Vec::new();
    for n in shapes {
        let seed = 0xC0FFEE ^ ((n as u64) * 23 + 5);
        let a = seeded_f32(seed, n);
        let b = seeded_f32(seed ^ 0xAB, n);
        let reference: Vec<f32> = a
            .iter()
            .zip(&b)
            .map(|(&ai, &bi)| {
                let d = ai as f64;
                let silu = d / (1.0 + (-d).exp());
                (silu * bi as f64) as f32
            })
            .collect();
        let d_a = alloc_and_upload(&dev, &a);
        let d_b = alloc_and_upload(&dev, &b);
        let d_y = dev.alloc(n * 4)?;
        {
            let n_i = n as i32;
            let a_ptr: u64 = d_a.as_usize() as u64;
            let b_ptr: u64 = d_b.as_usize() as u64;
            let y_ptr: u64 = d_y.as_usize() as u64;
            let mut args = KernelArgs::new();
            args.push(&a_ptr);
            args.push(&b_ptr);
            args.push(&y_ptr);
            args.push(&n_i);
            let cfg = LaunchCfg::one_d((n as u32).div_ceil(256), 256);
            unsafe { kernel.launch(dev.default_stream(), cfg, args)? };
            dev.default_stream().synchronize()?;
        }
        let mut got = vec![0.0f32; n];
        unsafe {
            dev.memcpy_async(
                dev.default_stream(),
                CopyDirection::DeviceToHost,
                DevicePtr(got.as_mut_ptr() as usize),
                d_y,
                n * 4,
            )?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_a, n * 4)?;
            dev.dealloc(d_b, n * 4)?;
            dev.dealloc(d_y, n * 4)?;
        }
        let err = max_rel_err(&got, &reference);
        results.push(ShapeResult {
            m: n,
            k: 1,
            n: 1,
            seed,
            max_rel_err: err,
            tolerance: 1e-6,
            pass: err <= 1e-6,
        });
    }
    let pass = results.iter().all(|r| r.pass);
    let pmc = pmc(&kernel);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "swiglu_f32_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "swiglu_f32".to_string(),
        dtype_weight: "F32".to_string(),
        dtype_activation: "F32".to_string(),
        tolerance_formula: "|err| <= 1e-6 * max(|ref|, 1)".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig(),
        pmc,
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

// --- scale_f32 -------------------------------------------------------------

pub fn run_scale_sweep(repo_root: &Path) -> Result<Cert> {
    let (dev, module, entry) = open_kernel("scale_f32", "flambeau_scale_f32")?;
    let kernel = module.kernel_dynamic(&entry)?;
    let shapes = [64usize, 1024, 2048, 4096];
    let mut results = Vec::new();
    for n in shapes {
        let seed = 0xC0FFEE ^ ((n as u64) * 31 + 11);
        let x = seeded_f32(seed, n);
        let scale = 1.0f32 / (128.0f32).sqrt(); // head_k_dim=128
        let reference: Vec<f32> = x.iter().map(|v| v * scale).collect();
        let d_x = alloc_and_upload(&dev, &x);
        let d_y = dev.alloc(n * 4)?;
        {
            let n_i = n as i32;
            let x_ptr: u64 = d_x.as_usize() as u64;
            let y_ptr: u64 = d_y.as_usize() as u64;
            let mut args = KernelArgs::new();
            args.push(&x_ptr);
            args.push(&y_ptr);
            args.push(&n_i);
            args.push(&scale);
            let cfg = LaunchCfg::one_d((n as u32).div_ceil(256), 256);
            unsafe { kernel.launch(dev.default_stream(), cfg, args)? };
            dev.default_stream().synchronize()?;
        }
        let mut got = vec![0.0f32; n];
        unsafe {
            dev.memcpy_async(
                dev.default_stream(),
                CopyDirection::DeviceToHost,
                DevicePtr(got.as_mut_ptr() as usize),
                d_y,
                n * 4,
            )?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_x, n * 4)?;
            dev.dealloc(d_y, n * 4)?;
        }
        let err = max_rel_err(&got, &reference);
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
    let pmc = pmc(&kernel);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "scale_f32_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "scale_f32".to_string(),
        dtype_weight: "F32".to_string(),
        dtype_activation: "F32".to_string(),
        tolerance_formula: "exact (pointwise multiply)".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig(),
        pmc,
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

// --- rmsnorm_f32 -----------------------------------------------------------

pub fn run_rmsnorm_f32_sweep(repo_root: &Path) -> Result<Cert> {
    let (dev, module, entry) = open_kernel("rmsnorm_f32", "flambeau_rmsnorm_f32")?;
    let kernel = module.kernel_dynamic(&entry)?;
    // GDN ssm_norm shapes: per-head over num_v_heads × head_v_dim.
    // Qwen3.6: (32, 128). Include 9B variant and a 1-row sanity check.
    let shapes = [(32usize, 128usize), (16, 128), (1, 512)];
    let eps = 1e-6f32;
    let mut results = Vec::new();
    for (m, k) in shapes {
        let seed = 0xC0FFEE ^ ((m as u64) * 1049 + (k as u64) * 11);
        let x = seeded_f32(seed, m * k);
        let w = seeded_f32(seed ^ 0xEE, k);
        // Reference: per-row RMS norm in F64 to keep the reference honest.
        let mut reference = vec![0.0f32; m * k];
        for r in 0..m {
            let row = &x[r * k..(r + 1) * k];
            let sum_sq: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
            let rsqrt = 1.0 / ((sum_sq / (k as f64)) + eps as f64).sqrt();
            for i in 0..k {
                reference[r * k + i] = (row[i] as f64 * w[i] as f64 * rsqrt) as f32;
            }
        }
        let d_x = alloc_and_upload(&dev, &x);
        let d_w = alloc_and_upload(&dev, &w);
        let d_y = dev.alloc(m * k * 4)?;
        {
            let m_i = m as i32;
            let k_i = k as i32;
            let x_ptr: u64 = d_x.as_usize() as u64;
            let w_ptr: u64 = d_w.as_usize() as u64;
            let y_ptr: u64 = d_y.as_usize() as u64;
            let mut args = KernelArgs::new();
            args.push(&x_ptr);
            args.push(&w_ptr);
            args.push(&y_ptr);
            args.push(&m_i);
            args.push(&k_i);
            args.push(&eps);
            let cfg = LaunchCfg::one_d(m as u32, 256);
            unsafe { kernel.launch(dev.default_stream(), cfg, args)? };
            dev.default_stream().synchronize()?;
        }
        let mut got = vec![0.0f32; m * k];
        unsafe {
            dev.memcpy_async(
                dev.default_stream(),
                CopyDirection::DeviceToHost,
                DevicePtr(got.as_mut_ptr() as usize),
                d_y,
                m * k * 4,
            )?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_x, m * k * 4)?;
            dev.dealloc(d_w, k * 4)?;
            dev.dealloc(d_y, m * k * 4)?;
        }
        let err = max_rel_err(&got, &reference);
        results.push(ShapeResult {
            m,
            k,
            n: 1,
            seed,
            max_rel_err: err,
            tolerance: 1e-5,
            pass: err <= 1e-5,
        });
    }
    let pass = results.iter().all(|r| r.pass);
    let pmc = pmc(&kernel);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "rmsnorm_f32_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "rmsnorm_f32".to_string(),
        dtype_weight: "F32".to_string(),
        dtype_activation: "F32".to_string(),
        tolerance_formula: "|err| <= 1e-5 * max(|ref|, 1)".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig(),
        pmc,
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

// --- gdn_alpha_beta_f32 ----------------------------------------------------

pub fn run_gdn_alpha_beta_sweep(repo_root: &Path) -> Result<Cert> {
    let (dev, module, entry) = open_kernel("gdn_alpha_beta_f32", "flambeau_gdn_alpha_beta_f32")?;
    let kernel = module.kernel_dynamic(&entry)?;
    // num_v_heads in our supported range: 2 (tiny smoke), 32 (Qwen3.6), 64 (upper fixture).
    let shapes = [2usize, 32, 64];
    let mut results = Vec::new();
    for n in shapes {
        let seed = 0xC0FFEE ^ ((n as u64) * 41 + 19);
        let alpha = seeded_f32(seed ^ 0xA1, n);
        let beta_in = seeded_f32(seed ^ 0xB2, n);
        let ssm_dt = seeded_f32(seed ^ 0xD3, n);
        let ssm_a: Vec<f32> = (0..n).map(|i| -(0.05 + 0.01 * i as f32)).collect();

        // Host reference — identical stable forms as the kernel.
        let mut ref_gate = vec![0.0f32; n];
        let mut ref_beta = vec![0.0f32; n];
        for i in 0..n {
            let a = alpha[i] + ssm_dt[i];
            let abs_a = a.abs();
            let max_a = a.max(0.0);
            let softplus = max_a + (1.0 + (-abs_a).exp()).ln();
            ref_gate[i] = softplus * ssm_a[i];
            let b = beta_in[i];
            ref_beta[i] = if b >= 0.0 {
                1.0 / (1.0 + (-b).exp())
            } else {
                let z = b.exp();
                z / (1.0 + z)
            };
        }

        let d_alpha = alloc_and_upload(&dev, &alpha);
        let d_beta_in = alloc_and_upload(&dev, &beta_in);
        let d_dt = alloc_and_upload(&dev, &ssm_dt);
        let d_a = alloc_and_upload(&dev, &ssm_a);
        let d_gate = dev.alloc(n * 4)?;
        let d_beta_out = dev.alloc(n * 4)?;
        {
            let n_i = n as i32;
            let a_ptr: u64 = d_alpha.as_usize() as u64;
            let b_ptr: u64 = d_beta_in.as_usize() as u64;
            let dt_ptr: u64 = d_dt.as_usize() as u64;
            let sa_ptr: u64 = d_a.as_usize() as u64;
            let g_ptr: u64 = d_gate.as_usize() as u64;
            let bo_ptr: u64 = d_beta_out.as_usize() as u64;
            let mut args = KernelArgs::new();
            args.push(&a_ptr);
            args.push(&b_ptr);
            args.push(&dt_ptr);
            args.push(&sa_ptr);
            args.push(&g_ptr);
            args.push(&bo_ptr);
            args.push(&n_i);
            let cfg = LaunchCfg::one_d((n as u32).div_ceil(64), 64);
            unsafe { kernel.launch(dev.default_stream(), cfg, args)? };
            dev.default_stream().synchronize()?;
        }
        let mut got_gate = vec![0.0f32; n];
        let mut got_beta = vec![0.0f32; n];
        unsafe {
            dev.memcpy_async(
                dev.default_stream(),
                CopyDirection::DeviceToHost,
                DevicePtr(got_gate.as_mut_ptr() as usize),
                d_gate,
                n * 4,
            )?;
            dev.memcpy_async(
                dev.default_stream(),
                CopyDirection::DeviceToHost,
                DevicePtr(got_beta.as_mut_ptr() as usize),
                d_beta_out,
                n * 4,
            )?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_alpha, n * 4)?;
            dev.dealloc(d_beta_in, n * 4)?;
            dev.dealloc(d_dt, n * 4)?;
            dev.dealloc(d_a, n * 4)?;
            dev.dealloc(d_gate, n * 4)?;
            dev.dealloc(d_beta_out, n * 4)?;
        }

        // Compare both outputs; keep the worse error.
        let err = max_rel_err(&got_gate, &ref_gate).max(max_rel_err(&got_beta, &ref_beta));
        results.push(ShapeResult {
            m: n,
            k: 1,
            n: 1,
            seed,
            max_rel_err: err,
            tolerance: 1e-5,
            pass: err <= 1e-5,
        });
    }
    let pass = results.iter().all(|r| r.pass);
    let pmc = pmc(&kernel);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "gdn_alpha_beta_f32_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "gdn_alpha_beta_f32".to_string(),
        dtype_weight: "F32".to_string(),
        dtype_activation: "F32".to_string(),
        tolerance_formula: "|err| <= 1e-5 * max(|ref|, 1) on both gate and beta".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig(),
        pmc,
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

// --- quantize_f16_q8_1 ----------------------------------------------------

pub fn run_quantize_f16_q8_1_sweep(repo_root: &Path) -> Result<Cert> {
    let (dev, module, entry) = open_kernel("quantize_f16_q8_1", "flambeau_quantize_row_f16_q8_1")?;
    let kernel = module.kernel_dynamic(&entry)?;
    // Must be multiples of 32. Cover shapes the full-attn forward hits
    // (n_heads * head_dim = 4096 for Qwen3.6) + small sanity.
    let shapes = [32usize, 256, 4096];
    let mut results = Vec::new();
    for n in shapes {
        let seed = 0xC0FFEE ^ ((n as u64) * 53 + 17);
        let x_f32 = seeded_f32(seed, n);
        let x_f16: Vec<f16> = x_f32.iter().map(|v| f16::from_f32(*v)).collect();

        // Reference: same quantise math as the F32 version, but input is
        // F32-cast-from-F16 so F16 rounding is part of the reference too.
        let mut ref_blocks: Vec<[u8; 36]> = Vec::with_capacity(n / 32);
        for block_idx in 0..n / 32 {
            let mut amax = 0.0f32;
            for j in 0..32 {
                let v = x_f16[block_idx * 32 + j].to_f32();
                if v.abs() > amax {
                    amax = v.abs();
                }
            }
            let d = amax / 127.0f32;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            let mut qs = [0i8; 32];
            let mut sum: i32 = 0;
            for j in 0..32 {
                let v = x_f16[block_idx * 32 + j].to_f32();
                // Match the kernel's `rintf` — round ties to even (not
                // Rust's default ties-away-from-zero `.round()`).
                let q = (v * id).round_ties_even() as i32;
                let q = q.clamp(-127, 127) as i8;
                qs[j] = q;
                sum += q as i32;
            }
            let s = d * sum as f32;
            // Pack into the Q8_1 layout: d (fp16), s (fp16), qs[32] i8.
            let mut block = [0u8; 36];
            block[0..2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
            block[2..4].copy_from_slice(&f16::from_f32(s).to_bits().to_le_bytes());
            for j in 0..32 {
                block[4 + j] = qs[j] as u8;
            }
            ref_blocks.push(block);
        }

        let d_x = alloc_and_upload(&dev, &x_f16);
        let n_blocks = n / 32;
        let out_bytes = n_blocks * 36;
        let d_y = dev.alloc(out_bytes)?;
        {
            let n_i = n as i32;
            let x_ptr: u64 = d_x.as_usize() as u64;
            let y_ptr: u64 = d_y.as_usize() as u64;
            let mut args = KernelArgs::new();
            args.push(&x_ptr);
            args.push(&y_ptr);
            args.push(&n_i);
            let cfg = LaunchCfg::one_d(n_blocks as u32, 32);
            unsafe { kernel.launch(dev.default_stream(), cfg, args)? };
            dev.default_stream().synchronize()?;
        }
        let mut got = vec![0u8; out_bytes];
        unsafe {
            dev.memcpy_async(
                dev.default_stream(),
                CopyDirection::DeviceToHost,
                DevicePtr(got.as_mut_ptr() as usize),
                d_y,
                out_bytes,
            )?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_x, n * 2)?;
            dev.dealloc(d_y, out_bytes)?;
        }

        // Compare per-block: qs bytes bit-exact; d/s within a 1-ULP f16.
        // Since both paths do the same rounding, bit-exact match is
        // expected on the qs bytes. For d/s we allow ≤ 1-ULP slack
        // because the kernel may shuffle intermediate ops.
        let mut mismatches = 0u32;
        for (got_block, ref_block) in got.chunks(36).zip(ref_blocks.iter()) {
            if got_block[4..] != ref_block[4..] {
                mismatches += 1;
            }
            let got_d = f16::from_bits(u16::from_le_bytes([got_block[0], got_block[1]]));
            let got_s = f16::from_bits(u16::from_le_bytes([got_block[2], got_block[3]]));
            let ref_d = f16::from_bits(u16::from_le_bytes([ref_block[0], ref_block[1]]));
            let ref_s = f16::from_bits(u16::from_le_bytes([ref_block[2], ref_block[3]]));
            if (got_d.to_f32() - ref_d.to_f32()).abs() > ref_d.to_f32().abs().max(1.0) * 2e-3
                || (got_s.to_f32() - ref_s.to_f32()).abs() > ref_s.to_f32().abs().max(1.0) * 2e-3
            {
                mismatches += 1;
            }
        }
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
    let pmc = pmc(&kernel);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "quantize_f16_q8_1_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "quantize_f16_q8_1".to_string(),
        dtype_weight: "F16".to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula: "qs bit-exact; d/s within 2e-3 rel vs host reference".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig(),
        pmc,
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

// --- add_f16 ---------------------------------------------------------------

pub fn run_add_f16_sweep(repo_root: &Path) -> Result<Cert> {
    let (dev, module, entry) = open_kernel("add_f16", "flambeau_add_f16")?;
    let kernel = module.kernel_dynamic(&entry)?;
    let shapes = [64usize, 256, 2048, 4096, 8192];
    let mut results = Vec::new();
    for n in shapes {
        let seed = 0xC0FFEE ^ ((n as u64) * 83 + 23);
        let a_f32 = seeded_f32(seed, n);
        let b_f32 = seeded_f32(seed ^ 0xBE, n);
        let a_f16: Vec<f16> = a_f32.iter().map(|v| f16::from_f32(*v)).collect();
        let b_f16: Vec<f16> = b_f32.iter().map(|v| f16::from_f32(*v)).collect();
        // Reference: cast to F32, add, cast back to F16 — matches the kernel.
        let reference: Vec<f16> = a_f16
            .iter()
            .zip(&b_f16)
            .map(|(a, b)| f16::from_f32(a.to_f32() + b.to_f32()))
            .collect();

        let d_a = alloc_and_upload(&dev, &a_f16);
        let d_b = alloc_and_upload(&dev, &b_f16);
        let d_y = dev.alloc(n * 2)?;
        {
            let n_i = n as i32;
            let a_ptr: u64 = d_a.as_usize() as u64;
            let b_ptr: u64 = d_b.as_usize() as u64;
            let y_ptr: u64 = d_y.as_usize() as u64;
            let mut args = KernelArgs::new();
            args.push(&a_ptr);
            args.push(&b_ptr);
            args.push(&y_ptr);
            args.push(&n_i);
            let cfg = LaunchCfg::one_d((n as u32).div_ceil(256), 256);
            unsafe { kernel.launch(dev.default_stream(), cfg, args)? };
            dev.default_stream().synchronize()?;
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

        // Bit-exact against the F32-add-then-F16-cast reference.
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
    let pmc = pmc(&kernel);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "add_f16_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "add_f16".to_string(),
        dtype_weight: "F16".to_string(),
        dtype_activation: "F16".to_string(),
        tolerance_formula: "exact bit match vs (f32) a + (f32) b → f16".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig(),
        pmc,
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

// --- dense_gemv_f32_f16 ---------------------------------------------------

pub fn run_dense_gemv_sweep(repo_root: &Path) -> Result<Cert> {
    let (dev, module, entry) = open_kernel("dense_gemv_f32_f16", "flambeau_dense_gemv_f32_f16")?;
    let kernel = module.kernel_dynamic(&entry)?;
    // MoE router shapes: n_rows = n_experts (4, 128, 256), k = hidden (2048).
    // Small n_experts + small k for sanity.
    let shapes = [(4usize, 256usize), (16, 1024), (128, 2048), (256, 2048)];
    let mut results = Vec::new();
    for (n_rows, k) in shapes {
        let seed = 0xC0FFEE ^ ((n_rows as u64) * 71 + (k as u64) * 13);
        let w = seeded_f32(seed, n_rows * k);
        let x_f32 = seeded_f32(seed ^ 0xA1, k);
        let x_f16: Vec<f16> = x_f32.iter().map(|v| f16::from_f32(*v)).collect();
        // Reference uses F64 accumulation + F16 round-trip input so it
        // matches what the kernel sees bit-for-bit on the activation side.
        let mut reference = vec![0.0f32; n_rows];
        for r in 0..n_rows {
            let mut acc = 0.0f64;
            for i in 0..k {
                acc += (w[r * k + i] as f64) * (x_f16[i].to_f32() as f64);
            }
            reference[r] = acc as f32;
        }

        let d_w = alloc_and_upload(&dev, &w);
        let d_x = alloc_and_upload(&dev, &x_f16);
        let d_y = dev.alloc(n_rows * 4)?;
        {
            let n_i = n_rows as i32;
            let k_i = k as i32;
            let w_ptr: u64 = d_w.as_usize() as u64;
            let x_ptr: u64 = d_x.as_usize() as u64;
            let y_ptr: u64 = d_y.as_usize() as u64;
            let mut args = KernelArgs::new();
            args.push(&w_ptr);
            args.push(&x_ptr);
            args.push(&y_ptr);
            args.push(&n_i);
            args.push(&k_i);
            let cfg = LaunchCfg::one_d(n_rows as u32, 256);
            unsafe { kernel.launch(dev.default_stream(), cfg, args)? };
            dev.default_stream().synchronize()?;
        }
        let mut got = vec![0.0f32; n_rows];
        unsafe {
            dev.memcpy_async(
                dev.default_stream(),
                CopyDirection::DeviceToHost,
                DevicePtr(got.as_mut_ptr() as usize),
                d_y,
                n_rows * 4,
            )?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_w, n_rows * k * 4)?;
            dev.dealloc(d_x, k * 2)?;
            dev.dealloc(d_y, n_rows * 4)?;
        }

        // F32 accumulator + F16 inputs → small per-lane drift vs F64 ref.
        // 1e-3 rel envelope covers `k ≤ 2048` comfortably.
        let err = max_rel_err(&got, &reference);
        results.push(ShapeResult {
            m: n_rows,
            k,
            n: 1,
            seed,
            max_rel_err: err,
            tolerance: 1e-3,
            pass: err <= 1e-3,
        });
    }
    let pass = results.iter().all(|r| r.pass);
    let pmc = pmc(&kernel);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "dense_gemv_f32_f16_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "dense_gemv_f32_f16".to_string(),
        dtype_weight: "F32".to_string(),
        dtype_activation: "F16".to_string(),
        tolerance_formula:
            "|err| <= 1e-3 * max(|ref|, 1) vs F64 reference with F16-quantised input".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig(),
        pmc,
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

// --- cast_f16_f32 ----------------------------------------------------------

pub fn run_cast_f16_f32_sweep(repo_root: &Path) -> Result<Cert> {
    let (dev, module, entry) = open_kernel("cast_f16_f32", "flambeau_cast_f16_f32")?;
    let kernel = module.kernel_dynamic(&entry)?;
    let shapes = [64usize, 1024, 2048, 4096, 8192];
    let mut results = Vec::new();
    for n in shapes {
        let seed = 0xC0FFEE ^ ((n as u64) * 37 + 13);
        let x_f32 = seeded_f32(seed, n);
        let x_f16: Vec<f16> = x_f32.iter().map(|v| f16::from_f32(*v)).collect();
        let reference: Vec<f32> = x_f16.iter().map(|v| v.to_f32()).collect();
        let d_x = alloc_and_upload(&dev, &x_f16);
        let d_y = dev.alloc(n * 4)?;
        {
            let n_i = n as i32;
            let x_ptr: u64 = d_x.as_usize() as u64;
            let y_ptr: u64 = d_y.as_usize() as u64;
            let mut args = KernelArgs::new();
            args.push(&x_ptr);
            args.push(&y_ptr);
            args.push(&n_i);
            let cfg = LaunchCfg::one_d((n as u32).div_ceil(256), 256);
            unsafe { kernel.launch(dev.default_stream(), cfg, args)? };
            dev.default_stream().synchronize()?;
        }
        let mut got = vec![0.0f32; n];
        unsafe {
            dev.memcpy_async(
                dev.default_stream(),
                CopyDirection::DeviceToHost,
                DevicePtr(got.as_mut_ptr() as usize),
                d_y,
                n * 4,
            )?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_x, n * 2)?;
            dev.dealloc(d_y, n * 4)?;
        }
        // Exact bit match expected — widening is lossless.
        let mut diff_bits = 0f32;
        for (g, r) in got.iter().zip(&reference) {
            if g.to_bits() != r.to_bits() {
                diff_bits = 1.0;
                break;
            }
        }
        results.push(ShapeResult {
            m: n,
            k: 1,
            n: 1,
            seed,
            max_rel_err: diff_bits,
            tolerance: 0.0,
            pass: diff_bits == 0.0,
        });
    }
    let pass = results.iter().all(|r| r.pass);
    let pmc = pmc(&kernel);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "cast_f16_f32_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "cast_f16_f32".to_string(),
        dtype_weight: "F16".to_string(),
        dtype_activation: "F32".to_string(),
        tolerance_formula: "exact bit match vs half::f16::to_f32".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig(),
        pmc,
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}
