//! 3.a — Q4_0 dense, Q5_0 dense, and Q4_0 indexed-MoE MMVQ correctness
//! sweeps. Each kernel is validated against a CPU F32 reference using
//! Q8_1-roundtripped activations (same envelope as other MMVQ certs).

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
use flambeau_quant::{BlockQ4_0, BlockQ4_1, BlockQ5_0, BlockQ8_1, QK8_0};

use crate::cert::{now_utc_iso8601, Cert, PmcSnapshot, ShapeResult, SCHEMA_VERSION};
use crate::harness::{max_rel_err_with_floor, rig, seeded_f32_range};

pub fn run_mmvq_q4_0_sweep(repo_root: &Path) -> Result<Cert> {
    run_dense_sweep(
        repo_root,
        "mmvq_q4_0",
        "flambeau_mmvq_q4_0_q8_1",
        "mmvq_q4_0_gfx906",
        "Q4_0",
        /*encode*/ encode_q4_0,
    )
}

pub fn run_mmvq_q5_0_sweep(repo_root: &Path) -> Result<Cert> {
    run_dense_sweep(
        repo_root,
        "mmvq_q5_0",
        "flambeau_mmvq_q5_0_q8_1",
        "mmvq_q5_0_gfx906",
        "Q5_0",
        /*encode*/ encode_q5_0,
    )
}

pub fn run_mmvq_q5_1_sweep(repo_root: &Path) -> Result<Cert> {
    run_dense_sweep(
        repo_root,
        "mmvq_q5_1",
        "flambeau_mmvq_q5_1_q8_1",
        "mmvq_q5_1_gfx906",
        "Q5_1",
        /*encode*/ encode_q5_1,
    )
}

type EncodeFn = fn(&[f32]) -> Vec<u8>;

/// **C6-i1** — Q4_0 single-warp (64 t/block) MMVQ correctness sweep.
/// Same shapes as `run_mmvq_q4_0_sweep`, only the block-thread count and
/// kernel stem change.
pub fn run_mmvq_q4_0_warpcoop64_sweep(repo_root: &Path) -> Result<Cert> {
    run_dense_sweep_with_block(
        repo_root,
        "mmvq_q4_0_warpcoop64",
        "flambeau_mmvq_q4_0_warpcoop64_q8_1",
        "mmvq_q4_0_warpcoop64_gfx906",
        "Q4_0",
        encode_q4_0,
        64,
    )
}

fn run_dense_sweep(
    repo_root: &Path,
    stem: &str,
    entry: &str,
    impl_id: &str,
    dtype_weight: &str,
    encode: EncodeFn,
) -> Result<Cert> {
    run_dense_sweep_with_block(repo_root, stem, entry, impl_id, dtype_weight, encode, 256)
}

fn run_dense_sweep_with_block(
    repo_root: &Path,
    stem: &str,
    entry: &str,
    impl_id: &str,
    dtype_weight: &str,
    encode: EncodeFn,
    block_threads: u32,
) -> Result<Cert> {
    if device_count().context("hipGetDeviceCount")? < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;
    let kb = kernels::hsaco(stem).unwrap();
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel_dynamic(entry)?;
    let attrs: FuncAttributes = kernel.attributes()?;
    let q_kb = kernels::hsaco("quantize_q8_1").unwrap();
    let q_module = HipModule::load(dev.id(), q_kb)?;
    let q_kernel: HipKernel<'_> = q_module.kernel("flambeau_quantize_row_q8_1")?;

    // (n_rows, k)
    let shapes = [(256usize, 256usize), (2048, 2048), (6144, 5120)];
    let mut results = Vec::new();
    for (n_rows, k) in shapes {
        let seed = 0x4E0Cu64 ^ (n_rows as u64 * 7919) ^ (k as u64 * 101);
        let max_rel = run_dense_shape(&dev, &kernel, &q_kernel, n_rows, k, seed, encode, block_threads)?;
        let tol = 3e-2;
        results.push(ShapeResult {
            m: 1,
            k,
            n: n_rows,
            seed,
            max_rel_err: max_rel,
            tolerance: tol,
            pass: max_rel <= tol,
        });
    }
    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: impl_id.to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "qmatmul_mmvq".to_string(),
        dtype_weight: dtype_weight.to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula: "|err| <= 3e-2 * max(|ref|, sqrt(k))".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig(),
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

fn run_dense_shape(
    dev: &HipDevice,
    kernel: &HipKernel<'_>,
    q_kernel: &HipKernel<'_>,
    n_rows: usize,
    k: usize,
    seed: u64,
    encode: EncodeFn,
    block_threads: u32,
) -> Result<f32> {
    let w_f32 = seeded_f32_range(seed, n_rows * k, -0.5, 0.5);
    let x_f32 = seeded_f32_range(seed.wrapping_add(0xA1), k, -0.5, 0.5);
    let w_bytes = encode(&w_f32);
    let n_blocks_per_row = k / 32;

    let d_w = upload(dev, &w_bytes);
    let d_x = upload(dev, &x_f32);
    let d_y = dev.alloc(n_blocks_per_row * std::mem::size_of::<BlockQ8_1>())?;
    let d_out = dev.alloc(n_rows * 4)?;

    // F32 → Q8_1.
    {
        let stream = dev.default_stream();
        let n_elems = k as i32;
        let d_x_p: u64 = d_x.as_usize() as u64;
        let d_y_p: u64 = d_y.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_x_p);
        args.push(&d_y_p);
        args.push(&n_elems);
        let cfg = LaunchCfg::one_d(n_blocks_per_row as u32, 32);
        unsafe { q_kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    // Kernel launch.
    {
        let stream = dev.default_stream();
        let n_rows_i = n_rows as i32;
        let n_blocks_i = n_blocks_per_row as i32;
        let w_p: u64 = d_w.as_usize() as u64;
        let y_p: u64 = d_y.as_usize() as u64;
        let o_p: u64 = d_out.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&w_p);
        args.push(&y_p);
        args.push(&o_p);
        args.push(&n_rows_i);
        args.push(&n_blocks_i);
        let cfg = LaunchCfg::one_d(n_rows as u32, block_threads);
        unsafe { kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    let mut got = vec![0.0f32; n_rows];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got.as_mut_ptr() as usize),
            d_out,
            n_rows * 4,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_w, w_bytes.len())?;
        dev.dealloc(d_x, x_f32.len() * 4)?;
        dev.dealloc(d_y, n_blocks_per_row * std::mem::size_of::<BlockQ8_1>())?;
        dev.dealloc(d_out, n_rows * 4)?;
    }

    // Reference: CPU dequant (via the ggml dtype helper) × Q8_1-roundtrip activation.
    let mut w_dequant = vec![0.0f32; n_rows * k];
    let block_bytes = w_bytes.len() / n_blocks_per_row / n_rows;
    let dtype = match block_bytes {
        18 => flambeau_quant::GgmlDType::Q4_0,   // 2 d + 16 nibbles
        22 => flambeau_quant::GgmlDType::Q5_0,   // 2 d + 4 qh + 16 nibbles
        24 => flambeau_quant::GgmlDType::Q5_1,   // 2 d + 2 m + 4 qh + 16 nibbles
        other => panic!("unexpected block size {other}"),
    };
    flambeau_quant::dequantize_into(dtype, &w_bytes, &mut w_dequant)?;
    let x_rt = q8_1_roundtrip(&x_f32);
    let mut reference = vec![0.0f32; n_rows];
    for row in 0..n_rows {
        let mut acc = 0.0f64;
        for j in 0..k {
            acc += (w_dequant[row * k + j] * x_rt[j]) as f64;
        }
        reference[row] = acc as f32;
    }
    Ok(max_rel_err_with_floor(&got, &reference, (k as f32).sqrt() * 0.01))
}

pub fn run_indexed_moe_mmvq_q4_0_sweep(repo_root: &Path) -> Result<Cert> {
    if device_count().context("hipGetDeviceCount")? < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;
    let kb = kernels::hsaco("indexed_moe_mmvq_q4_0").unwrap();
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> =
        module.kernel("flambeau_indexed_moe_mmvq_q4_0_q8_1")?;
    let attrs: FuncAttributes = kernel.attributes()?;
    let q_kb = kernels::hsaco("quantize_q8_1").unwrap();
    let q_module = HipModule::load(dev.id(), q_kb)?;
    let q_kernel: HipKernel<'_> = q_module.kernel("flambeau_quantize_row_q8_1")?;

    let n_experts = 16usize;
    let cases = [
        (1usize, 4usize, 256usize, 2048usize),
        (4, 4, 256, 2048),
    ];
    let mut results = Vec::new();
    for (n_tokens, top_k, n_rows, k_dim) in cases {
        let seed = 0x4EC0u64
            ^ (n_tokens as u64 * 53 + top_k as u64 * 17 + n_rows as u64 * 7);
        let max_rel = run_q4_0_moe_shape(
            &dev, &kernel, &q_kernel, n_experts, n_rows, k_dim, top_k, n_tokens, seed,
        )?;
        let tol = 3e-2;
        results.push(ShapeResult {
            m: n_tokens,
            k: k_dim,
            n: n_rows,
            seed,
            max_rel_err: max_rel,
            tolerance: tol,
            pass: max_rel <= tol,
        });
    }
    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "indexed_moe_mmvq_q4_0_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "indexed_moe_mmvq".to_string(),
        dtype_weight: "Q4_0".to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula: "|err| <= 3e-2 * max(|ref|, sqrt(k))".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig(),
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

pub fn run_indexed_moe_mmvq_q5_0_sweep(repo_root: &Path) -> Result<Cert> {
    if device_count().context("hipGetDeviceCount")? < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;
    let kb = kernels::hsaco("indexed_moe_mmvq_q5_0").unwrap();
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> =
        module.kernel("flambeau_indexed_moe_mmvq_q5_0_q8_1")?;
    let attrs: FuncAttributes = kernel.attributes()?;
    let q_kb = kernels::hsaco("quantize_q8_1").unwrap();
    let q_module = HipModule::load(dev.id(), q_kb)?;
    let q_kernel: HipKernel<'_> = q_module.kernel("flambeau_quantize_row_q8_1")?;

    let n_experts = 16usize;
    let cases = [
        (1usize, 4usize, 256usize, 2048usize),
        (4, 4, 256, 2048),
    ];
    let mut results = Vec::new();
    for (n_tokens, top_k, n_rows, k_dim) in cases {
        let seed = 0x5EC0u64
            ^ (n_tokens as u64 * 53 + top_k as u64 * 17 + n_rows as u64 * 7);
        let max_rel = run_q5_0_moe_shape(
            &dev, &kernel, &q_kernel, n_experts, n_rows, k_dim, top_k, n_tokens, seed,
        )?;
        let tol = 3e-2;
        results.push(ShapeResult {
            m: n_tokens,
            k: k_dim,
            n: n_rows,
            seed,
            max_rel_err: max_rel,
            tolerance: tol,
            pass: max_rel <= tol,
        });
    }
    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "indexed_moe_mmvq_q5_0_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "indexed_moe_mmvq".to_string(),
        dtype_weight: "Q5_0".to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula: "|err| <= 3e-2 * max(|ref|, sqrt(k))".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig(),
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

pub fn run_indexed_moe_mmvq_q5_1_sweep(repo_root: &Path) -> Result<Cert> {
    if device_count().context("hipGetDeviceCount")? < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;
    let kb = kernels::hsaco("indexed_moe_mmvq_q5_1").unwrap();
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> =
        module.kernel("flambeau_indexed_moe_mmvq_q5_1_q8_1")?;
    let attrs: FuncAttributes = kernel.attributes()?;
    let q_kb = kernels::hsaco("quantize_q8_1").unwrap();
    let q_module = HipModule::load(dev.id(), q_kb)?;
    let q_kernel: HipKernel<'_> = q_module.kernel("flambeau_quantize_row_q8_1")?;

    let n_experts = 16usize;
    let cases = [
        (1usize, 4usize, 256usize, 2048usize),
        (4, 4, 256, 2048),
    ];
    let mut results = Vec::new();
    for (n_tokens, top_k, n_rows, k_dim) in cases {
        let seed = 0x5EC1u64
            ^ (n_tokens as u64 * 53 + top_k as u64 * 17 + n_rows as u64 * 7);
        let max_rel = run_q5_1_moe_shape(
            &dev, &kernel, &q_kernel, n_experts, n_rows, k_dim, top_k, n_tokens, seed,
        )?;
        let tol = 3e-2;
        results.push(ShapeResult {
            m: n_tokens,
            k: k_dim,
            n: n_rows,
            seed,
            max_rel_err: max_rel,
            tolerance: tol,
            pass: max_rel <= tol,
        });
    }
    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "indexed_moe_mmvq_q5_1_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "indexed_moe_mmvq".to_string(),
        dtype_weight: "Q5_1".to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula: "|err| <= 3e-2 * max(|ref|, sqrt(k))".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig(),
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

/// B6 / 5.a — Q4_1 indexed-MoE MMVQ correctness sweep. Mirrors
/// `run_indexed_moe_mmvq_q4_0_sweep`; only the encoder + reference
/// reconstruction change (Q4_1 affine `d*q + m` instead of Q4_0
/// symmetric `d*(q-8)`).
pub fn run_indexed_moe_mmvq_q4_1_sweep(repo_root: &Path) -> Result<Cert> {
    if device_count().context("hipGetDeviceCount")? < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;
    let kb = kernels::hsaco("indexed_moe_mmvq_q4_1").unwrap();
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> =
        module.kernel("flambeau_indexed_moe_mmvq_q4_1_q8_1")?;
    let attrs: FuncAttributes = kernel.attributes()?;
    let q_kb = kernels::hsaco("quantize_q8_1").unwrap();
    let q_module = HipModule::load(dev.id(), q_kb)?;
    let q_kernel: HipKernel<'_> = q_module.kernel("flambeau_quantize_row_q8_1")?;

    let n_experts = 16usize;
    // Mirror Q4_0 cases plus a 35B-A3B-shape (k=512, n=2048, top_k=8) row that
    // exercises the actual production shape for `ffn_down_exps` Q4_1.
    let cases = [
        (1usize, 4usize, 256usize, 2048usize),
        (4, 4, 256, 2048),
        (1, 8, 2048, 512),
    ];
    let mut results = Vec::new();
    for (n_tokens, top_k, n_rows, k_dim) in cases {
        let seed = 0x4EC1u64
            ^ (n_tokens as u64 * 53 + top_k as u64 * 17 + n_rows as u64 * 7);
        let max_rel = run_q4_1_moe_shape(
            &dev, &kernel, &q_kernel, n_experts, n_rows, k_dim, top_k, n_tokens, seed,
        )?;
        let tol = 3e-2;
        results.push(ShapeResult {
            m: n_tokens,
            k: k_dim,
            n: n_rows,
            seed,
            max_rel_err: max_rel,
            tolerance: tol,
            pass: max_rel <= tol,
        });
    }
    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "indexed_moe_mmvq_q4_1_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "indexed_moe_mmvq".to_string(),
        dtype_weight: "Q4_1".to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula: "|err| <= 3e-2 * max(|ref|, sqrt(k))".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig(),
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

fn run_q4_1_moe_shape(
    dev: &HipDevice,
    kernel: &HipKernel<'_>,
    q_kernel: &HipKernel<'_>,
    n_experts: usize,
    n_rows: usize,
    k_dim: usize,
    top_k: usize,
    n_tokens: usize,
    seed: u64,
) -> Result<f32> {
    let nb_per_row = k_dim / 32;
    let w_elems = n_experts * n_rows * k_dim;
    let w_f32 = seeded_f32_range(seed, w_elems, -0.5, 0.5);
    let w_bytes = encode_q4_1(&w_f32);

    let act_f32 = seeded_f32_range(seed.wrapping_add(0xA1), n_tokens * k_dim, -0.5, 0.5);
    let expert_ids: Vec<i32> = (0..n_tokens * top_k)
        .map(|i| {
            let h = (i as u64)
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(seed.wrapping_mul(0x12345));
            ((h >> 32) as u32 % n_experts as u32) as i32
        })
        .collect();

    let d_w = upload(dev, &w_bytes);
    let d_act = upload(dev, &act_f32);
    let y_blocks_total = n_tokens * nb_per_row;
    let d_y = dev.alloc(y_blocks_total * std::mem::size_of::<BlockQ8_1>())?;
    let d_ids = upload(dev, &expert_ids);
    let d_dst = dev.alloc(n_tokens * top_k * n_rows * 4)?;

    {
        let stream = dev.default_stream();
        let n_elems = (n_tokens * k_dim) as i32;
        let d_a_p: u64 = d_act.as_usize() as u64;
        let d_y_p: u64 = d_y.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_a_p);
        args.push(&d_y_p);
        args.push(&n_elems);
        let cfg = LaunchCfg::one_d(y_blocks_total as u32, 32);
        unsafe { q_kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    {
        let stream = dev.default_stream();
        let n_rows_i = n_rows as i32;
        let n_tokens_i = n_tokens as i32;
        let top_k_i = top_k as i32;
        let nb_i = nb_per_row as i32;
        let d_w_p: u64 = d_w.as_usize() as u64;
        let d_y_p: u64 = d_y.as_usize() as u64;
        let d_ids_p: u64 = d_ids.as_usize() as u64;
        let d_dst_p: u64 = d_dst.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_w_p);
        args.push(&d_y_p);
        args.push(&d_ids_p);
        args.push(&d_dst_p);
        args.push(&n_rows_i);
        args.push(&n_tokens_i);
        args.push(&top_k_i);
        args.push(&nb_i);
        let cfg = LaunchCfg {
            grid: (n_rows as u32, (n_tokens * top_k) as u32, 1),
            block: (256, 1, 1),
            shared_bytes: 0,
        };
        unsafe { kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    let mut got = vec![0.0f32; n_tokens * top_k * n_rows];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got.as_mut_ptr() as usize),
            d_dst,
            n_tokens * top_k * n_rows * 4,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_w, w_bytes.len())?;
        dev.dealloc(d_act, act_f32.len() * 4)?;
        dev.dealloc(d_y, y_blocks_total * std::mem::size_of::<BlockQ8_1>())?;
        dev.dealloc(d_ids, expert_ids.len() * 4)?;
        dev.dealloc(d_dst, n_tokens * top_k * n_rows * 4)?;
    }

    let mut w_dequant = vec![0.0f32; w_elems];
    flambeau_quant::dequantize_into(flambeau_quant::GgmlDType::Q4_1, &w_bytes, &mut w_dequant)?;
    let act_rt = q8_1_roundtrip(&act_f32);
    let mut reference = vec![0.0f32; n_tokens * top_k * n_rows];
    for t in 0..n_tokens {
        for slot in 0..top_k {
            let expert = expert_ids[t * top_k + slot] as usize;
            for row in 0..n_rows {
                let mut acc = 0.0f64;
                for j in 0..k_dim {
                    let w = w_dequant[((expert * n_rows) + row) * k_dim + j];
                    let a = act_rt[t * k_dim + j];
                    acc += (w * a) as f64;
                }
                reference[(t * top_k + slot) * n_rows + row] = acc as f32;
            }
        }
    }
    Ok(max_rel_err_with_floor(&got, &reference, (k_dim as f32).sqrt() * 0.01))
}

fn run_q4_0_moe_shape(
    dev: &HipDevice,
    kernel: &HipKernel<'_>,
    q_kernel: &HipKernel<'_>,
    n_experts: usize,
    n_rows: usize,
    k_dim: usize,
    top_k: usize,
    n_tokens: usize,
    seed: u64,
) -> Result<f32> {
    let nb_per_row = k_dim / 32;
    let w_elems = n_experts * n_rows * k_dim;
    let w_f32 = seeded_f32_range(seed, w_elems, -0.5, 0.5);
    let w_bytes = encode_q4_0(&w_f32);

    let act_f32 = seeded_f32_range(seed.wrapping_add(0xA1), n_tokens * k_dim, -0.5, 0.5);
    let expert_ids: Vec<i32> = (0..n_tokens * top_k)
        .map(|i| {
            let h = (i as u64)
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(seed.wrapping_mul(0x12345));
            ((h >> 32) as u32 % n_experts as u32) as i32
        })
        .collect();

    let d_w = upload(dev, &w_bytes);
    let d_act = upload(dev, &act_f32);
    let y_blocks_total = n_tokens * nb_per_row;
    let d_y = dev.alloc(y_blocks_total * std::mem::size_of::<BlockQ8_1>())?;
    let d_ids = upload(dev, &expert_ids);
    let d_dst = dev.alloc(n_tokens * top_k * n_rows * 4)?;

    // Activation F32 → Q8_1.
    {
        let stream = dev.default_stream();
        let n_elems = (n_tokens * k_dim) as i32;
        let d_a_p: u64 = d_act.as_usize() as u64;
        let d_y_p: u64 = d_y.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_a_p);
        args.push(&d_y_p);
        args.push(&n_elems);
        let cfg = LaunchCfg::one_d(y_blocks_total as u32, 32);
        unsafe { q_kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    // Launch MoE Q4_0 MMVQ.
    {
        let stream = dev.default_stream();
        let n_rows_i = n_rows as i32;
        let n_tokens_i = n_tokens as i32;
        let top_k_i = top_k as i32;
        let nb_i = nb_per_row as i32;
        let d_w_p: u64 = d_w.as_usize() as u64;
        let d_y_p: u64 = d_y.as_usize() as u64;
        let d_ids_p: u64 = d_ids.as_usize() as u64;
        let d_dst_p: u64 = d_dst.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_w_p);
        args.push(&d_y_p);
        args.push(&d_ids_p);
        args.push(&d_dst_p);
        args.push(&n_rows_i);
        args.push(&n_tokens_i);
        args.push(&top_k_i);
        args.push(&nb_i);
        let cfg = LaunchCfg {
            grid: (n_rows as u32, (n_tokens * top_k) as u32, 1),
            block: (256, 1, 1),
            shared_bytes: 0,
        };
        unsafe { kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    let mut got = vec![0.0f32; n_tokens * top_k * n_rows];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got.as_mut_ptr() as usize),
            d_dst,
            n_tokens * top_k * n_rows * 4,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_w, w_bytes.len())?;
        dev.dealloc(d_act, act_f32.len() * 4)?;
        dev.dealloc(d_y, y_blocks_total * std::mem::size_of::<BlockQ8_1>())?;
        dev.dealloc(d_ids, expert_ids.len() * 4)?;
        dev.dealloc(d_dst, n_tokens * top_k * n_rows * 4)?;
    }

    let mut w_dequant = vec![0.0f32; w_elems];
    flambeau_quant::dequantize_into(flambeau_quant::GgmlDType::Q4_0, &w_bytes, &mut w_dequant)?;
    let act_rt = q8_1_roundtrip(&act_f32);
    let mut reference = vec![0.0f32; n_tokens * top_k * n_rows];
    for t in 0..n_tokens {
        for slot in 0..top_k {
            let expert = expert_ids[t * top_k + slot] as usize;
            for row in 0..n_rows {
                let mut acc = 0.0f64;
                for j in 0..k_dim {
                    let w = w_dequant[((expert * n_rows) + row) * k_dim + j];
                    let a = act_rt[t * k_dim + j];
                    acc += (w * a) as f64;
                }
                reference[(t * top_k + slot) * n_rows + row] = acc as f32;
            }
        }
    }
    Ok(max_rel_err_with_floor(&got, &reference, (k_dim as f32).sqrt() * 0.01))
}

#[allow(clippy::too_many_arguments)]
fn run_q5_0_moe_shape(
    dev: &HipDevice,
    kernel: &HipKernel<'_>,
    q_kernel: &HipKernel<'_>,
    n_experts: usize,
    n_rows: usize,
    k_dim: usize,
    top_k: usize,
    n_tokens: usize,
    seed: u64,
) -> Result<f32> {
    let nb_per_row = k_dim / 32;
    let w_elems = n_experts * n_rows * k_dim;
    let w_f32 = seeded_f32_range(seed, w_elems, -0.5, 0.5);
    let w_bytes = encode_q5_0(&w_f32);

    let act_f32 = seeded_f32_range(seed.wrapping_add(0xA1), n_tokens * k_dim, -0.5, 0.5);
    let expert_ids: Vec<i32> = (0..n_tokens * top_k)
        .map(|i| {
            let h = (i as u64)
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(seed.wrapping_mul(0x12345));
            ((h >> 32) as u32 % n_experts as u32) as i32
        })
        .collect();

    let d_w = upload(dev, &w_bytes);
    let d_act = upload(dev, &act_f32);
    let y_blocks_total = n_tokens * nb_per_row;
    let d_y = dev.alloc(y_blocks_total * std::mem::size_of::<BlockQ8_1>())?;
    let d_ids = upload(dev, &expert_ids);
    let d_dst = dev.alloc(n_tokens * top_k * n_rows * 4)?;

    {
        let stream = dev.default_stream();
        let n_elems = (n_tokens * k_dim) as i32;
        let d_a_p: u64 = d_act.as_usize() as u64;
        let d_y_p: u64 = d_y.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_a_p);
        args.push(&d_y_p);
        args.push(&n_elems);
        let cfg = LaunchCfg::one_d(y_blocks_total as u32, 32);
        unsafe { q_kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    {
        let stream = dev.default_stream();
        let n_rows_i = n_rows as i32;
        let n_tokens_i = n_tokens as i32;
        let top_k_i = top_k as i32;
        let nb_i = nb_per_row as i32;
        let d_w_p: u64 = d_w.as_usize() as u64;
        let d_y_p: u64 = d_y.as_usize() as u64;
        let d_ids_p: u64 = d_ids.as_usize() as u64;
        let d_dst_p: u64 = d_dst.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_w_p);
        args.push(&d_y_p);
        args.push(&d_ids_p);
        args.push(&d_dst_p);
        args.push(&n_rows_i);
        args.push(&n_tokens_i);
        args.push(&top_k_i);
        args.push(&nb_i);
        let cfg = LaunchCfg {
            grid: (n_rows as u32, (n_tokens * top_k) as u32, 1),
            block: (256, 1, 1),
            shared_bytes: 0,
        };
        unsafe { kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    let mut got = vec![0.0f32; n_tokens * top_k * n_rows];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got.as_mut_ptr() as usize),
            d_dst,
            n_tokens * top_k * n_rows * 4,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_w, w_bytes.len())?;
        dev.dealloc(d_act, act_f32.len() * 4)?;
        dev.dealloc(d_y, y_blocks_total * std::mem::size_of::<BlockQ8_1>())?;
        dev.dealloc(d_ids, expert_ids.len() * 4)?;
        dev.dealloc(d_dst, n_tokens * top_k * n_rows * 4)?;
    }

    let mut w_dequant = vec![0.0f32; w_elems];
    flambeau_quant::dequantize_into(flambeau_quant::GgmlDType::Q5_0, &w_bytes, &mut w_dequant)?;
    let act_rt = q8_1_roundtrip(&act_f32);
    let mut reference = vec![0.0f32; n_tokens * top_k * n_rows];
    for t in 0..n_tokens {
        for slot in 0..top_k {
            let expert = expert_ids[t * top_k + slot] as usize;
            for row in 0..n_rows {
                let mut acc = 0.0f64;
                for j in 0..k_dim {
                    let w = w_dequant[((expert * n_rows) + row) * k_dim + j];
                    let a = act_rt[t * k_dim + j];
                    acc += (w * a) as f64;
                }
                reference[(t * top_k + slot) * n_rows + row] = acc as f32;
            }
        }
    }
    Ok(max_rel_err_with_floor(&got, &reference, (k_dim as f32).sqrt() * 0.01))
}

#[allow(clippy::too_many_arguments)]
fn run_q5_1_moe_shape(
    dev: &HipDevice,
    kernel: &HipKernel<'_>,
    q_kernel: &HipKernel<'_>,
    n_experts: usize,
    n_rows: usize,
    k_dim: usize,
    top_k: usize,
    n_tokens: usize,
    seed: u64,
) -> Result<f32> {
    let nb_per_row = k_dim / 32;
    let w_elems = n_experts * n_rows * k_dim;
    let w_f32 = seeded_f32_range(seed, w_elems, -0.5, 0.5);
    let w_bytes = encode_q5_1(&w_f32);

    let act_f32 = seeded_f32_range(seed.wrapping_add(0xA1), n_tokens * k_dim, -0.5, 0.5);
    let expert_ids: Vec<i32> = (0..n_tokens * top_k)
        .map(|i| {
            let h = (i as u64)
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(seed.wrapping_mul(0x12345));
            ((h >> 32) as u32 % n_experts as u32) as i32
        })
        .collect();

    let d_w = upload(dev, &w_bytes);
    let d_act = upload(dev, &act_f32);
    let y_blocks_total = n_tokens * nb_per_row;
    let d_y = dev.alloc(y_blocks_total * std::mem::size_of::<BlockQ8_1>())?;
    let d_ids = upload(dev, &expert_ids);
    let d_dst = dev.alloc(n_tokens * top_k * n_rows * 4)?;

    {
        let stream = dev.default_stream();
        let n_elems = (n_tokens * k_dim) as i32;
        let d_a_p: u64 = d_act.as_usize() as u64;
        let d_y_p: u64 = d_y.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_a_p);
        args.push(&d_y_p);
        args.push(&n_elems);
        let cfg = LaunchCfg::one_d(y_blocks_total as u32, 32);
        unsafe { q_kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    {
        let stream = dev.default_stream();
        let n_rows_i = n_rows as i32;
        let n_tokens_i = n_tokens as i32;
        let top_k_i = top_k as i32;
        let nb_i = nb_per_row as i32;
        let d_w_p: u64 = d_w.as_usize() as u64;
        let d_y_p: u64 = d_y.as_usize() as u64;
        let d_ids_p: u64 = d_ids.as_usize() as u64;
        let d_dst_p: u64 = d_dst.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_w_p);
        args.push(&d_y_p);
        args.push(&d_ids_p);
        args.push(&d_dst_p);
        args.push(&n_rows_i);
        args.push(&n_tokens_i);
        args.push(&top_k_i);
        args.push(&nb_i);
        let cfg = LaunchCfg {
            grid: (n_rows as u32, (n_tokens * top_k) as u32, 1),
            block: (256, 1, 1),
            shared_bytes: 0,
        };
        unsafe { kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    let mut got = vec![0.0f32; n_tokens * top_k * n_rows];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got.as_mut_ptr() as usize),
            d_dst,
            n_tokens * top_k * n_rows * 4,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_w, w_bytes.len())?;
        dev.dealloc(d_act, act_f32.len() * 4)?;
        dev.dealloc(d_y, y_blocks_total * std::mem::size_of::<BlockQ8_1>())?;
        dev.dealloc(d_ids, expert_ids.len() * 4)?;
        dev.dealloc(d_dst, n_tokens * top_k * n_rows * 4)?;
    }

    let mut w_dequant = vec![0.0f32; w_elems];
    flambeau_quant::dequantize_into(flambeau_quant::GgmlDType::Q5_1, &w_bytes, &mut w_dequant)?;
    let act_rt = q8_1_roundtrip(&act_f32);
    let mut reference = vec![0.0f32; n_tokens * top_k * n_rows];
    for t in 0..n_tokens {
        for slot in 0..top_k {
            let expert = expert_ids[t * top_k + slot] as usize;
            for row in 0..n_rows {
                let mut acc = 0.0f64;
                for j in 0..k_dim {
                    let w = w_dequant[((expert * n_rows) + row) * k_dim + j];
                    let a = act_rt[t * k_dim + j];
                    acc += (w * a) as f64;
                }
                reference[(t * top_k + slot) * n_rows + row] = acc as f32;
            }
        }
    }
    Ok(max_rel_err_with_floor(&got, &reference, (k_dim as f32).sqrt() * 0.01))
}

// ---------- encoders + helpers ----------

fn encode_q4_0(xs: &[f32]) -> Vec<u8> {
    assert_eq!(xs.len() % QK8_0, 0);
    let nb = xs.len() / QK8_0;
    let mut out = Vec::with_capacity(nb * std::mem::size_of::<BlockQ4_0>());
    for block in xs.chunks_exact(QK8_0) {
        // Q4_0: per-block signed range [-d*8, d*7]. d = absmax / -8 for ggml
        // style (keeps negative values lossless). Match `candle-core`
        // `BlockQ4_0::from_float` exactly.
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &v in block {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let d_f16 = half::f16::from_f32(d);
        out.extend_from_slice(&d_f16.to_bits().to_le_bytes());
        for i in 0..16 {
            let x0 = block[i] * id;
            let x1 = block[i + 16] * id;
            let lo = (x0 + 8.5) as i32;
            let hi = (x1 + 8.5) as i32;
            let lo = lo.clamp(0, 15) as u8;
            let hi = hi.clamp(0, 15) as u8;
            out.push((hi << 4) | lo);
        }
    }
    out
}

fn encode_q4_1(xs: &[f32]) -> Vec<u8> {
    assert_eq!(xs.len() % QK8_0, 0);
    let nb = xs.len() / QK8_0;
    let mut out = Vec::with_capacity(nb * std::mem::size_of::<BlockQ4_1>());
    for block in xs.chunks_exact(QK8_0) {
        // Q4_1 affine: d = (max - min) / 15, m = min, q = round((x - m) / d).
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        for &v in block {
            if v < mn { mn = v; }
            if v > mx { mx = v; }
        }
        let d = (mx - mn) / 15.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let m = mn;
        let d_f16 = half::f16::from_f32(d);
        let m_f16 = half::f16::from_f32(m);
        out.extend_from_slice(&d_f16.to_bits().to_le_bytes());
        out.extend_from_slice(&m_f16.to_bits().to_le_bytes());
        for i in 0..16 {
            let x0 = (block[i] - m) * id;
            let x1 = (block[i + 16] - m) * id;
            let lo = (x0 + 0.5) as i32;
            let hi = (x1 + 0.5) as i32;
            let lo = lo.clamp(0, 15) as u8;
            let hi = hi.clamp(0, 15) as u8;
            out.push((hi << 4) | lo);
        }
    }
    out
}

fn encode_q5_1(xs: &[f32]) -> Vec<u8> {
    use flambeau_quant::BlockQ5_1;
    assert_eq!(xs.len() % QK8_0, 0);
    let nb = xs.len() / QK8_0;
    let mut out = Vec::with_capacity(nb * std::mem::size_of::<BlockQ5_1>());
    for block in xs.chunks_exact(QK8_0) {
        // Q5_1 affine quant: d = (max - min) / 31, m = min, q5 = round((x - m) / d).
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        for &v in block {
            if v < mn { mn = v; }
            if v > mx { mx = v; }
        }
        let d = (mx - mn) / 31.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let m = mn;
        let d_f16 = half::f16::from_f32(d);
        let m_f16 = half::f16::from_f32(m);
        out.extend_from_slice(&d_f16.to_bits().to_le_bytes());
        out.extend_from_slice(&m_f16.to_bits().to_le_bytes());
        let mut qh: u32 = 0;
        let mut nibbles = [0u8; 16];
        for i in 0..16 {
            let x0 = (block[i] - m) * id;
            let x1 = (block[i + 16] - m) * id;
            let q0 = (x0 + 0.5) as i32;
            let q1 = (x1 + 0.5) as i32;
            let lo = q0.clamp(0, 31) as u8;
            let hi = q1.clamp(0, 31) as u8;
            qh |= ((lo as u32 & 0x10) >> 4) << i;
            qh |= ((hi as u32 & 0x10) >> 4) << (i + 16);
            nibbles[i] = ((hi & 0x0F) << 4) | (lo & 0x0F);
        }
        out.extend_from_slice(&qh.to_le_bytes());
        out.extend_from_slice(&nibbles);
    }
    out
}

fn encode_q5_0(xs: &[f32]) -> Vec<u8> {
    assert_eq!(xs.len() % QK8_0, 0);
    let nb = xs.len() / QK8_0;
    let mut out = Vec::with_capacity(nb * std::mem::size_of::<BlockQ5_0>());
    for block in xs.chunks_exact(QK8_0) {
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &v in block {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        let d = max / -16.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let d_f16 = half::f16::from_f32(d);
        out.extend_from_slice(&d_f16.to_bits().to_le_bytes());
        let mut qh: u32 = 0;
        let mut nibbles = [0u8; 16];
        for i in 0..16 {
            let x0 = block[i] * id;
            let x1 = block[i + 16] * id;
            let lo = ((x0 + 16.5) as i32).clamp(0, 31) as u8;
            let hi = ((x1 + 16.5) as i32).clamp(0, 31) as u8;
            qh |= ((lo as u32 & 0x10) >> 4) << i;
            qh |= ((hi as u32 & 0x10) >> 4) << (i + 16);
            nibbles[i] = ((hi & 0x0F) << 4) | (lo & 0x0F);
        }
        out.extend_from_slice(&qh.to_le_bytes());
        out.extend_from_slice(&nibbles);
    }
    out
}

fn upload<T: Copy>(dev: &HipDevice, data: &[T]) -> DevicePtr {
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

fn q8_1_roundtrip(xs: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; xs.len()];
    for (i, block) in xs.chunks_exact(32).enumerate() {
        let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        for (j, &v) in block.iter().enumerate() {
            let q = (v * id).round().clamp(-127.0, 127.0) as i32;
            out[i * 32 + j] = d * q as f32;
        }
    }
    out
}

