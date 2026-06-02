//! MoE kernel certs — TopK router, IndexedMoE MMVQ Q4_K, MoE combine.
//! Three targeted sweeps, each emitting its own cert. Shapes are chosen to
//! exercise Qwen3.6's MoE regime: 128 experts, top-8 routing, head-shaped
//! hidden dims. No delta-ppl quality cert — that's a model-loader
//! deliverable.

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
use flambeau_quant::{
    BlockQ2K, BlockQ3K, BlockQ4K, BlockQ4_1, BlockQ5K, BlockQ6K, BlockQ8_1, GgmlDType, QK8_0, QK_K,
};
use half::f16;

use crate::cert::{now_utc_iso8601, Cert, PmcSnapshot, ShapeResult, SCHEMA_VERSION};
use crate::harness::{
    alloc_and_upload, max_rel_err_with_floor as harness_err_floor, rig, seeded_f32_range,
};

const QK8: usize = QK8_0;

// ---------------------------------------------------------------------------
// TopK cert
// ---------------------------------------------------------------------------

pub fn run_topk_sweep(repo_root: &Path) -> Result<Cert> {
    let dev = ensure_dev()?;
    let kb = kernels::hsaco("topk_f32").unwrap();
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_topk_softmax_f32")?;
    let attrs: FuncAttributes = kernel.attributes()?;

    // Qwen3.5 is (128 experts, top-8); Qwen3.6 is (256 experts, top-8).
    // Cover both n_experts values at varying batch sizes. (fixed
    // 2026-04-21) uncovered TOPK_MAX_EXPERTS=128 silently dropping experts
    // 128..255 — the n_experts=256 rows below gate against regression.
    let cases = [
        (1usize, 128usize, 8usize),
        (8, 128, 8),
        (128, 128, 8),
        (512, 128, 8),
        (1, 256, 8),
        (8, 256, 8),
        (128, 256, 8),
    ];
    let mut results = Vec::new();
    for (n_tokens, n_experts, k) in cases {
        let logits = seeded_f32(0xDEC0DE ^ (n_tokens as u64 * 101), n_tokens * n_experts);
        let (got_idx, got_wts) = run_topk(&dev, &kernel, &logits, n_tokens, n_experts, k)?;
        let (ref_idx, ref_wts) = cpu_topk(&logits, n_tokens, n_experts, k);
        // Indices must match exactly; weights within 1e-5.
        let mut idx_mismatch = 0usize;
        let mut wts_err: f32 = 0.0;
        for i in 0..n_tokens * k {
            if got_idx[i] != ref_idx[i] {
                idx_mismatch += 1;
            }
            wts_err = wts_err.max((got_wts[i] - ref_wts[i]).abs());
        }
        // Report idx_mismatch as a large rel_err when non-zero so the cert
        // picks it up; otherwise use the weights err.
        let max_rel = if idx_mismatch > 0 { 1.0 } else { wts_err };
        results.push(ShapeResult {
            m: n_tokens,
            k: n_experts,
            n: k,
            seed: 0xDEC0DE ^ (n_tokens as u64 * 101),
            max_rel_err: max_rel,
            tolerance: 1e-4,
            pass: max_rel <= 1e-4,
        });
    }
    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "topk_f32_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "topk_softmax".to_string(),
        dtype_weight: "F32".to_string(),
        dtype_activation: "F32".to_string(),
        tolerance_formula: "indices match exactly; softmax weights within 1e-4 abs".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_tag(),
        pmc: Some(pmc_from(&attrs)),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

fn cpu_topk(logits: &[f32], n_tokens: usize, n_experts: usize, k: usize) -> (Vec<i32>, Vec<f32>) {
    let mut idxs = vec![0i32; n_tokens * k];
    let mut wts = vec![0f32; n_tokens * k];
    for t in 0..n_tokens {
        let row = &logits[t * n_experts..(t + 1) * n_experts];
        let mut pairs: Vec<(f32, i32)> = row
            .iter()
            .enumerate()
            .map(|(i, &v)| (v, i as i32))
            .collect();
        pairs.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
        let top: Vec<f32> = pairs[..k].iter().map(|p| p.0).collect();
        let m = top.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = top.iter().map(|v| (v - m).exp()).sum();
        for i in 0..k {
            idxs[t * k + i] = pairs[i].1;
            wts[t * k + i] = ((pairs[i].0 - m).exp()) / sum;
        }
    }
    (idxs, wts)
}

fn run_topk(
    dev: &HipDevice,
    kernel: &HipKernel<'_>,
    logits: &[f32],
    n_tokens: usize,
    n_experts: usize,
    k: usize,
) -> Result<(Vec<i32>, Vec<f32>)> {
    let d_l = upload(dev, logits);
    let d_i = dev.alloc(n_tokens * k * 4)?;
    let d_w = dev.alloc(n_tokens * k * 4)?;
    let stream = dev.default_stream();
    let n_tokens_i = n_tokens as i32;
    let n_experts_i = n_experts as i32;
    let k_i = k as i32;
    let d_l_p: u64 = d_l.as_usize() as u64;
    let d_i_p: u64 = d_i.as_usize() as u64;
    let d_w_p: u64 = d_w.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&d_l_p);
    args.push(&d_i_p);
    args.push(&d_w_p);
    args.push(&n_tokens_i);
    args.push(&n_experts_i);
    args.push(&k_i);
    let cfg = LaunchCfg::one_d(n_tokens as u32, n_experts as u32);
    unsafe { kernel.launch(stream, cfg, args)? };
    stream.synchronize()?;
    let mut out_i = vec![0i32; n_tokens * k];
    let mut out_w = vec![0f32; n_tokens * k];
    unsafe {
        dev.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            DevicePtr(out_i.as_mut_ptr() as usize),
            d_i,
            n_tokens * k * 4,
        )?;
        dev.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            DevicePtr(out_w.as_mut_ptr() as usize),
            d_w,
            n_tokens * k * 4,
        )?;
    }
    stream.synchronize()?;
    unsafe {
        dev.dealloc(d_l, logits.len() * 4)?;
        dev.dealloc(d_i, n_tokens * k * 4)?;
        dev.dealloc(d_w, n_tokens * k * 4)?;
    }
    Ok((out_i, out_w))
}

// ---------------------------------------------------------------------------
// IndexedMoE MMVQ cert
// ---------------------------------------------------------------------------

pub fn run_indexed_moe_mmvq_sweep(repo_root: &Path) -> Result<Cert> {
    let dev = ensure_dev()?;
    let kb = kernels::hsaco("indexed_moe_mmvq_q4_k").unwrap();
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_indexed_moe_mmvq_q4_k_q8_1")?;
    let attrs: FuncAttributes = kernel.attributes()?;
    let q_kb = kernels::hsaco("quantize_q8_1").unwrap();
    let q_module = HipModule::load(dev.id(), q_kb)?;
    let q_kernel: HipKernel<'_> = q_module.kernel("flambeau_quantize_row_q8_1")?;

    // Qwen3.6-scale MoE (scaled down by 10× to keep test runtime sane):
    // n_experts = 16 (of 128)
    // n_rows = 256 (expert output dim / 10)
    // k = 2048 (hidden)
    // top_k = 4 (of 8)
    // n_tokens = 1, 8
    let cases = [(1usize, 4usize, 256usize, 2048usize), (8, 4, 256, 2048)];
    let n_experts = 16usize;
    let mut results = Vec::new();
    for (n_tokens, top_k, n_rows, k_dim) in cases {
        let seed = 0xDEC0DE ^ (n_tokens as u64 * 31 + top_k as u64 * 7);
        let max_rel = run_moe_mmvq_shape(
            &dev, &kernel, &q_kernel, n_experts, n_rows, k_dim, top_k, n_tokens, seed,
        )?;
        let tol = 5e-2;
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
        impl_id: "indexed_moe_mmvq_q4_k_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "indexed_moe_mmvq".to_string(),
        dtype_weight: "Q4_K".to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula: "|err| <= 5e-2 * max(|ref|, sqrt(k))  (MoE MMVQ)".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_tag(),
        pmc: Some(pmc_from(&attrs)),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

fn run_moe_mmvq_shape(
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
    assert_eq!(k_dim % QK_K, 0);
    let nb_per_row = k_dim / QK_K;

    // Expert weights: random Q4_K with "realistic" d/dmin (same taming as
    // the sweep).
    let total_blocks = n_experts * n_rows * nb_per_row;
    let w_bytes = total_blocks * std::mem::size_of::<BlockQ4K>();
    let w_raw = tame_q4k_scales(seeded_bytes(seed, w_bytes));
    let _w_blocks: &[BlockQ4K] = bytemuck::cast_slice(&w_raw);

    // Activations + router-selected experts.
    let act_f32 = seeded_f32(seed.wrapping_add(0xA1), n_tokens * k_dim);
    let expert_ids: Vec<i32> = (0..n_tokens * top_k)
        .map(|i| {
            let h = (i as u64)
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(seed.wrapping_mul(0x12345));
            ((h >> 32) as u32 % n_experts as u32) as i32
        })
        .collect();

    // Upload.
    let d_w = upload(dev, &w_raw);
    let d_act = upload(dev, &act_f32);
    let y_blocks_total = n_tokens * nb_per_row * 8;
    let d_y = dev.alloc(y_blocks_total * std::mem::size_of::<BlockQ8_1>())?;
    let d_ids = upload(dev, &expert_ids);
    let d_dst = dev.alloc(n_tokens * top_k * n_rows * 4)?;

    // Quantise activation.
    {
        let stream = dev.default_stream();
        let n_elems = (n_tokens * k_dim) as i32;
        let d_a_p: u64 = d_act.as_usize() as u64;
        let d_y_p: u64 = d_y.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_a_p);
        args.push(&d_y_p);
        args.push(&n_elems);
        let cfg = LaunchCfg::one_d(y_blocks_total as u32, QK8 as u32);
        unsafe { q_kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    // Launch MMVQ.
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
            block: (64, 1, 1),
            shared_bytes: 0,
        };
        unsafe { kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    // Download GPU output.
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
        dev.dealloc(d_w, w_bytes)?;
        dev.dealloc(d_act, act_f32.len() * 4)?;
        dev.dealloc(d_y, y_blocks_total * std::mem::size_of::<BlockQ8_1>())?;
        dev.dealloc(d_ids, expert_ids.len() * 4)?;
        dev.dealloc(d_dst, n_tokens * top_k * n_rows * 4)?;
    }

    // Reference: CPU dequant + Q8_1-round-trip activation + matmul per
    // (token, slot, row). Dequantise the whole weight tensor once.
    let total_elems = n_experts * n_rows * k_dim;
    let mut w_dequant = vec![0.0f32; total_elems];
    flambeau_quant::dequantize_into(flambeau_quant::GgmlDType::Q4K, &w_raw, &mut w_dequant)?;
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
    Ok(max_rel_err_with_floor(&got, &reference, k_dim))
}

fn tame_q4k_scales(mut raw: Vec<u8>) -> Vec<u8> {
    let bs = std::mem::size_of::<BlockQ4K>();
    let nblocks = raw.len() / bs;
    for i in 0..nblocks {
        let block = &mut raw[i * bs..(i + 1) * bs];
        let d = f16::from_f32((block[0] as f32 / 255.0) * 0.1 + 0.01);
        let dmin = f16::from_f32((block[1] as f32 / 255.0) * 0.05);
        block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
        block[2..4].copy_from_slice(&dmin.to_bits().to_le_bytes());
    }
    raw
}

fn q8_1_roundtrip(xs: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; xs.len()];
    for i in 0..(xs.len() / QK8) {
        let block = &xs[i * QK8..(i + 1) * QK8];
        let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        for (j, &v) in block.iter().enumerate() {
            let q = (v * id).round().clamp(-127.0, 127.0) as i32;
            out[i * QK8 + j] = (q as f32) * d;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Multi-row r2 IndexedMoE MMVQ cert (candle P29)
// ---------------------------------------------------------------------------

pub fn run_indexed_moe_mmvq_r2_sweep(repo_root: &Path) -> Result<Cert> {
    let dev = ensure_dev()?;
    let kb = kernels::hsaco("indexed_moe_mmvq_q4_k_r2").unwrap();
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_indexed_moe_mmvq_q4_k_r2_q8_1")?;
    let attrs: FuncAttributes = kernel.attributes()?;
    let q_kb = kernels::hsaco("quantize_q8_1").unwrap();
    let q_module = HipModule::load(dev.id(), q_kb)?;
    let q_kernel: HipKernel<'_> = q_module.kernel("flambeau_quantize_row_q8_1")?;

    let n_experts = 16usize;
    let cases = [(1usize, 4usize, 256usize, 2048usize), (8, 4, 256, 2048)];
    let mut results = Vec::new();
    for (n_tokens, top_k, n_rows, k_dim) in cases {
        let seed = 0xDEC0DE ^ (n_tokens as u64 * 83 + top_k as u64 * 29);
        let max_rel = run_r2_shape(
            &dev, &kernel, &q_kernel, n_experts, n_rows, k_dim, top_k, n_tokens, seed,
        )?;
        let tol = 5e-2;
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
        impl_id: "indexed_moe_mmvq_q4_k_r2_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "indexed_moe_mmvq".to_string(),
        dtype_weight: "Q4_K".to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula: "|err| <= 5e-2 * max(|ref|, sqrt(k))  (same envelope as single-row)"
            .to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_tag(),
        pmc: Some(pmc_from(&attrs)),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

fn run_r2_shape(
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
    assert_eq!(k_dim % QK_K, 0);
    assert_eq!(n_rows % 2, 0, "r2 grid expects even n_rows");
    let nb_per_row = k_dim / QK_K;

    let total_blocks = n_experts * n_rows * nb_per_row;
    let w_bytes = total_blocks * std::mem::size_of::<BlockQ4K>();
    let w_raw = tame_q4k_scales(seeded_bytes(seed, w_bytes));

    let act_f32 = seeded_f32(seed.wrapping_add(0xA1), n_tokens * k_dim);
    let expert_ids: Vec<i32> = (0..n_tokens * top_k)
        .map(|i| {
            let h = (i as u64)
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(seed.wrapping_mul(0x12345));
            ((h >> 32) as u32 % n_experts as u32) as i32
        })
        .collect();

    let d_w = upload(dev, &w_raw);
    let d_act = upload(dev, &act_f32);
    let y_blocks_total = n_tokens * nb_per_row * 8;
    let d_y = dev.alloc(y_blocks_total * std::mem::size_of::<BlockQ8_1>())?;
    let d_ids = upload(dev, &expert_ids);
    let d_dst = dev.alloc(n_tokens * top_k * n_rows * 4)?;

    // Quantise activation.
    {
        let stream = dev.default_stream();
        let n_elems = (n_tokens * k_dim) as i32;
        let d_a_p: u64 = d_act.as_usize() as u64;
        let d_y_p: u64 = d_y.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_a_p);
        args.push(&d_y_p);
        args.push(&n_elems);
        let cfg = LaunchCfg::one_d(y_blocks_total as u32, QK8 as u32);
        unsafe { q_kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    // Launch r2 kernel — grid.x is ceil(n_rows / 2).
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
        let grid_x = (n_rows as u32).div_ceil(2);
        let cfg = LaunchCfg {
            grid: (grid_x, (n_tokens * top_k) as u32, 1),
            block: (64, 1, 1),
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
        dev.dealloc(d_w, w_bytes)?;
        dev.dealloc(d_act, act_f32.len() * 4)?;
        dev.dealloc(d_y, y_blocks_total * std::mem::size_of::<BlockQ8_1>())?;
        dev.dealloc(d_ids, expert_ids.len() * 4)?;
        dev.dealloc(d_dst, n_tokens * top_k * n_rows * 4)?;
    }

    // Reference: same CPU path as single-row cert.
    let total_elems = n_experts * n_rows * k_dim;
    let mut w_dequant = vec![0.0f32; total_elems];
    flambeau_quant::dequantize_into(flambeau_quant::GgmlDType::Q4K, &w_raw, &mut w_dequant)?;
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
    Ok(max_rel_err_with_floor(&got, &reference, k_dim))
}

// ---------------------------------------------------------------------------
// Q6_K IndexedMoE MMVQ cert (UD-Q4_K_S mixed-quant down_exps)
// ---------------------------------------------------------------------------

/// 8.b-i4 — Q5_K indexed-MoE MMVQ correctness sweep. Shapes cover
/// the Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL ffn_down_exps footprint:
/// hidden=2048, inter=768, n_experts=128, top_k=8. Scaled down to 16
/// experts for cert runtime.
pub fn run_indexed_moe_mmvq_q5_k_sweep(repo_root: &Path) -> Result<Cert> {
    let dev = ensure_dev()?;
    let kb = kernels::hsaco("indexed_moe_mmvq_q5_k_r2_dp4a").unwrap();
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_indexed_moe_mmvq_q5_k_r2_dp4a_q8_1")?;
    let attrs: FuncAttributes = kernel.attributes()?;
    let q_kb = kernels::hsaco("quantize_q8_1").unwrap();
    let q_module = HipModule::load(dev.id(), q_kb)?;
    let q_kernel: HipKernel<'_> = q_module.kernel("flambeau_quantize_row_q8_1")?;

    let n_experts = 16usize;
    // Down shape: n_rows = hidden (2048 on Qwen3-Coder), k = inter (768).
    let cases = [
        (1usize, 4usize, 2048usize, 768usize),
        (4, 4, 2048, 768),
        (1, 8, 2048, 768), // top_k=8 matches Qwen3-Coder
    ];
    let mut results = Vec::new();
    for (n_tokens, top_k, n_rows, k_dim) in cases {
        let seed = 0xDEC0DE ^ (n_tokens as u64 * 53 + top_k as u64 * 17) ^ 0xC5C5_u64; // Q5-specific spice
        let max_rel = run_q5k_shape(
            &dev, &kernel, &q_kernel, n_experts, n_rows, k_dim, top_k, n_tokens, seed,
        )?;
        let tol = 5e-2;
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
        impl_id: "indexed_moe_mmvq_q5_k_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "indexed_moe_mmvq".to_string(),
        dtype_weight: "Q5_K".to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula:
            "|err| <= 5e-2 * max(|ref|, sqrt(k))  (Q5_K weights; same envelope as Q4_K MoE MMVQ)"
                .to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_tag(),
        pmc: Some(pmc_from(&attrs)),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

fn run_q5k_shape(
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
    assert_eq!(k_dim % QK_K, 0);
    let nb_per_row = k_dim / QK_K;

    let total_blocks = n_experts * n_rows * nb_per_row;
    let w_bytes = total_blocks * std::mem::size_of::<BlockQ5K>();
    let w_raw = tame_q5k_scales(seeded_bytes(seed, w_bytes));

    let act_f32 = seeded_f32(seed.wrapping_add(0xA1), n_tokens * k_dim);
    let expert_ids: Vec<i32> = (0..n_tokens * top_k)
        .map(|i| {
            let h = (i as u64)
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(seed.wrapping_mul(0x12345));
            ((h >> 32) as u32 % n_experts as u32) as i32
        })
        .collect();

    let d_w = upload(dev, &w_raw);
    let d_act = upload(dev, &act_f32);
    let y_blocks_total = n_tokens * nb_per_row * 8;
    let d_y = dev.alloc(y_blocks_total * std::mem::size_of::<BlockQ8_1>())?;
    let d_ids = upload(dev, &expert_ids);
    let d_dst = dev.alloc(n_tokens * top_k * n_rows * 4)?;

    // Quantise activation.
    {
        let stream = dev.default_stream();
        let n_elems = (n_tokens * k_dim) as i32;
        let d_a_p: u64 = d_act.as_usize() as u64;
        let d_y_p: u64 = d_y.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_a_p);
        args.push(&d_y_p);
        args.push(&n_elems);
        let cfg = LaunchCfg::one_d(y_blocks_total as u32, QK8 as u32);
        unsafe { q_kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    // Launch Q5_K MoE MMVQ (r2 dp4a: 2 rows / block, half-warp DPP reduce).
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
        let grid_x = (n_rows as u32).div_ceil(2);
        let cfg = LaunchCfg {
            grid: (grid_x, (n_tokens * top_k) as u32, 1),
            block: (64, 1, 1),
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
        dev.dealloc(d_w, w_bytes)?;
        dev.dealloc(d_act, act_f32.len() * 4)?;
        dev.dealloc(d_y, y_blocks_total * std::mem::size_of::<BlockQ8_1>())?;
        dev.dealloc(d_ids, expert_ids.len() * 4)?;
        dev.dealloc(d_dst, n_tokens * top_k * n_rows * 4)?;
    }

    // Reference: CPU dequant + Q8_1 roundtrip activation + per-(t, slot, row) matmul.
    let total_elems = n_experts * n_rows * k_dim;
    let mut w_dequant = vec![0.0f32; total_elems];
    flambeau_quant::dequantize_into(flambeau_quant::GgmlDType::Q5K, &w_raw, &mut w_dequant)?;
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
    Ok(max_rel_err_with_floor(&got, &reference, k_dim))
}

fn tame_q5k_scales(mut raw: Vec<u8>) -> Vec<u8> {
    // Same treatment as Q4_K: bound d + dmin. Q5_K layout matches Q4_K's
    // d/dmin header exactly (offset 0..4), so the same byte-level fix
    // applies. The 5th-bit qh[32] stays as random bytes.
    let bs = std::mem::size_of::<BlockQ5K>();
    let nblocks = raw.len() / bs;
    for i in 0..nblocks {
        let block = &mut raw[i * bs..(i + 1) * bs];
        let d = f16::from_f32((block[0] as f32 / 255.0) * 0.1 + 0.01);
        let dmin = f16::from_f32((block[1] as f32 / 255.0) * 0.05);
        block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
        block[2..4].copy_from_slice(&dmin.to_bits().to_le_bytes());
    }
    raw
}

pub fn run_indexed_moe_mmvq_q6_k_sweep(repo_root: &Path) -> Result<Cert> {
    let dev = ensure_dev()?;
    let kb = kernels::hsaco("indexed_moe_mmvq_q6_k").unwrap();
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_indexed_moe_mmvq_q6_k_q8_1")?;
    let attrs: FuncAttributes = kernel.attributes()?;
    let q_kb = kernels::hsaco("quantize_q8_1").unwrap();
    let q_module = HipModule::load(dev.id(), q_kb)?;
    let q_kernel: HipKernel<'_> = q_module.kernel("flambeau_quantize_row_q8_1")?;

    let n_experts = 16usize;
    // Qwen3.6 ffn_down_exps shapes: (n_experts, hidden, inter) = (256, 2048, 512).
    // Sweep a scaled version: (n_experts=16, n_rows=256, k=512).
    let cases = [(1usize, 4usize, 256usize, 512usize), (4, 4, 256, 512)];
    let mut results = Vec::new();
    for (n_tokens, top_k, n_rows, k_dim) in cases {
        let seed = 0xDEC0DE ^ (n_tokens as u64 * 53 + top_k as u64 * 17) ^ 0xABCD_u64; // Q6_K-specific spice so seeds differ from Q4_K sweeps
        let max_rel = run_q6k_shape(
            &dev, &kernel, &q_kernel, n_experts, n_rows, k_dim, top_k, n_tokens, seed,
        )?;
        let tol = 5e-2;
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
        impl_id: "indexed_moe_mmvq_q6_k_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "indexed_moe_mmvq".to_string(),
        dtype_weight: "Q6_K".to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula:
            "|err| <= 5e-2 * max(|ref|, sqrt(k))  (Q6_K weights; same envelope as Q4_K MoE MMVQ)"
                .to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_tag(),
        pmc: Some(pmc_from(&attrs)),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

/// 2.a — Q8_0 indexed-MoE MMVQ correctness sweep. Shapes cover the
/// Qwen3.6-35B-A3B MoE footprint in UD-Q8_K_XL: hidden=2048, inter=768,
/// n_experts=256, top_k=8. Scaled down to 16 experts for cert runtime.
pub fn run_indexed_moe_mmvq_q8_0_sweep(repo_root: &Path) -> Result<Cert> {
    let dev = ensure_dev()?;
    let kb = kernels::hsaco("indexed_moe_mmvq_q8_0").unwrap();
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_indexed_moe_mmvq_q8_0_dp4a_q8_1")?;
    let attrs: FuncAttributes = kernel.attributes()?;
    let q_kb = kernels::hsaco("quantize_q8_1").unwrap();
    let q_module = HipModule::load(dev.id(), q_kb)?;
    let q_kernel: HipKernel<'_> = q_module.kernel("flambeau_quantize_row_q8_1")?;

    let n_experts = 16usize;
    let cases = [
        (1usize, 4usize, 256usize, 2048usize), // Qwen3.6 gate/up: hidden=2048, inter≈ n_rows here
        (4, 4, 256, 2048),                     // prefill L=4
        (1, 4, 2048, 768),                     // down-like: n_rows=hidden, k=inter
    ];
    let mut results = Vec::new();
    for (n_tokens, top_k, n_rows, k_dim) in cases {
        let seed =
            0xDEC0DE ^ (n_tokens as u64 * 53 + top_k as u64 * 17 + n_rows as u64 * 7) ^ 0xB8D0u64;
        let max_rel = run_q8_0_shape(
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
        impl_id: "indexed_moe_mmvq_q8_0_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "indexed_moe_mmvq".to_string(),
        dtype_weight: "Q8_0".to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula:
            "|err| <= 3e-2 * max(|ref|, sqrt(k))  (Q8_0 indexed-MoE; DP4A VDR=2 same envelope as dense Q8_0 MMVQ)".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_tag(),
        pmc: Some(pmc_from(&attrs)),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

fn run_q8_0_shape(
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
    assert_eq!(k_dim % QK8, 0);
    let nb_per_row = k_dim / QK8;

    // Quantise weights on host to Q8_0 from seeded F32 (same distribution as
    // the activation sweeps, so kernel-side rounding bias is clean).
    let w_elems = n_experts * n_rows * k_dim;
    let w_f32 = seeded_f32(seed, w_elems);
    let w_q8_0: Vec<u8> = {
        let mut out = Vec::with_capacity(n_experts * n_rows * nb_per_row * 34);
        for block in w_f32.chunks_exact(QK8) {
            let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            let d_f16 = half::f16::from_f32(d);
            out.extend_from_slice(&d_f16.to_bits().to_le_bytes());
            for &v in block {
                let q = (v * id).round().clamp(-127.0, 127.0) as i8;
                out.push(q as u8);
            }
        }
        out
    };

    let act_f32 = seeded_f32(seed.wrapping_add(0xA1), n_tokens * k_dim);
    let expert_ids: Vec<i32> = (0..n_tokens * top_k)
        .map(|i| {
            let h = (i as u64)
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(seed.wrapping_mul(0x12345));
            ((h >> 32) as u32 % n_experts as u32) as i32
        })
        .collect();

    let d_w = upload(dev, &w_q8_0);
    let d_act = upload(dev, &act_f32);
    let y_blocks_total = n_tokens * nb_per_row;
    let d_y = dev.alloc(y_blocks_total * std::mem::size_of::<BlockQ8_1>())?;
    let d_ids = upload(dev, &expert_ids);
    let d_dst = dev.alloc(n_tokens * top_k * n_rows * 4)?;

    // Quantise activation F32 → Q8_1.
    {
        let stream = dev.default_stream();
        let n_elems = (n_tokens * k_dim) as i32;
        let d_a_p: u64 = d_act.as_usize() as u64;
        let d_y_p: u64 = d_y.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_a_p);
        args.push(&d_y_p);
        args.push(&n_elems);
        let cfg = LaunchCfg::one_d(y_blocks_total as u32, QK8 as u32);
        unsafe { q_kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    // Q8_0 MoE MMVQ: block=256 threads, grid={n_rows, n_tokens*top_k, 1}.
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
        dev.dealloc(d_w, w_q8_0.len())?;
        dev.dealloc(d_act, act_f32.len() * 4)?;
        dev.dealloc(d_y, y_blocks_total * std::mem::size_of::<BlockQ8_1>())?;
        dev.dealloc(d_ids, expert_ids.len() * 4)?;
        dev.dealloc(d_dst, n_tokens * top_k * n_rows * 4)?;
    }

    // Reference: CPU F32 dequant of weights + Q8_1-roundtripped activation.
    let mut w_dequant = vec![0.0f32; w_elems];
    flambeau_quant::dequantize_into(flambeau_quant::GgmlDType::Q8_0, &w_q8_0, &mut w_dequant)?;
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
    Ok(max_rel_err_with_floor(&got, &reference, k_dim))
}

fn run_q6k_shape(
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
    assert_eq!(k_dim % QK_K, 0);
    let nb_per_row = k_dim / QK_K;

    let total_blocks = n_experts * n_rows * nb_per_row;
    let w_bytes = total_blocks * std::mem::size_of::<BlockQ6K>();
    let w_raw = tame_q6k_scales(seeded_bytes(seed, w_bytes));

    let act_f32 = seeded_f32(seed.wrapping_add(0xA1), n_tokens * k_dim);
    let expert_ids: Vec<i32> = (0..n_tokens * top_k)
        .map(|i| {
            let h = (i as u64)
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(seed.wrapping_mul(0x12345));
            ((h >> 32) as u32 % n_experts as u32) as i32
        })
        .collect();

    let d_w = upload(dev, &w_raw);
    let d_act = upload(dev, &act_f32);
    let y_blocks_total = n_tokens * nb_per_row * 8;
    let d_y = dev.alloc(y_blocks_total * std::mem::size_of::<BlockQ8_1>())?;
    let d_ids = upload(dev, &expert_ids);
    let d_dst = dev.alloc(n_tokens * top_k * n_rows * 4)?;

    // Quantise activation.
    {
        let stream = dev.default_stream();
        let n_elems = (n_tokens * k_dim) as i32;
        let d_a_p: u64 = d_act.as_usize() as u64;
        let d_y_p: u64 = d_y.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_a_p);
        args.push(&d_y_p);
        args.push(&n_elems);
        let cfg = LaunchCfg::one_d(y_blocks_total as u32, QK8 as u32);
        unsafe { q_kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    // Launch Q6_K MoE MMVQ — single-row grid: {n_rows, n_tokens*top_k, 1}.
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
            block: (64, 1, 1),
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
        dev.dealloc(d_w, w_bytes)?;
        dev.dealloc(d_act, act_f32.len() * 4)?;
        dev.dealloc(d_y, y_blocks_total * std::mem::size_of::<BlockQ8_1>())?;
        dev.dealloc(d_ids, expert_ids.len() * 4)?;
        dev.dealloc(d_dst, n_tokens * top_k * n_rows * 4)?;
    }

    // Reference: CPU dequant + Q8_1 roundtrip activation + matmul per
    // (token, slot, row). Dequantise once up front.
    let total_elems = n_experts * n_rows * k_dim;
    let mut w_dequant = vec![0.0f32; total_elems];
    flambeau_quant::dequantize_into(flambeau_quant::GgmlDType::Q6K, &w_raw, &mut w_dequant)?;
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
    Ok(max_rel_err_with_floor(&got, &reference, k_dim))
}

fn tame_q6k_scales(mut raw: Vec<u8>) -> Vec<u8> {
    // Match the single-row Q6_K sweep in `sweep_mmvq.rs`: bound i8 scales
    // into [-32, 32] and keep super-block `d` small enough that F32
    // accumulation stays inside the 5e-2 envelope.
    let bs = std::mem::size_of::<BlockQ6K>();
    let nblocks = raw.len() / bs;
    let scales_off = QK_K / 2 + QK_K / 4; // 128+64 = 192
    let d_off = scales_off + QK_K / 16;
    for i in 0..nblocks {
        let block = &mut raw[i * bs..(i + 1) * bs];
        for s in &mut block[scales_off..scales_off + QK_K / 16] {
            let sv = (*s as i32 % 65) - 32;
            *s = sv as u8;
        }
        let d = f16::from_f32((block[d_off] as f32 / 255.0) * 0.05 + 0.01);
        block[d_off..d_off + 2].copy_from_slice(&d.to_bits().to_le_bytes());
    }
    raw
}

// ---------------------------------------------------------------------------
// Fused gate+up MoE MMVQ cert (candle P30)
// ---------------------------------------------------------------------------

pub fn run_gate_up_sweep(repo_root: &Path) -> Result<Cert> {
    let dev = ensure_dev()?;
    let kb = kernels::hsaco("indexed_moe_mmvq_q4_k_gate_up").unwrap();
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_indexed_moe_mmvq_q4_k_gate_up_q8_1")?;
    let attrs: FuncAttributes = kernel.attributes()?;
    let q_kb = kernels::hsaco("quantize_q8_1").unwrap();
    let q_module = HipModule::load(dev.id(), q_kb)?;
    let q_kernel: HipKernel<'_> = q_module.kernel("flambeau_quantize_row_q8_1")?;

    let n_experts = 16usize;
    let cases = [(1usize, 4usize, 256usize, 2048usize), (8, 4, 256, 2048)];
    let mut results = Vec::new();
    for (n_tokens, top_k, n_rows, k_dim) in cases {
        let seed = 0xDEC0DE ^ (n_tokens as u64 * 47 + top_k as u64 * 101);
        let max_rel = run_gate_up_shape(
            &dev, &kernel, &q_kernel, n_experts, n_rows, k_dim, top_k, n_tokens, seed,
        )?;
        let tol = 5e-2;
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
        impl_id: "indexed_moe_mmvq_q4_k_gate_up_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "indexed_moe_mmvq_gate_up".to_string(),
        dtype_weight: "Q4_K".to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula:
            "|err| <= 5e-2 * max(|ref|, sqrt(k))  (same envelope as non-fused MoE MMVQ)".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_tag(),
        pmc: Some(pmc_from(&attrs)),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

fn run_gate_up_shape(
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
    assert_eq!(k_dim % QK_K, 0);
    let nb_per_row = k_dim / QK_K;

    // Two independent Q4_K weight tensors for gate and up.
    let total_blocks = n_experts * n_rows * nb_per_row;
    let w_bytes = total_blocks * std::mem::size_of::<BlockQ4K>();
    let gate_raw = tame_q4k_scales(seeded_bytes(seed, w_bytes));
    let up_raw = tame_q4k_scales(seeded_bytes(seed.wrapping_add(0x1337), w_bytes));

    let act_f32 = seeded_f32(seed.wrapping_add(0xA1), n_tokens * k_dim);
    let expert_ids: Vec<i32> = (0..n_tokens * top_k)
        .map(|i| {
            let h = (i as u64)
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(seed.wrapping_mul(0x12345));
            ((h >> 32) as u32 % n_experts as u32) as i32
        })
        .collect();

    let d_g = upload(dev, &gate_raw);
    let d_u = upload(dev, &up_raw);
    let d_act = upload(dev, &act_f32);
    let y_blocks_total = n_tokens * nb_per_row * 8;
    let d_y = dev.alloc(y_blocks_total * std::mem::size_of::<BlockQ8_1>())?;
    let d_ids = upload(dev, &expert_ids);
    let d_gate_out = dev.alloc(n_tokens * top_k * n_rows * 4)?;
    let d_up_out = dev.alloc(n_tokens * top_k * n_rows * 4)?;

    // Quantise activation.
    {
        let stream = dev.default_stream();
        let n_elems = (n_tokens * k_dim) as i32;
        let d_a_p: u64 = d_act.as_usize() as u64;
        let d_y_p: u64 = d_y.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_a_p);
        args.push(&d_y_p);
        args.push(&n_elems);
        let cfg = LaunchCfg::one_d(y_blocks_total as u32, QK8 as u32);
        unsafe { q_kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    // Launch fused gate+up.
    {
        let stream = dev.default_stream();
        let n_rows_i = n_rows as i32;
        let n_tokens_i = n_tokens as i32;
        let top_k_i = top_k as i32;
        let nb_i = nb_per_row as i32;
        let d_g_p: u64 = d_g.as_usize() as u64;
        let d_u_p: u64 = d_u.as_usize() as u64;
        let d_y_p: u64 = d_y.as_usize() as u64;
        let d_ids_p: u64 = d_ids.as_usize() as u64;
        let d_gout_p: u64 = d_gate_out.as_usize() as u64;
        let d_uout_p: u64 = d_up_out.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_g_p);
        args.push(&d_u_p);
        args.push(&d_y_p);
        args.push(&d_ids_p);
        args.push(&d_gout_p);
        args.push(&d_uout_p);
        args.push(&n_rows_i);
        args.push(&n_tokens_i);
        args.push(&top_k_i);
        args.push(&nb_i);
        let cfg = LaunchCfg {
            grid: (n_rows as u32, (n_tokens * top_k) as u32, 1),
            block: (64, 1, 1),
            shared_bytes: 0,
        };
        unsafe { kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    let total_out = n_tokens * top_k * n_rows;
    let mut got_gate = vec![0.0f32; total_out];
    let mut got_up = vec![0.0f32; total_out];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got_gate.as_mut_ptr() as usize),
            d_gate_out,
            total_out * 4,
        )?;
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got_up.as_mut_ptr() as usize),
            d_up_out,
            total_out * 4,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_g, w_bytes)?;
        dev.dealloc(d_u, w_bytes)?;
        dev.dealloc(d_act, act_f32.len() * 4)?;
        dev.dealloc(d_y, y_blocks_total * std::mem::size_of::<BlockQ8_1>())?;
        dev.dealloc(d_ids, expert_ids.len() * 4)?;
        dev.dealloc(d_gate_out, total_out * 4)?;
        dev.dealloc(d_up_out, total_out * 4)?;
    }

    // Reference: same CPU path, computed for both gate and up weights.
    let total_elems = n_experts * n_rows * k_dim;
    let mut gate_dequant = vec![0.0f32; total_elems];
    let mut up_dequant = vec![0.0f32; total_elems];
    flambeau_quant::dequantize_into(flambeau_quant::GgmlDType::Q4K, &gate_raw, &mut gate_dequant)?;
    flambeau_quant::dequantize_into(flambeau_quant::GgmlDType::Q4K, &up_raw, &mut up_dequant)?;
    let act_rt = q8_1_roundtrip(&act_f32);

    let mut ref_gate = vec![0.0f32; total_out];
    let mut ref_up = vec![0.0f32; total_out];
    for t in 0..n_tokens {
        for slot in 0..top_k {
            let expert = expert_ids[t * top_k + slot] as usize;
            for row in 0..n_rows {
                let mut acc_g = 0.0f64;
                let mut acc_u = 0.0f64;
                for j in 0..k_dim {
                    let wg = gate_dequant[((expert * n_rows) + row) * k_dim + j];
                    let wu = up_dequant[((expert * n_rows) + row) * k_dim + j];
                    let a = act_rt[t * k_dim + j];
                    acc_g += (wg * a) as f64;
                    acc_u += (wu * a) as f64;
                }
                let idx = (t * top_k + slot) * n_rows + row;
                ref_gate[idx] = acc_g as f32;
                ref_up[idx] = acc_u as f32;
            }
        }
    }
    let g_err = max_rel_err_with_floor(&got_gate, &ref_gate, k_dim);
    let u_err = max_rel_err_with_floor(&got_up, &ref_up, k_dim);
    Ok(g_err.max(u_err))
}

// ---------------------------------------------------------------------------
// IndexedMoE MMQ Q4_K cert (4-warp LDS-tiled prefill)
// ---------------------------------------------------------------------------
// MMQ lives in the prefill regime: many (token, slot) pairs processed together.
// The caller (CPU) sorts (token, slot) pairs into per-expert buckets of
// MMQ_X=8 slots so that each block shares ONE expert and can amortise the
// weight tile across all slots. Sentinel -1 marks an unfilled tail slot.

const MMQ_Y: usize = 16;
const MMQ_X: usize = 8;

/// Group (token, slot) pairs into per-expert buckets of size MMQ_X. Returns
/// `(bucket_expert, bucket_slots)` where `bucket_slots[i]` is a flat row of
/// MMQ_X refs (`token << 16 | slot` or `-1` for padding).
fn build_expert_buckets(expert_ids: &[i32], n_tokens: usize, top_k: usize) -> (Vec<i32>, Vec<i32>) {
    use std::collections::HashMap;
    let mut per_expert: HashMap<i32, Vec<i32>> = HashMap::new();
    for t in 0..n_tokens {
        for slot in 0..top_k {
            let e = expert_ids[t * top_k + slot];
            let packed = ((t as i32) << 16) | (slot as i32);
            per_expert.entry(e).or_default().push(packed);
        }
    }
    // Deterministic order: sort by expert id.
    let mut experts: Vec<i32> = per_expert.keys().copied().collect();
    experts.sort();
    let mut bucket_expert = Vec::new();
    let mut bucket_slots = Vec::new();
    for e in experts {
        let refs = per_expert.remove(&e).unwrap();
        for chunk in refs.chunks(MMQ_X) {
            bucket_expert.push(e);
            for &r in chunk {
                bucket_slots.push(r);
            }
            for _ in chunk.len()..MMQ_X {
                bucket_slots.push(-1);
            }
        }
    }
    (bucket_expert, bucket_slots)
}

pub fn run_indexed_moe_mmq_sweep(repo_root: &Path) -> Result<Cert> {
    let dev = ensure_dev()?;
    let kb = kernels::hsaco("indexed_moe_mmq_q4_k").unwrap();
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_indexed_moe_mmq_q4_k_q8_1")?;
    let attrs: FuncAttributes = kernel.attributes()?;
    let q_kb = kernels::hsaco("quantize_q8_1").unwrap();
    let q_module = HipModule::load(dev.id(), q_kb)?;
    let q_kernel: HipKernel<'_> = q_module.kernel("flambeau_quantize_row_q8_1")?;

    // Prefill regime: 128 tokens × 8 slots = 1024 work items. Matches
    // MMQ cert shape band. n_rows=256, k=2048 stays comparable to the MMVQ
    // certs; we also run a tall-K case for amortisation sanity.
    let cases = [(128usize, 8usize, 256usize, 2048usize), (512, 8, 256, 2048)];
    let n_experts = 16usize;
    let mut results = Vec::new();
    for (n_tokens, top_k, n_rows, k_dim) in cases {
        let seed = 0xDEC0DE ^ (n_tokens as u64 * 131 + top_k as u64 * 17);
        let max_rel = run_mmq_shape(
            &dev, &kernel, &q_kernel, n_experts, n_rows, k_dim, top_k, n_tokens, seed,
        )?;
        let tol = 5e-2;
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
        impl_id: "indexed_moe_mmq_q4_k_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "indexed_moe_mmq".to_string(),
        dtype_weight: "Q4_K".to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula: "|err| <= 5e-2 * max(|ref|, sqrt(k))  (MoE MMQ)".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_tag(),
        pmc: Some(pmc_from(&attrs)),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

fn run_mmq_shape(
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
    assert_eq!(k_dim % QK_K, 0);
    let nb_per_row = k_dim / QK_K;

    let total_blocks = n_experts * n_rows * nb_per_row;
    let w_bytes = total_blocks * std::mem::size_of::<BlockQ4K>();
    let w_raw = tame_q4k_scales(seeded_bytes(seed, w_bytes));

    let act_f32 = seeded_f32(seed.wrapping_add(0xA1), n_tokens * k_dim);
    let expert_ids: Vec<i32> = (0..n_tokens * top_k)
        .map(|i| {
            let h = (i as u64)
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(seed.wrapping_mul(0x12345));
            ((h >> 32) as u32 % n_experts as u32) as i32
        })
        .collect();

    let (bucket_expert, bucket_slots) = build_expert_buckets(&expert_ids, n_tokens, top_k);
    let n_buckets = bucket_expert.len();
    assert_eq!(bucket_slots.len(), n_buckets * MMQ_X);

    let d_w = upload(dev, &w_raw);
    let d_act = upload(dev, &act_f32);
    let y_blocks_total = n_tokens * nb_per_row * 8;
    let d_y = dev.alloc(y_blocks_total * std::mem::size_of::<BlockQ8_1>())?;
    let d_be = upload(dev, &bucket_expert);
    let d_bs = upload(dev, &bucket_slots);
    let d_dst = dev.alloc(n_tokens * top_k * n_rows * 4)?;

    // Quantise activation.
    {
        let stream = dev.default_stream();
        let n_elems = (n_tokens * k_dim) as i32;
        let d_a_p: u64 = d_act.as_usize() as u64;
        let d_y_p: u64 = d_y.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_a_p);
        args.push(&d_y_p);
        args.push(&n_elems);
        let cfg = LaunchCfg::one_d(y_blocks_total as u32, QK8 as u32);
        unsafe { q_kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    // Launch MMQ.
    {
        let stream = dev.default_stream();
        let n_rows_i = n_rows as i32;
        let nb_i = nb_per_row as i32;
        let top_k_i = top_k as i32;
        let d_w_p: u64 = d_w.as_usize() as u64;
        let d_y_p: u64 = d_y.as_usize() as u64;
        let d_be_p: u64 = d_be.as_usize() as u64;
        let d_bs_p: u64 = d_bs.as_usize() as u64;
        let d_dst_p: u64 = d_dst.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_w_p);
        args.push(&d_y_p);
        args.push(&d_be_p);
        args.push(&d_bs_p);
        args.push(&d_dst_p);
        args.push(&n_rows_i);
        args.push(&nb_i);
        args.push(&top_k_i);
        let grid_x = (n_rows as u32).div_ceil(MMQ_Y as u32);
        let cfg = LaunchCfg {
            grid: (grid_x, n_buckets as u32, 1),
            block: (128, 1, 1),
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
        dev.dealloc(d_w, w_bytes)?;
        dev.dealloc(d_act, act_f32.len() * 4)?;
        dev.dealloc(d_y, y_blocks_total * std::mem::size_of::<BlockQ8_1>())?;
        dev.dealloc(d_be, bucket_expert.len() * 4)?;
        dev.dealloc(d_bs, bucket_slots.len() * 4)?;
        dev.dealloc(d_dst, n_tokens * top_k * n_rows * 4)?;
    }

    // Reference: CPU dequant + Q8_1 roundtrip activation + matmul per
    // (token, slot, row). Note: unfilled output slots (where the bucket
    // padding wrote nothing) stay at the uninitialised alloc value, so we
    // only compare slots referenced by `expert_ids`.
    let total_elems = n_experts * n_rows * k_dim;
    let mut w_dequant = vec![0.0f32; total_elems];
    flambeau_quant::dequantize_into(flambeau_quant::GgmlDType::Q4K, &w_raw, &mut w_dequant)?;
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
    Ok(max_rel_err_with_floor(&got, &reference, k_dim))
}

// ---------------------------------------------------------------------------
// MoE combine cert
// ---------------------------------------------------------------------------

pub fn run_moe_combine_sweep(repo_root: &Path) -> Result<Cert> {
    let dev = ensure_dev()?;
    let kb = kernels::hsaco("moe_combine_f16").unwrap();
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_moe_combine_f16")?;
    let attrs: FuncAttributes = kernel.attributes()?;

    // Shapes: (n_tokens, top_k, hidden).
    let cases = [(1usize, 8usize, 2048usize), (8, 8, 2048), (128, 8, 5120)];
    let mut results = Vec::new();
    for (n_tokens, top_k, hidden) in cases {
        let seed = 0xDEC0DE ^ (hidden as u64 * 31);
        let max_rel = run_combine_shape(&dev, &kernel, n_tokens, top_k, hidden, seed)?;
        let tol = 1e-2;
        results.push(ShapeResult {
            m: n_tokens,
            k: top_k,
            n: hidden,
            seed,
            max_rel_err: max_rel,
            tolerance: tol,
            pass: max_rel <= tol,
        });
    }
    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "moe_combine_f16_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "moe_combine".to_string(),
        dtype_weight: "F16".to_string(),
        dtype_activation: "F16".to_string(),
        tolerance_formula: "|err| <= 1e-2 * max(|ref|, 1)".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_tag(),
        pmc: Some(pmc_from(&attrs)),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

fn run_combine_shape(
    dev: &HipDevice,
    kernel: &HipKernel<'_>,
    n_tokens: usize,
    top_k: usize,
    hidden: usize,
    seed: u64,
) -> Result<f32> {
    let expert_outs_f32 = seeded_f32(seed, n_tokens * top_k * hidden);
    let residual_f32 = seeded_f32(seed.wrapping_add(1), n_tokens * hidden);
    let weights_f32 = {
        // Softmax over random values per (token, top_k) to simulate router output.
        let raw = seeded_f32(seed.wrapping_add(2), n_tokens * top_k);
        let mut w = vec![0.0f32; raw.len()];
        for t in 0..n_tokens {
            let row = &raw[t * top_k..(t + 1) * top_k];
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|v| (v - m).exp()).sum();
            for k in 0..top_k {
                w[t * top_k + k] = (row[k] - m).exp() / sum;
            }
        }
        w
    };

    let expert_outs_f16: Vec<f16> = expert_outs_f32.iter().map(|v| f16::from_f32(*v)).collect();
    let residual_f16: Vec<f16> = residual_f32.iter().map(|v| f16::from_f32(*v)).collect();

    let d_e = upload(dev, &expert_outs_f16);
    let d_w = upload(dev, &weights_f32);
    let d_r = upload(dev, &residual_f16);
    let d_o = dev.alloc(n_tokens * hidden * 2)?;

    {
        let stream = dev.default_stream();
        let n_tokens_i = n_tokens as i32;
        let top_k_i = top_k as i32;
        let hidden_i = hidden as i32;
        let d_e_p: u64 = d_e.as_usize() as u64;
        let d_w_p: u64 = d_w.as_usize() as u64;
        let d_r_p: u64 = d_r.as_usize() as u64;
        let d_o_p: u64 = d_o.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_e_p);
        args.push(&d_w_p);
        args.push(&d_r_p);
        args.push(&d_o_p);
        args.push(&n_tokens_i);
        args.push(&top_k_i);
        args.push(&hidden_i);
        let total = n_tokens * hidden;
        let cfg = LaunchCfg::one_d(total.div_ceil(256) as u32, 256);
        unsafe { kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    let mut out_f16: Vec<f16> = vec![f16::from_f32(0.0); n_tokens * hidden];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(out_f16.as_mut_ptr() as usize),
            d_o,
            n_tokens * hidden * 2,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_e, expert_outs_f16.len() * 2)?;
        dev.dealloc(d_w, weights_f32.len() * 4)?;
        dev.dealloc(d_r, residual_f16.len() * 2)?;
        dev.dealloc(d_o, n_tokens * hidden * 2)?;
    }
    let got: Vec<f32> = out_f16.iter().map(|v| v.to_f32()).collect();
    // Reference: same math in F32 with F16-casted inputs.
    let e_in: Vec<f32> = expert_outs_f16.iter().map(|v| v.to_f32()).collect();
    let r_in: Vec<f32> = residual_f16.iter().map(|v| v.to_f32()).collect();
    let mut reference = vec![0.0f32; n_tokens * hidden];
    for t in 0..n_tokens {
        for d in 0..hidden {
            let mut acc = r_in[t * hidden + d];
            for k in 0..top_k {
                let w = weights_f32[t * top_k + k];
                let e = e_in[(t * top_k + k) * hidden + d];
                acc += w * e;
            }
            reference[t * hidden + d] = f16::from_f32(acc).to_f32();
        }
    }
    Ok(max_rel_err(&got, &reference))
}

// ---------------------------------------------------------------------------
// 2.b — Q8_0 indexed-MoE tile8 MMQ certs.
// Fused gate+up and standalone down kernels on Q8_0 expert weights. Weight
// dtype identical to the existing 2.a Q8_0 MoE MMVQ path — this cert
// gates the tile8 structural port specifically.
// ---------------------------------------------------------------------------

/// Build a pad-to-8 sort on host for a synthetic `expert_ids[n_tokens * top_k]`
/// array. Returns `(sorted_pair_idx_padded, padded_offsets)` matching the
/// layout produced on-device by `flambeau_moe_sort_*_padded` kernels (pairs
/// grouped by expert, each expert's range padded to a multiple of 8 by
/// repeating the last real pair index).
fn build_padded_sort_host(expert_ids: &[i32], n_experts: usize) -> (Vec<i32>, Vec<i32>) {
    let mut groups: Vec<Vec<i32>> = vec![Vec::new(); n_experts];
    for (pair, &e) in expert_ids.iter().enumerate() {
        groups[e as usize].push(pair as i32);
    }
    let mut sorted_padded = Vec::new();
    let mut padded_offsets = Vec::with_capacity(n_experts + 1);
    padded_offsets.push(0);
    for g in &groups {
        let padded_count = (g.len() + 7) & !7;
        for &p in g {
            sorted_padded.push(p);
        }
        if let Some(&last) = g.last() {
            for _ in g.len()..padded_count {
                sorted_padded.push(last);
            }
        }
        padded_offsets.push(sorted_padded.len() as i32);
    }
    (sorted_padded, padded_offsets)
}

pub fn run_indexed_moe_mmq_q8_0_gate_up_tile8_sweep(repo_root: &Path) -> Result<Cert> {
    let dev = ensure_dev()?;
    let kb = kernels::hsaco("indexed_moe_mmq_q8_0_gate_up_tile8_dp4a").unwrap();
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> =
        module.kernel("flambeau_indexed_moe_mmq_q8_0_gate_up_tile8_dp4a_q8_1")?;
    let attrs: FuncAttributes = kernel.attributes()?;
    let q_kb = kernels::hsaco("quantize_q8_1").unwrap();
    let q_module = HipModule::load(dev.id(), q_kb)?;
    let q_kernel: HipKernel<'_> = q_module.kernel("flambeau_quantize_row_q8_1")?;

    let n_experts = 4usize;
    // Shapes hit tile8's n_tokens >= 32 regime + typical MoE gate/up widths.
    let cases = [
        (32usize, 4usize, 256usize, 2048usize), // gate/up at prefill L=32
        (128, 4, 256, 2048),                    // larger prefill
        (32, 4, 64, 768),                       // tall-thin shape (row < MMQ_Y=64)
    ];
    let mut results = Vec::new();
    for (n_tokens, top_k, n_rows, k_dim) in cases {
        let seed =
            0xDEC0DE ^ (n_tokens as u64 * 53 + top_k as u64 * 17 + n_rows as u64 * 7) ^ 0xB822u64;
        let max_rel = run_q8_0_tile8_gate_up_shape(
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
        impl_id: "indexed_moe_mmq_q8_0_gate_up_tile8_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "indexed_moe_mmq".to_string(),
        dtype_weight: "Q8_0".to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula: "|err| <= 3e-2 * max(|ref|, sqrt(k))  (Q8_0 tile8 fused gate+up)"
            .to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_tag(),
        pmc: Some(pmc_from(&attrs)),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

pub fn run_indexed_moe_mmq_q8_0_down_tile8_sweep(repo_root: &Path) -> Result<Cert> {
    let dev = ensure_dev()?;
    let kb = kernels::hsaco("indexed_moe_mmq_q8_0_down_tile8_dp4a").unwrap();
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> =
        module.kernel("flambeau_indexed_moe_mmq_q8_0_down_tile8_dp4a_q8_1")?;
    let attrs: FuncAttributes = kernel.attributes()?;
    let q_kb = kernels::hsaco("quantize_q8_1").unwrap();
    let q_module = HipModule::load(dev.id(), q_kb)?;
    let q_kernel: HipKernel<'_> = q_module.kernel("flambeau_quantize_row_q8_1")?;

    let n_experts = 4usize;
    // Down path: each pair is its own "effective token", top_k_inner = 1.
    // Caller in moe.rs passes n_tokens = n_pairs, top_k = 1.
    let cases = [
        (128usize, 1usize, 2048usize, 768usize), // n_pairs=128, down shape
        (256, 1, 2048, 768),
        (128, 1, 128, 768), // row < MMQ_Y
    ];
    let mut results = Vec::new();
    for (n_tokens, top_k, n_rows, k_dim) in cases {
        let seed =
            0xDEC0DE ^ (n_tokens as u64 * 53 + top_k as u64 * 17 + n_rows as u64 * 7) ^ 0xB844u64;
        let max_rel = run_q8_0_tile8_down_shape(
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
        impl_id: "indexed_moe_mmq_q8_0_down_tile8_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "indexed_moe_mmq".to_string(),
        dtype_weight: "Q8_0".to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula: "|err| <= 3e-2 * max(|ref|, sqrt(k))  (Q8_0 tile8 down)".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_tag(),
        pmc: Some(pmc_from(&attrs)),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

fn quantize_weights_q8_0(xs: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(xs.len() / QK8 * 34);
    for block in xs.chunks_exact(QK8) {
        let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let d_f16 = f16::from_f32(d);
        out.extend_from_slice(&d_f16.to_bits().to_le_bytes());
        for &v in block {
            let q = (v * id).round().clamp(-127.0, 127.0) as i8;
            out.push(q as u8);
        }
    }
    out
}

fn run_q8_0_tile8_gate_up_shape(
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
    assert_eq!(k_dim % QK8, 0);
    let nb_per_row = k_dim / QK8;

    let w_elems = n_experts * n_rows * k_dim;
    let gate_f32 = seeded_f32(seed, w_elems);
    let up_f32 = seeded_f32(seed.wrapping_add(0x71), w_elems);
    let gate_q = quantize_weights_q8_0(&gate_f32);
    let up_q = quantize_weights_q8_0(&up_f32);

    // Activation is per-TOKEN (not per-pair) for gate+up. n_tokens rows of k.
    let act_f32 = seeded_f32(seed.wrapping_add(0xA1), n_tokens * k_dim);

    // Uniform-ish expert routing: pair i → expert (i * 2654435761) mod n_experts.
    let expert_ids: Vec<i32> = (0..n_tokens * top_k)
        .map(|i| {
            let h = (i as u64)
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(seed.wrapping_mul(0x12345));
            ((h >> 32) as u32 % n_experts as u32) as i32
        })
        .collect();
    let (sorted_padded, padded_offsets) = build_padded_sort_host(&expert_ids, n_experts);
    let padded_total = *padded_offsets.last().unwrap() as usize;

    let d_gate = upload(dev, &gate_q);
    let d_up = upload(dev, &up_q);
    let d_act = upload(dev, &act_f32);
    let y_blocks_total = n_tokens * nb_per_row;
    let d_y = dev.alloc(y_blocks_total * std::mem::size_of::<BlockQ8_1>())?;
    let d_ids = upload(dev, &expert_ids);
    let d_sorted = upload(dev, &sorted_padded);
    let d_pofs = upload(dev, &padded_offsets);
    let d_gate_out = dev.alloc(n_tokens * top_k * n_rows * 4)?;
    let d_up_out = dev.alloc(n_tokens * top_k * n_rows * 4)?;

    // Quantise activation F32 → Q8_1.
    {
        let stream = dev.default_stream();
        let n_elems = (n_tokens * k_dim) as i32;
        let d_a_p: u64 = d_act.as_usize() as u64;
        let d_y_p: u64 = d_y.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_a_p);
        args.push(&d_y_p);
        args.push(&n_elems);
        let cfg = LaunchCfg::one_d(y_blocks_total as u32, QK8 as u32);
        unsafe { q_kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    // Launch tile8 gate+up kernel.
    {
        let stream = dev.default_stream();
        let n_rows_i = n_rows as i32;
        let n_tokens_i = n_tokens as i32;
        let top_k_i = top_k as i32;
        let nb_i = nb_per_row as i32;
        let n_experts_i = n_experts as i32;
        let g_p: u64 = d_gate.as_usize() as u64;
        let u_p: u64 = d_up.as_usize() as u64;
        let y_p: u64 = d_y.as_usize() as u64;
        let e_p: u64 = d_ids.as_usize() as u64;
        let s_p: u64 = d_sorted.as_usize() as u64;
        let po_p: u64 = d_pofs.as_usize() as u64;
        let go_p: u64 = d_gate_out.as_usize() as u64;
        let uo_p: u64 = d_up_out.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&g_p);
        args.push(&u_p);
        args.push(&y_p);
        args.push(&e_p);
        args.push(&s_p);
        args.push(&po_p);
        args.push(&go_p);
        args.push(&uo_p);
        args.push(&n_rows_i);
        args.push(&n_tokens_i);
        args.push(&top_k_i);
        args.push(&nb_i);
        args.push(&n_experts_i);
        let grid_y = padded_total.div_ceil(8) as u32;
        let cfg = LaunchCfg {
            grid: ((n_rows as u32).div_ceil(64), grid_y, 1),
            block: (64, 1, 1),
            shared_bytes: 0,
        };
        unsafe { kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    let mut got_gate = vec![0.0f32; n_tokens * top_k * n_rows];
    let mut got_up = vec![0.0f32; n_tokens * top_k * n_rows];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got_gate.as_mut_ptr() as usize),
            d_gate_out,
            n_tokens * top_k * n_rows * 4,
        )?;
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got_up.as_mut_ptr() as usize),
            d_up_out,
            n_tokens * top_k * n_rows * 4,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_gate, gate_q.len())?;
        dev.dealloc(d_up, up_q.len())?;
        dev.dealloc(d_act, act_f32.len() * 4)?;
        dev.dealloc(d_y, y_blocks_total * std::mem::size_of::<BlockQ8_1>())?;
        dev.dealloc(d_ids, expert_ids.len() * 4)?;
        dev.dealloc(d_sorted, sorted_padded.len() * 4)?;
        dev.dealloc(d_pofs, padded_offsets.len() * 4)?;
        dev.dealloc(d_gate_out, n_tokens * top_k * n_rows * 4)?;
        dev.dealloc(d_up_out, n_tokens * top_k * n_rows * 4)?;
    }

    // CPU F32 reference.
    let mut gate_dq = vec![0.0f32; w_elems];
    let mut up_dq = vec![0.0f32; w_elems];
    flambeau_quant::dequantize_into(flambeau_quant::GgmlDType::Q8_0, &gate_q, &mut gate_dq)?;
    flambeau_quant::dequantize_into(flambeau_quant::GgmlDType::Q8_0, &up_q, &mut up_dq)?;
    let act_rt = q8_1_roundtrip(&act_f32);
    let mut ref_gate = vec![0.0f32; n_tokens * top_k * n_rows];
    let mut ref_up = vec![0.0f32; n_tokens * top_k * n_rows];
    for t in 0..n_tokens {
        for slot in 0..top_k {
            let expert = expert_ids[t * top_k + slot] as usize;
            for row in 0..n_rows {
                let mut ag = 0.0f64;
                let mut au = 0.0f64;
                for j in 0..k_dim {
                    let wg = gate_dq[((expert * n_rows) + row) * k_dim + j];
                    let wu = up_dq[((expert * n_rows) + row) * k_dim + j];
                    let a = act_rt[t * k_dim + j];
                    ag += (wg * a) as f64;
                    au += (wu * a) as f64;
                }
                ref_gate[(t * top_k + slot) * n_rows + row] = ag as f32;
                ref_up[(t * top_k + slot) * n_rows + row] = au as f32;
            }
        }
    }
    let eg = max_rel_err_with_floor(&got_gate, &ref_gate, k_dim);
    let eu = max_rel_err_with_floor(&got_up, &ref_up, k_dim);
    Ok(eg.max(eu))
}

fn run_q8_0_tile8_down_shape(
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
    assert_eq!(k_dim % QK8, 0);
    assert_eq!(top_k, 1, "down path is re-indexed with top_k_inner=1");
    let nb_per_row = k_dim / QK8;

    let w_elems = n_experts * n_rows * k_dim;
    let w_f32 = seeded_f32(seed, w_elems);
    let w_q = quantize_weights_q8_0(&w_f32);

    // Down path: activation indexed per-PAIR (n_tokens = n_pairs).
    let act_f32 = seeded_f32(seed.wrapping_add(0xA1), n_tokens * k_dim);

    let expert_ids: Vec<i32> = (0..n_tokens)
        .map(|i| {
            let h = (i as u64)
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(seed.wrapping_mul(0x12345));
            ((h >> 32) as u32 % n_experts as u32) as i32
        })
        .collect();
    let (sorted_padded, padded_offsets) = build_padded_sort_host(&expert_ids, n_experts);
    let padded_total = *padded_offsets.last().unwrap() as usize;

    let d_w = upload(dev, &w_q);
    let d_act = upload(dev, &act_f32);
    let y_blocks_total = n_tokens * nb_per_row;
    let d_y = dev.alloc(y_blocks_total * std::mem::size_of::<BlockQ8_1>())?;
    let d_ids = upload(dev, &expert_ids);
    let d_sorted = upload(dev, &sorted_padded);
    let d_pofs = upload(dev, &padded_offsets);
    let d_dst = dev.alloc(n_tokens * n_rows * 4)?;

    // Quantise activation F32 → Q8_1.
    {
        let stream = dev.default_stream();
        let n_elems = (n_tokens * k_dim) as i32;
        let d_a_p: u64 = d_act.as_usize() as u64;
        let d_y_p: u64 = d_y.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_a_p);
        args.push(&d_y_p);
        args.push(&n_elems);
        let cfg = LaunchCfg::one_d(y_blocks_total as u32, QK8 as u32);
        unsafe { q_kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    {
        let stream = dev.default_stream();
        let n_rows_i = n_rows as i32;
        let n_tokens_i = n_tokens as i32;
        let top_k_i = top_k as i32;
        let nb_i = nb_per_row as i32;
        let n_experts_i = n_experts as i32;
        let w_p: u64 = d_w.as_usize() as u64;
        let y_p: u64 = d_y.as_usize() as u64;
        let e_p: u64 = d_ids.as_usize() as u64;
        let s_p: u64 = d_sorted.as_usize() as u64;
        let po_p: u64 = d_pofs.as_usize() as u64;
        let dst_p: u64 = d_dst.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&w_p);
        args.push(&y_p);
        args.push(&e_p);
        args.push(&s_p);
        args.push(&po_p);
        args.push(&dst_p);
        args.push(&n_rows_i);
        args.push(&n_tokens_i);
        args.push(&top_k_i);
        args.push(&nb_i);
        args.push(&n_experts_i);
        let grid_y = padded_total.div_ceil(8) as u32;
        let cfg = LaunchCfg {
            grid: ((n_rows as u32).div_ceil(64), grid_y, 1),
            block: (64, 1, 1),
            shared_bytes: 0,
        };
        unsafe { kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    let mut got = vec![0.0f32; n_tokens * n_rows];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got.as_mut_ptr() as usize),
            d_dst,
            n_tokens * n_rows * 4,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_w, w_q.len())?;
        dev.dealloc(d_act, act_f32.len() * 4)?;
        dev.dealloc(d_y, y_blocks_total * std::mem::size_of::<BlockQ8_1>())?;
        dev.dealloc(d_ids, expert_ids.len() * 4)?;
        dev.dealloc(d_sorted, sorted_padded.len() * 4)?;
        dev.dealloc(d_pofs, padded_offsets.len() * 4)?;
        dev.dealloc(d_dst, n_tokens * n_rows * 4)?;
    }

    let mut w_dq = vec![0.0f32; w_elems];
    flambeau_quant::dequantize_into(flambeau_quant::GgmlDType::Q8_0, &w_q, &mut w_dq)?;
    let act_rt = q8_1_roundtrip(&act_f32);
    let mut reference = vec![0.0f32; n_tokens * n_rows];
    for pair in 0..n_tokens {
        let expert = expert_ids[pair] as usize;
        for row in 0..n_rows {
            let mut acc = 0.0f64;
            for j in 0..k_dim {
                let w = w_dq[((expert * n_rows) + row) * k_dim + j];
                let a = act_rt[pair * k_dim + j];
                acc += (w * a) as f64;
            }
            reference[pair * n_rows + row] = acc as f32;
        }
    }
    Ok(max_rel_err_with_floor(&got, &reference, k_dim))
}

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

fn ensure_dev() -> Result<HipDevice> {
    let n = device_count().context("hipGetDeviceCount")?;
    if n < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;
    Ok(dev)
}

fn rig_tag() -> String {
    rig()
}

fn pmc_from(a: &FuncAttributes) -> PmcSnapshot {
    PmcSnapshot {
        vgpr_count: Some(a.num_regs),
        sgpr_count: None,
        waves_per_simd: Some(a.gfx906_waves_per_simd()),
        mem_busy_pct: None,
        valu_busy_pct: None,
    }
}

fn seeded_f32(seed: u64, n: usize) -> Vec<f32> {
    seeded_f32_range(seed, n, -0.5, 0.5)
}

fn seeded_bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 24) as u8
        })
        .collect()
}

fn upload<T: Copy>(dev: &HipDevice, data: &[T]) -> DevicePtr {
    alloc_and_upload(dev, data)
}

fn max_rel_err(got: &[f32], reference: &[f32]) -> f32 {
    harness_err_floor(got, reference, 1.0)
}

fn max_rel_err_with_floor(got: &[f32], reference: &[f32], k: usize) -> f32 {
    harness_err_floor(got, reference, (k as f32).sqrt())
}

// ---------------------------------------------------------------------------
// Q3_K + Q2_K indexed-MoE MMVQ correctness sweeps
// ---------------------------------------------------------------------------

pub fn run_indexed_moe_mmvq_q3_k_sweep(repo_root: &Path) -> Result<Cert> {
    let dev = ensure_dev()?;
    let kb = kernels::hsaco("indexed_moe_mmvq_q3_k").unwrap();
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_indexed_moe_mmvq_q3_k_q8_1")?;
    let attrs: FuncAttributes = kernel.attributes()?;
    let q_kb = kernels::hsaco("quantize_q8_1").unwrap();
    let q_module = HipModule::load(dev.id(), q_kb)?;
    let q_kernel: HipKernel<'_> = q_module.kernel("flambeau_quantize_row_q8_1")?;

    let n_experts = 16usize;
    let cases = [(1usize, 4usize, 256usize, 512usize), (4, 4, 256, 512)];
    let mut results = Vec::new();
    for (n_tokens, top_k, n_rows, k_dim) in cases {
        let seed = 0xDEC0DE ^ (n_tokens as u64 * 53 + top_k as u64 * 17) ^ 0xD3D3_u64;
        let max_rel = run_q3k_shape(
            &dev, &kernel, &q_kernel, n_experts, n_rows, k_dim, top_k, n_tokens, seed,
        )?;
        let tol = 5e-2;
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
        impl_id: "indexed_moe_mmvq_q3_k_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "indexed_moe_mmvq".to_string(),
        dtype_weight: "Q3_K".to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula:
            "|err| <= 5e-2 * max(|ref|, sqrt(k))  (Q3_K weights; same envelope as Q4_K MoE MMVQ)"
                .to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_tag(),
        pmc: Some(pmc_from(&attrs)),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

pub fn run_indexed_moe_mmvq_q2_k_sweep(repo_root: &Path) -> Result<Cert> {
    let dev = ensure_dev()?;
    let kb = kernels::hsaco("indexed_moe_mmvq_q2_k").unwrap();
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_indexed_moe_mmvq_q2_K_q8_1")?;
    let attrs: FuncAttributes = kernel.attributes()?;
    let q_kb = kernels::hsaco("quantize_q8_1").unwrap();
    let q_module = HipModule::load(dev.id(), q_kb)?;
    let q_kernel: HipKernel<'_> = q_module.kernel("flambeau_quantize_row_q8_1")?;

    let n_experts = 16usize;
    let cases = [(1usize, 4usize, 256usize, 512usize), (4, 4, 256, 512)];
    let mut results = Vec::new();
    for (n_tokens, top_k, n_rows, k_dim) in cases {
        let seed = 0xDEC0DE ^ (n_tokens as u64 * 53 + top_k as u64 * 17) ^ 0xD2D2_u64;
        let max_rel = run_q2k_shape(
            &dev, &kernel, &q_kernel, n_experts, n_rows, k_dim, top_k, n_tokens, seed,
        )?;
        let tol = 5e-2;
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
        impl_id: "indexed_moe_mmvq_q2_k_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "indexed_moe_mmvq".to_string(),
        dtype_weight: "Q2_K".to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula:
            "|err| <= 5e-2 * max(|ref|, sqrt(k))  (Q2_K weights; same envelope as Q4_K MoE MMVQ)"
                .to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_tag(),
        pmc: Some(pmc_from(&attrs)),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

fn run_q3k_shape(
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
    assert_eq!(k_dim % QK_K, 0);
    let nb_per_row = k_dim / QK_K;

    let total_blocks = n_experts * n_rows * nb_per_row;
    let w_bytes = total_blocks * std::mem::size_of::<BlockQ3K>();
    let w_raw = tame_q3k_scales(seeded_bytes(seed, w_bytes));

    let act_f32 = seeded_f32(seed.wrapping_add(0xA1), n_tokens * k_dim);
    let expert_ids: Vec<i32> = (0..n_tokens * top_k)
        .map(|i| {
            let h = (i as u64)
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(seed.wrapping_mul(0x12345));
            ((h >> 32) as u32 % n_experts as u32) as i32
        })
        .collect();

    let d_w = upload(dev, &w_raw);
    let d_act = upload(dev, &act_f32);
    let y_blocks_total = n_tokens * nb_per_row * 8;
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
        let cfg = LaunchCfg::one_d(y_blocks_total as u32, QK8 as u32);
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
            block: (64, 1, 1),
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
        dev.dealloc(d_w, w_bytes)?;
        dev.dealloc(d_act, act_f32.len() * 4)?;
        dev.dealloc(d_y, y_blocks_total * std::mem::size_of::<BlockQ8_1>())?;
        dev.dealloc(d_ids, expert_ids.len() * 4)?;
        dev.dealloc(d_dst, n_tokens * top_k * n_rows * 4)?;
    }

    let total_elems = n_experts * n_rows * k_dim;
    let mut w_dequant = vec![0.0f32; total_elems];
    flambeau_quant::dequantize_into(flambeau_quant::GgmlDType::Q3K, &w_raw, &mut w_dequant)?;
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
    Ok(max_rel_err_with_floor(&got, &reference, k_dim))
}

fn run_q2k_shape(
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
    assert_eq!(k_dim % QK_K, 0);
    let nb_per_row = k_dim / QK_K;

    let total_blocks = n_experts * n_rows * nb_per_row;
    let w_bytes = total_blocks * std::mem::size_of::<BlockQ2K>();
    let w_raw = tame_q2k_scales(seeded_bytes(seed, w_bytes));

    let act_f32 = seeded_f32(seed.wrapping_add(0xA1), n_tokens * k_dim);
    let expert_ids: Vec<i32> = (0..n_tokens * top_k)
        .map(|i| {
            let h = (i as u64)
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(seed.wrapping_mul(0x12345));
            ((h >> 32) as u32 % n_experts as u32) as i32
        })
        .collect();

    let d_w = upload(dev, &w_raw);
    let d_act = upload(dev, &act_f32);
    let y_blocks_total = n_tokens * nb_per_row * 8;
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
        let cfg = LaunchCfg::one_d(y_blocks_total as u32, QK8 as u32);
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
            block: (64, 1, 1),
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
        dev.dealloc(d_w, w_bytes)?;
        dev.dealloc(d_act, act_f32.len() * 4)?;
        dev.dealloc(d_y, y_blocks_total * std::mem::size_of::<BlockQ8_1>())?;
        dev.dealloc(d_ids, expert_ids.len() * 4)?;
        dev.dealloc(d_dst, n_tokens * top_k * n_rows * 4)?;
    }

    let total_elems = n_experts * n_rows * k_dim;
    let mut w_dequant = vec![0.0f32; total_elems];
    flambeau_quant::dequantize_into(flambeau_quant::GgmlDType::Q2K, &w_raw, &mut w_dequant)?;
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
    Ok(max_rel_err_with_floor(&got, &reference, k_dim))
}

fn tame_q3k_scales(mut raw: Vec<u8>) -> Vec<u8> {
    // Same treatment as the Q3_K MMVQ sweep: signed-6-bit scale bytes
    // (the 12 packed bytes at offset 96..108) clamped via mod-32; d (f16
    // at 108..110) bounded small.
    let bs = std::mem::size_of::<BlockQ3K>();
    let nblocks = raw.len() / bs;
    let scales_off = QK_K / 8 + QK_K / 4; // 32+64 = 96
    let d_off = scales_off + 12;
    for i in 0..nblocks {
        let block = &mut raw[i * bs..(i + 1) * bs];
        for s in &mut block[scales_off..scales_off + 12] {
            *s = (*s as i32 % 32) as u8;
        }
        let d = f16::from_f32((block[d_off] as f32 / 255.0) * 0.05 + 0.005);
        block[d_off..d_off + 2].copy_from_slice(&d.to_bits().to_le_bytes());
    }
    raw
}

fn tame_q2k_scales(mut raw: Vec<u8>) -> Vec<u8> {
    // BlockQ2K: scales[16] + qs[64] + d (f16 @ 80) + dmin (f16 @ 82).
    // Scales bytes are 4-bit (scale, min) packed; leave random. Bound d/dmin.
    let bs = std::mem::size_of::<BlockQ2K>();
    let nblocks = raw.len() / bs;
    for i in 0..nblocks {
        let block = &mut raw[i * bs..(i + 1) * bs];
        let d = f16::from_f32((block[80] as f32 / 255.0) * 0.05 + 0.005);
        let dmin = f16::from_f32((block[81] as f32 / 255.0) * 0.02);
        block[80..82].copy_from_slice(&d.to_bits().to_le_bytes());
        block[82..84].copy_from_slice(&dmin.to_bits().to_le_bytes());
    }
    raw
}

// ===========================================================================
// Tile8 MoE MMQ cert harness — generic over weight dtype
// ===========================================================================
//
// Covers the 7 dtypes that already have tile8 down/gate_up kernels in tree:
// Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K. Each dtype has two sweep entries
// (one for gate_up, one for down). The shape functions are generic over
// `Tile8Wk` — a (ggml, block_bytes, tame_fn) tag — so the per-dtype sweep
// wrappers stay thin.

#[derive(Clone, Copy)]
enum Tile8Wk {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
}

impl Tile8Wk {
    fn ggml(self) -> GgmlDType {
        match self {
            Tile8Wk::Q4_0 => GgmlDType::Q4_0,
            Tile8Wk::Q4_1 => GgmlDType::Q4_1,
            Tile8Wk::Q5_0 => GgmlDType::Q5_0,
            Tile8Wk::Q5_1 => GgmlDType::Q5_1,
            Tile8Wk::Q2K => GgmlDType::Q2K,
            Tile8Wk::Q3K => GgmlDType::Q3K,
            Tile8Wk::Q4K => GgmlDType::Q4K,
            Tile8Wk::Q5K => GgmlDType::Q5K,
            Tile8Wk::Q6K => GgmlDType::Q6K,
        }
    }

    fn block_bytes(self) -> usize {
        match self {
            Tile8Wk::Q4_0 => std::mem::size_of::<flambeau_quant::BlockQ4_0>(),
            Tile8Wk::Q4_1 => std::mem::size_of::<BlockQ4_1>(),
            Tile8Wk::Q5_0 => std::mem::size_of::<flambeau_quant::BlockQ5_0>(),
            Tile8Wk::Q5_1 => std::mem::size_of::<flambeau_quant::BlockQ5_1>(),
            Tile8Wk::Q2K => std::mem::size_of::<BlockQ2K>(),
            Tile8Wk::Q3K => std::mem::size_of::<BlockQ3K>(),
            Tile8Wk::Q4K => std::mem::size_of::<BlockQ4K>(),
            Tile8Wk::Q5K => std::mem::size_of::<BlockQ5K>(),
            Tile8Wk::Q6K => std::mem::size_of::<BlockQ6K>(),
        }
    }

    fn block_elems(self) -> usize {
        self.ggml().block_size()
    }

    fn tol(self) -> f32 {
        match self {
            Tile8Wk::Q2K | Tile8Wk::Q3K | Tile8Wk::Q4K | Tile8Wk::Q5K | Tile8Wk::Q6K => 5e-2,
            _ => 3e-2,
        }
    }

    fn tame_block(self, block: &mut [u8]) {
        match self {
            Tile8Wk::Q4_0 => {
                let d = f16::from_f32((block[0] as f32 / 255.0) * 0.1 + 0.01);
                block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
            }
            Tile8Wk::Q4_1 => {
                let d = f16::from_f32((block[0] as f32 / 255.0) * 0.1 + 0.01);
                let m = f16::from_f32((block[1] as f32 / 255.0) * 0.05 - 0.025);
                block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
                block[2..4].copy_from_slice(&m.to_bits().to_le_bytes());
            }
            Tile8Wk::Q5_0 => {
                let d = f16::from_f32((block[0] as f32 / 255.0) * 0.1 + 0.01);
                block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
            }
            Tile8Wk::Q5_1 => {
                let d = f16::from_f32((block[0] as f32 / 255.0) * 0.1 + 0.01);
                let m = f16::from_f32((block[1] as f32 / 255.0) * 0.05 - 0.025);
                block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
                block[2..4].copy_from_slice(&m.to_bits().to_le_bytes());
            }
            Tile8Wk::Q4K | Tile8Wk::Q5K => {
                let d = f16::from_f32((block[0] as f32 / 255.0) * 0.1 + 0.01);
                let dmin = f16::from_f32((block[1] as f32 / 255.0) * 0.05);
                block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
                block[2..4].copy_from_slice(&dmin.to_bits().to_le_bytes());
            }
            Tile8Wk::Q2K => {
                // Q2_K: scales[16] + qs[64] + d (f16 @ 80) + dmin (f16 @ 82).
                let d = f16::from_f32((block[80] as f32 / 255.0) * 0.05 + 0.005);
                let dmin = f16::from_f32((block[81] as f32 / 255.0) * 0.02);
                block[80..82].copy_from_slice(&d.to_bits().to_le_bytes());
                block[82..84].copy_from_slice(&dmin.to_bits().to_le_bytes());
            }
            Tile8Wk::Q3K => {
                // Q3_K: hmask[32] + qs[64] + scales[12] + d (f16 @ 108).
                let scales_off = QK_K / 8 + QK_K / 4;
                for s in &mut block[scales_off..scales_off + 12] {
                    *s = (*s as i32 % 32) as u8;
                }
                let d_off = scales_off + 12;
                let d = f16::from_f32((block[d_off] as f32 / 255.0) * 0.05 + 0.005);
                block[d_off..d_off + 2].copy_from_slice(&d.to_bits().to_le_bytes());
            }
            Tile8Wk::Q6K => {
                let scales_off = QK_K / 2 + QK_K / 4; // 192
                for s in &mut block[scales_off..scales_off + QK_K / 16] {
                    let sv = (*s as i32 % 65) - 32;
                    *s = sv as u8;
                }
                let d_off = scales_off + QK_K / 16;
                let d = f16::from_f32((block[d_off] as f32 / 255.0) * 0.05 + 0.01);
                block[d_off..d_off + 2].copy_from_slice(&d.to_bits().to_le_bytes());
            }
        }
    }

    fn tame_bytes(self, mut raw: Vec<u8>) -> Vec<u8> {
        let bs = self.block_bytes();
        for chunk in raw.chunks_exact_mut(bs) {
            self.tame_block(chunk);
        }
        raw
    }
}

fn run_tile8_gate_up_shape(
    dev: &HipDevice,
    kernel: &HipKernel<'_>,
    q_kernel: &HipKernel<'_>,
    wk: Tile8Wk,
    n_experts: usize,
    n_rows: usize,
    k_dim: usize,
    top_k: usize,
    n_tokens: usize,
    seed: u64,
) -> Result<f32> {
    let block_elems = wk.block_elems();
    assert_eq!(k_dim % block_elems, 0);
    let nb_per_row = k_dim / block_elems;
    let bs = wk.block_bytes();

    let w_blocks = n_experts * n_rows * nb_per_row;
    let w_bytes = w_blocks * bs;
    let gate_q = wk.tame_bytes(seeded_bytes(seed, w_bytes));
    let up_q = wk.tame_bytes(seeded_bytes(seed.wrapping_add(0x71), w_bytes));

    let act_f32 = seeded_f32(seed.wrapping_add(0xA1), n_tokens * k_dim);

    let expert_ids: Vec<i32> = (0..n_tokens * top_k)
        .map(|i| {
            let h = (i as u64)
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(seed.wrapping_mul(0x12345));
            ((h >> 32) as u32 % n_experts as u32) as i32
        })
        .collect();
    let (sorted_padded, padded_offsets) = build_padded_sort_host(&expert_ids, n_experts);
    let padded_total = *padded_offsets.last().unwrap() as usize;

    let d_gate = upload(dev, &gate_q);
    let d_up = upload(dev, &up_q);
    let d_act = upload(dev, &act_f32);
    // tile8 gate_up activation uses Q8_1 with QK8_1 (=32) block size — independent
    // of the weight super-block size.
    let y_blocks_total = n_tokens * (k_dim / QK8);
    let d_y = dev.alloc(y_blocks_total * std::mem::size_of::<BlockQ8_1>())?;
    let d_ids = upload(dev, &expert_ids);
    let d_sorted = upload(dev, &sorted_padded);
    let d_pofs = upload(dev, &padded_offsets);
    let d_gate_out = dev.alloc(n_tokens * top_k * n_rows * 4)?;
    let d_up_out = dev.alloc(n_tokens * top_k * n_rows * 4)?;

    {
        let stream = dev.default_stream();
        let n_elems = (n_tokens * k_dim) as i32;
        let d_a_p: u64 = d_act.as_usize() as u64;
        let d_y_p: u64 = d_y.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_a_p);
        args.push(&d_y_p);
        args.push(&n_elems);
        let cfg = LaunchCfg::one_d(y_blocks_total as u32, QK8 as u32);
        unsafe { q_kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    {
        let stream = dev.default_stream();
        let n_rows_i = n_rows as i32;
        let n_tokens_i = n_tokens as i32;
        let top_k_i = top_k as i32;
        // The tile8 kernels take `n_blocks_per_row` in the *weight* super-block
        // unit (one Q*_K super-block, or one QK8_0 block for legacy quants).
        let nb_i = nb_per_row as i32;
        let n_experts_i = n_experts as i32;
        let g_p: u64 = d_gate.as_usize() as u64;
        let u_p: u64 = d_up.as_usize() as u64;
        let y_p: u64 = d_y.as_usize() as u64;
        let e_p: u64 = d_ids.as_usize() as u64;
        let s_p: u64 = d_sorted.as_usize() as u64;
        let po_p: u64 = d_pofs.as_usize() as u64;
        let go_p: u64 = d_gate_out.as_usize() as u64;
        let uo_p: u64 = d_up_out.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&g_p);
        args.push(&u_p);
        args.push(&y_p);
        args.push(&e_p);
        args.push(&s_p);
        args.push(&po_p);
        args.push(&go_p);
        args.push(&uo_p);
        args.push(&n_rows_i);
        args.push(&n_tokens_i);
        args.push(&top_k_i);
        args.push(&nb_i);
        args.push(&n_experts_i);
        let grid_y = padded_total.div_ceil(8) as u32;
        let cfg = LaunchCfg {
            grid: ((n_rows as u32).div_ceil(64), grid_y, 1),
            block: (64, 1, 1),
            shared_bytes: 0,
        };
        unsafe { kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    let mut got_gate = vec![0.0f32; n_tokens * top_k * n_rows];
    let mut got_up = vec![0.0f32; n_tokens * top_k * n_rows];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got_gate.as_mut_ptr() as usize),
            d_gate_out,
            n_tokens * top_k * n_rows * 4,
        )?;
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got_up.as_mut_ptr() as usize),
            d_up_out,
            n_tokens * top_k * n_rows * 4,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_gate, w_bytes)?;
        dev.dealloc(d_up, w_bytes)?;
        dev.dealloc(d_act, act_f32.len() * 4)?;
        dev.dealloc(d_y, y_blocks_total * std::mem::size_of::<BlockQ8_1>())?;
        dev.dealloc(d_ids, expert_ids.len() * 4)?;
        dev.dealloc(d_sorted, sorted_padded.len() * 4)?;
        dev.dealloc(d_pofs, padded_offsets.len() * 4)?;
        dev.dealloc(d_gate_out, n_tokens * top_k * n_rows * 4)?;
        dev.dealloc(d_up_out, n_tokens * top_k * n_rows * 4)?;
    }

    let w_elems = n_experts * n_rows * k_dim;
    let mut gate_dq = vec![0.0f32; w_elems];
    let mut up_dq = vec![0.0f32; w_elems];
    flambeau_quant::dequantize_into(wk.ggml(), &gate_q, &mut gate_dq)?;
    flambeau_quant::dequantize_into(wk.ggml(), &up_q, &mut up_dq)?;
    let act_rt = q8_1_roundtrip(&act_f32);
    let mut ref_gate = vec![0.0f32; n_tokens * top_k * n_rows];
    let mut ref_up = vec![0.0f32; n_tokens * top_k * n_rows];
    for t in 0..n_tokens {
        for slot in 0..top_k {
            let expert = expert_ids[t * top_k + slot] as usize;
            for row in 0..n_rows {
                let mut ag = 0.0f64;
                let mut au = 0.0f64;
                for j in 0..k_dim {
                    let wg = gate_dq[((expert * n_rows) + row) * k_dim + j];
                    let wu = up_dq[((expert * n_rows) + row) * k_dim + j];
                    let a = act_rt[t * k_dim + j];
                    ag += (wg * a) as f64;
                    au += (wu * a) as f64;
                }
                ref_gate[(t * top_k + slot) * n_rows + row] = ag as f32;
                ref_up[(t * top_k + slot) * n_rows + row] = au as f32;
            }
        }
    }
    let eg = max_rel_err_with_floor(&got_gate, &ref_gate, k_dim);
    let eu = max_rel_err_with_floor(&got_up, &ref_up, k_dim);
    Ok(eg.max(eu))
}

fn run_tile8_down_shape(
    dev: &HipDevice,
    kernel: &HipKernel<'_>,
    q_kernel: &HipKernel<'_>,
    wk: Tile8Wk,
    n_experts: usize,
    n_rows: usize,
    k_dim: usize,
    n_tokens: usize,
    seed: u64,
) -> Result<f32> {
    let block_elems = wk.block_elems();
    assert_eq!(k_dim % block_elems, 0);
    let nb_per_row = k_dim / block_elems;
    let bs = wk.block_bytes();
    let top_k: usize = 1; // down is per-pair indexed

    let w_blocks = n_experts * n_rows * nb_per_row;
    let w_bytes = w_blocks * bs;
    let w_q = wk.tame_bytes(seeded_bytes(seed, w_bytes));

    let act_f32 = seeded_f32(seed.wrapping_add(0xA1), n_tokens * k_dim);

    let expert_ids: Vec<i32> = (0..n_tokens)
        .map(|i| {
            let h = (i as u64)
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(seed.wrapping_mul(0x12345));
            ((h >> 32) as u32 % n_experts as u32) as i32
        })
        .collect();
    let (sorted_padded, padded_offsets) = build_padded_sort_host(&expert_ids, n_experts);
    let padded_total = *padded_offsets.last().unwrap() as usize;

    let d_w = upload(dev, &w_q);
    let d_act = upload(dev, &act_f32);
    let y_blocks_total = n_tokens * (k_dim / QK8);
    let d_y = dev.alloc(y_blocks_total * std::mem::size_of::<BlockQ8_1>())?;
    let d_ids = upload(dev, &expert_ids);
    let d_sorted = upload(dev, &sorted_padded);
    let d_pofs = upload(dev, &padded_offsets);
    let d_dst = dev.alloc(n_tokens * n_rows * 4)?;

    {
        let stream = dev.default_stream();
        let n_elems = (n_tokens * k_dim) as i32;
        let d_a_p: u64 = d_act.as_usize() as u64;
        let d_y_p: u64 = d_y.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_a_p);
        args.push(&d_y_p);
        args.push(&n_elems);
        let cfg = LaunchCfg::one_d(y_blocks_total as u32, QK8 as u32);
        unsafe { q_kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    {
        let stream = dev.default_stream();
        let n_rows_i = n_rows as i32;
        let n_tokens_i = n_tokens as i32;
        let top_k_i = top_k as i32;
        let nb_i = nb_per_row as i32;
        let n_experts_i = n_experts as i32;
        let w_p: u64 = d_w.as_usize() as u64;
        let y_p: u64 = d_y.as_usize() as u64;
        let e_p: u64 = d_ids.as_usize() as u64;
        let s_p: u64 = d_sorted.as_usize() as u64;
        let po_p: u64 = d_pofs.as_usize() as u64;
        let d_p: u64 = d_dst.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&w_p);
        args.push(&y_p);
        args.push(&e_p);
        args.push(&s_p);
        args.push(&po_p);
        args.push(&d_p);
        args.push(&n_rows_i);
        args.push(&n_tokens_i);
        args.push(&top_k_i);
        args.push(&nb_i);
        args.push(&n_experts_i);
        let grid_y = padded_total.div_ceil(8) as u32;
        let cfg = LaunchCfg {
            grid: ((n_rows as u32).div_ceil(64), grid_y, 1),
            block: (64, 1, 1),
            shared_bytes: 0,
        };
        unsafe { kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    let mut got = vec![0.0f32; n_tokens * n_rows];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got.as_mut_ptr() as usize),
            d_dst,
            n_tokens * n_rows * 4,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_w, w_bytes)?;
        dev.dealloc(d_act, act_f32.len() * 4)?;
        dev.dealloc(d_y, y_blocks_total * std::mem::size_of::<BlockQ8_1>())?;
        dev.dealloc(d_ids, expert_ids.len() * 4)?;
        dev.dealloc(d_sorted, sorted_padded.len() * 4)?;
        dev.dealloc(d_pofs, padded_offsets.len() * 4)?;
        dev.dealloc(d_dst, n_tokens * n_rows * 4)?;
    }

    let w_elems = n_experts * n_rows * k_dim;
    let mut w_dq = vec![0.0f32; w_elems];
    flambeau_quant::dequantize_into(wk.ggml(), &w_q, &mut w_dq)?;
    let act_rt = q8_1_roundtrip(&act_f32);
    let mut reference = vec![0.0f32; n_tokens * n_rows];
    for t in 0..n_tokens {
        let expert = expert_ids[t] as usize;
        for row in 0..n_rows {
            let mut acc = 0.0f64;
            for j in 0..k_dim {
                let w = w_dq[((expert * n_rows) + row) * k_dim + j];
                let a = act_rt[t * k_dim + j];
                acc += (w * a) as f64;
            }
            reference[t * n_rows + row] = acc as f32;
        }
    }
    Ok(max_rel_err_with_floor(&got, &reference, k_dim))
}

fn tile8_gate_up_cases(wk: Tile8Wk) -> Vec<(usize, usize, usize, usize)> {
    // (n_tokens, top_k, n_rows, k_dim). Pick k_dim divisible by the dtype's
    // super-block size so the harness assert holds.
    let k = if wk.block_elems() == 256 { 768 } else { 768 };
    vec![(1, 4, 256, k), (8, 4, 256, k)]
}

fn tile8_down_cases(wk: Tile8Wk) -> Vec<(usize, usize, usize)> {
    // (n_tokens=n_pairs, n_rows, k_dim).
    let k = if wk.block_elems() == 256 { 768 } else { 768 };
    vec![(128, 2048, k), (256, 2048, k), (128, 128, k)]
}

fn run_tile8_gate_up_sweep(
    repo_root: &Path,
    wk: Tile8Wk,
    impl_id: &'static str,
    module_stem: &'static str,
    entry: &'static str,
    dtype_name: &'static str,
    extra_seed: u64,
) -> Result<Cert> {
    let dev = ensure_dev()?;
    let kb = kernels::hsaco(module_stem).unwrap();
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel(entry)?;
    let attrs: FuncAttributes = kernel.attributes()?;
    let q_kb = kernels::hsaco("quantize_q8_1").unwrap();
    let q_module = HipModule::load(dev.id(), q_kb)?;
    let q_kernel: HipKernel<'_> = q_module.kernel("flambeau_quantize_row_q8_1")?;

    let n_experts = 4usize;
    let mut results = Vec::new();
    for (n_tokens, top_k, n_rows, k_dim) in tile8_gate_up_cases(wk) {
        let seed =
            0xDEC0DE ^ (n_tokens as u64 * 53 + top_k as u64 * 17 + n_rows as u64 * 7) ^ extra_seed;
        let max_rel = run_tile8_gate_up_shape(
            &dev, &kernel, &q_kernel, wk, n_experts, n_rows, k_dim, top_k, n_tokens, seed,
        )?;
        let tol = wk.tol();
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
        impl_id: impl_id.to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "indexed_moe_mmq".to_string(),
        dtype_weight: dtype_name.to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula: format!(
            "|err| <= {:.0e} * max(|ref|, sqrt(k))  ({dtype_name} tile8 gate_up)",
            wk.tol()
        ),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_tag(),
        pmc: Some(pmc_from(&attrs)),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

fn run_tile8_down_sweep(
    repo_root: &Path,
    wk: Tile8Wk,
    impl_id: &'static str,
    module_stem: &'static str,
    entry: &'static str,
    dtype_name: &'static str,
    extra_seed: u64,
) -> Result<Cert> {
    let dev = ensure_dev()?;
    let kb = kernels::hsaco(module_stem).unwrap();
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel(entry)?;
    let attrs: FuncAttributes = kernel.attributes()?;
    let q_kb = kernels::hsaco("quantize_q8_1").unwrap();
    let q_module = HipModule::load(dev.id(), q_kb)?;
    let q_kernel: HipKernel<'_> = q_module.kernel("flambeau_quantize_row_q8_1")?;

    let n_experts = 4usize;
    let mut results = Vec::new();
    for (n_tokens, n_rows, k_dim) in tile8_down_cases(wk) {
        let seed = 0xDEC0DE ^ (n_tokens as u64 * 53 + n_rows as u64 * 7) ^ extra_seed;
        let max_rel = run_tile8_down_shape(
            &dev, &kernel, &q_kernel, wk, n_experts, n_rows, k_dim, n_tokens, seed,
        )?;
        let tol = wk.tol();
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
        impl_id: impl_id.to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "indexed_moe_mmq".to_string(),
        dtype_weight: dtype_name.to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula: format!(
            "|err| <= {:.0e} * max(|ref|, sqrt(k))  ({dtype_name} tile8 down)",
            wk.tol()
        ),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_tag(),
        pmc: Some(pmc_from(&attrs)),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

// --- Per-dtype sweep entries ----------------------------------------------

macro_rules! tile8_sweeps {
    ($wk:expr, $tag:expr, $tag_lower:expr, $extra:expr, $gate_up_name:ident, $down_name:ident) => {
        pub fn $gate_up_name(repo_root: &Path) -> Result<Cert> {
            run_tile8_gate_up_sweep(
                repo_root,
                $wk,
                concat!("indexed_moe_mmq_", $tag_lower, "_gate_up_tile8_gfx906"),
                concat!("indexed_moe_mmq_", $tag_lower, "_gate_up_tile8_dp4a"),
                concat!(
                    "flambeau_indexed_moe_mmq_",
                    $tag_lower,
                    "_gate_up_tile8_dp4a_q8_1"
                ),
                $tag,
                $extra,
            )
        }
        pub fn $down_name(repo_root: &Path) -> Result<Cert> {
            run_tile8_down_sweep(
                repo_root,
                $wk,
                concat!("indexed_moe_mmq_", $tag_lower, "_down_tile8_gfx906"),
                concat!("indexed_moe_mmq_", $tag_lower, "_down_tile8_dp4a"),
                concat!(
                    "flambeau_indexed_moe_mmq_",
                    $tag_lower,
                    "_down_tile8_dp4a_q8_1"
                ),
                $tag,
                $extra,
            )
        }
    };
}

tile8_sweeps!(
    Tile8Wk::Q4_0,
    "Q4_0",
    "q4_0",
    0xB401,
    run_indexed_moe_mmq_q4_0_gate_up_tile8_sweep,
    run_indexed_moe_mmq_q4_0_down_tile8_sweep
);
tile8_sweeps!(
    Tile8Wk::Q4_1,
    "Q4_1",
    "q4_1",
    0xB411,
    run_indexed_moe_mmq_q4_1_gate_up_tile8_sweep,
    run_indexed_moe_mmq_q4_1_down_tile8_sweep
);
tile8_sweeps!(
    Tile8Wk::Q5_0,
    "Q5_0",
    "q5_0",
    0xB501,
    run_indexed_moe_mmq_q5_0_gate_up_tile8_sweep,
    run_indexed_moe_mmq_q5_0_down_tile8_sweep
);
tile8_sweeps!(
    Tile8Wk::Q5_1,
    "Q5_1",
    "q5_1",
    0xB511,
    run_indexed_moe_mmq_q5_1_gate_up_tile8_sweep,
    run_indexed_moe_mmq_q5_1_down_tile8_sweep
);
tile8_sweeps!(
    Tile8Wk::Q4K,
    "Q4_K",
    "q4_k",
    0xB4C0,
    run_indexed_moe_mmq_q4_k_gate_up_tile8_sweep,
    run_indexed_moe_mmq_q4_k_down_tile8_sweep
);
tile8_sweeps!(
    Tile8Wk::Q5K,
    "Q5_K",
    "q5_k",
    0xB5C0,
    run_indexed_moe_mmq_q5_k_gate_up_tile8_sweep,
    run_indexed_moe_mmq_q5_k_down_tile8_sweep
);
tile8_sweeps!(
    Tile8Wk::Q6K,
    "Q6_K",
    "q6_k",
    0xB6C0,
    run_indexed_moe_mmq_q6_k_gate_up_tile8_sweep,
    run_indexed_moe_mmq_q6_k_down_tile8_sweep
);
tile8_sweeps!(
    Tile8Wk::Q2K,
    "Q2_K",
    "q2_k",
    0xB2C0,
    run_indexed_moe_mmq_q2_k_gate_up_tile8_sweep,
    run_indexed_moe_mmq_q2_k_down_tile8_sweep
);
tile8_sweeps!(
    Tile8Wk::Q3K,
    "Q3_K",
    "q3_k",
    0xB3C0,
    run_indexed_moe_mmq_q3_k_gate_up_tile8_sweep,
    run_indexed_moe_mmq_q3_k_down_tile8_sweep
);
