//! V1.5 MoE kernel certs — TopK router, IndexedMoE MMVQ Q4_K, MoE combine.
//!
//! Three targeted sweeps, each emitting its own cert. Shapes are chosen to
//! exercise Qwen3.6's MoE regime: 128 experts, top-8 routing, head-shaped
//! hidden dims. No delta-ppl quality cert — that's a V1.7 model-loader
//! deliverable.

#![cfg(feature = "hip")]

use std::path::Path;

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::{
    device_count, FuncAttributes, HipDevice, HipKernel, HipModule, KernelArgs, LaunchCfg,
};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;
use flambeau_quant::{BlockQ4K, BlockQ6K, BlockQ8_1, QK8_0, QK_K};
use half::f16;

use crate::cert::{now_utc_iso8601, Cert, PmcSnapshot, ShapeResult, SCHEMA_VERSION};

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
    // Cover both n_experts values at varying batch sizes. V1.7.4.a (fixed
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
        let max_rel = if idx_mismatch > 0 {
            1.0
        } else {
            wts_err
        };
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
        tolerance_formula:
            "indices match exactly; softmax weights within 1e-4 abs".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_tag(),
        pmc: Some(pmc_from(&attrs)),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

fn cpu_topk(
    logits: &[f32],
    n_tokens: usize,
    n_experts: usize,
    k: usize,
) -> (Vec<i32>, Vec<f32>) {
    let mut idxs = vec![0i32; n_tokens * k];
    let mut wts = vec![0f32; n_tokens * k];
    for t in 0..n_tokens {
        let row = &logits[t * n_experts..(t + 1) * n_experts];
        let mut pairs: Vec<(f32, i32)> =
            row.iter().enumerate().map(|(i, &v)| (v, i as i32)).collect();
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
        dev.memcpy_async(stream, CopyDirection::DeviceToHost, DevicePtr(out_i.as_mut_ptr() as usize), d_i, n_tokens * k * 4)?;
        dev.memcpy_async(stream, CopyDirection::DeviceToHost, DevicePtr(out_w.as_mut_ptr() as usize), d_w, n_tokens * k * 4)?;
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
    //   n_experts = 16 (of 128)
    //   n_rows    = 256  (expert output dim / 10)
    //   k         = 2048 (hidden)
    //   top_k     = 4 (of 8)
    //   n_tokens  = 1, 8
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
        tolerance_formula:
            "|err| <= 5e-2 * max(|ref|, sqrt(k))  (MoE MMVQ — same envelope as V1.3 MMVQ)".to_string(),
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
    // the V1.3 sweep).
    let total_blocks = n_experts * n_rows * nb_per_row;
    let w_bytes = total_blocks * std::mem::size_of::<BlockQ4K>();
    let w_raw = tame_q4k_scales(seeded_bytes(seed, w_bytes));
    let w_blocks: &[BlockQ4K] = bytemuck::cast_slice(&w_raw);

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
    let kernel: HipKernel<'_> =
        module.kernel("flambeau_indexed_moe_mmvq_q4_k_r2_q8_1")?;
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
        tolerance_formula:
            "|err| <= 5e-2 * max(|ref|, sqrt(k))  (same envelope as single-row)".to_string(),
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
        let grid_x = ((n_rows as u32) + 1) / 2;
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
// Q6_K IndexedMoE MMVQ cert (V1.7.5.K — UD-Q4_K_S mixed-quant down_exps)
// ---------------------------------------------------------------------------

pub fn run_indexed_moe_mmvq_q6_k_sweep(repo_root: &Path) -> Result<Cert> {
    let dev = ensure_dev()?;
    let kb = kernels::hsaco("indexed_moe_mmvq_q6_k").unwrap();
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> =
        module.kernel("flambeau_indexed_moe_mmvq_q6_k_q8_1")?;
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
        let seed = 0xDEC0DE
            ^ (n_tokens as u64 * 53 + top_k as u64 * 17)
            ^ 0xABCD_u64; // Q6_K-specific spice so seeds differ from Q4_K sweeps
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
            "|err| <= 5e-2 * max(|ref|, sqrt(k))  (Q6_K weights; same envelope as Q4_K MoE MMVQ)".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_tag(),
        pmc: Some(pmc_from(&attrs)),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
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
    let kernel: HipKernel<'_> =
        module.kernel("flambeau_indexed_moe_mmvq_q4_k_gate_up_q8_1")?;
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
    flambeau_quant::dequantize_into(
        flambeau_quant::GgmlDType::Q4K,
        &gate_raw,
        &mut gate_dequant,
    )?;
    flambeau_quant::dequantize_into(
        flambeau_quant::GgmlDType::Q4K,
        &up_raw,
        &mut up_dequant,
    )?;
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
// IndexedMoE MMQ Q4_K cert (V1.5.7 — 4-warp LDS-tiled prefill)
// ---------------------------------------------------------------------------
//
// MMQ lives in the prefill regime: many (token, slot) pairs processed together.
// The caller (CPU) sorts (token, slot) pairs into per-expert buckets of
// MMQ_X=8 slots so that each block shares ONE expert and can amortise the
// weight tile across all slots. Sentinel -1 marks an unfilled tail slot.

const MMQ_Y: usize = 16;
const MMQ_X: usize = 8;

/// Group (token, slot) pairs into per-expert buckets of size MMQ_X. Returns
/// `(bucket_expert, bucket_slots)` where `bucket_slots[i]` is a flat row of
/// MMQ_X refs (`token << 16 | slot` or `-1` for padding).
fn build_expert_buckets(
    expert_ids: &[i32],
    n_tokens: usize,
    top_k: usize,
) -> (Vec<i32>, Vec<i32>) {
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

    // Prefill regime: 128 tokens × 8 slots = 1024 work items. Matches V1.4
    // MMQ cert shape band. n_rows=256, k=2048 stays comparable to the MMVQ
    // certs; we also run a tall-K case for amortisation sanity.
    let cases = [
        (128usize, 8usize, 256usize, 2048usize),
        (512, 8, 256, 2048),
    ];
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
        tolerance_formula:
            "|err| <= 5e-2 * max(|ref|, sqrt(k))  (MoE MMQ — same envelope as V1.3 MMVQ)".to_string(),
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

    let (bucket_expert, bucket_slots) =
        build_expert_buckets(&expert_ids, n_tokens, top_k);
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
        let grid_x = ((n_rows as u32) + MMQ_Y as u32 - 1) / MMQ_Y as u32;
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
        let cfg = LaunchCfg::one_d(((total + 255) / 256) as u32, 256);
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
    format!("{}-gfx906", hostname().unwrap_or_else(|| "unknown".into()))
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
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (s >> 32) as u32;
            ((u as f32 / u32::MAX as f32) - 0.5)
        })
        .collect()
}

fn seeded_bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 24) as u8
        })
        .collect()
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

fn max_rel_err(got: &[f32], reference: &[f32]) -> f32 {
    got.iter()
        .zip(reference)
        .map(|(g, r)| (g - r).abs() / r.abs().max(1.0))
        .fold(0.0f32, f32::max)
}

fn max_rel_err_with_floor(got: &[f32], reference: &[f32], k: usize) -> f32 {
    let abs_floor = (k as f32).sqrt();
    got.iter()
        .zip(reference)
        .map(|(g, r)| (g - r).abs() / r.abs().max(abs_floor))
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
