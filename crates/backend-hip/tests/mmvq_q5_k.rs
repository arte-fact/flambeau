//! Q5_K MMVQ correctness cert on real MI50.
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture — every `unsafe {}` below is a kernel launch or `memcpy_async`               whose invariant is uniform: host/device buffers live for the bounded               `synchronize()` that follows, pointers are freshly allocated above, kernel               ABIs match kernels-hip. Per-site SAFETY comments would just repeat this."
)]

mod common;

use common::*;
use flambeau_backend_hip::{HipDevice, HipKernel, HipModule, KernelArgs, LaunchCfg};
use flambeau_core::{Device, Stream};
use flambeau_kernels_hip as kernels;
use flambeau_quant::{BlockQ5K, QK_K};
use half::f16;

fn random_q5k_block(seed: u64, idx: usize) -> BlockQ5K {
    let bytes = seeded_bytes(seed ^ ((idx as u64).wrapping_mul(0x9E3779B97F4A7C15)), 176);
    let d = f16::from_f32((bytes[0] as f32 / 255.0) * 0.1 + 0.01);
    let dmin = f16::from_f32((bytes[1] as f32 / 255.0) * 0.05);
    let mut scales = [0u8; 12];
    scales.copy_from_slice(&bytes[4..16]);
    let mut qh = [0u8; QK_K / 8];
    qh.copy_from_slice(&bytes[16..16 + QK_K / 8]);
    let mut qs = [0u8; QK_K / 2];
    qs.copy_from_slice(&bytes[16 + QK_K / 8..16 + QK_K / 8 + QK_K / 2]);
    BlockQ5K {
        d,
        dmin,
        scales,
        qh,
        qs,
    }
}

fn run(n_rows: usize, k: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(k % QK_K, 0);
    let sb = k / QK_K;

    let dev = HipDevice::new(0).unwrap();
    dev.bind().unwrap();

    let m_bytes = kernels::hsaco("mmvq_q5_k").unwrap();
    let m_module = HipModule::load(0, m_bytes).unwrap();
    let k_mmvq: HipKernel<'_> = m_module.kernel("flambeau_mmvq_q5_k_q8_1").unwrap();

    let mut x_blocks: Vec<BlockQ5K> = (0..n_rows * sb)
        .map(|i| random_q5k_block(seed, i))
        .collect();
    let _ = &mut x_blocks; // shut up unused_mut
    let total_elems = n_rows * k;
    let mut x_dequant = vec![0.0f32; total_elems];
    {
        let raw: &[u8] = bytemuck::cast_slice(&x_blocks);
        flambeau_quant::dequantize_into(flambeau_quant::GgmlDType::Q5K, raw, &mut x_dequant)
            .unwrap();
    }

    let y_f32 = seeded_f32(seed.wrapping_add(37), k);

    let d_x = alloc_and_upload(&dev, &x_blocks);
    let d_y_f32 = alloc_and_upload(&dev, &y_f32);
    let (d_y_q8_1, y_q8_1_bytes) = quantize_q8_1_on_device(&dev, d_y_f32, k);
    let d_dst = dev.alloc(n_rows * 4).unwrap();

    {
        let stream = dev.default_stream();
        let n_rows_i = n_rows as i32;
        let n_sb_i = sb as i32;
        let d_x_ptr: u64 = d_x.as_usize() as u64;
        let d_y_q8_1_ptr: u64 = d_y_q8_1.as_usize() as u64;
        let d_dst_ptr: u64 = d_dst.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_x_ptr);
        args.push(&d_y_q8_1_ptr);
        args.push(&d_dst_ptr);
        args.push(&n_rows_i);
        args.push(&n_sb_i);
        let cfg = LaunchCfg::one_d(n_rows as u32, 64);
        unsafe { k_mmvq.launch(stream, cfg, args).unwrap() };
        stream.synchronize().unwrap();
    }

    let dst = download_f32(&dev, d_dst, n_rows);

    unsafe {
        dev.dealloc(d_x, x_blocks.len() * std::mem::size_of::<BlockQ5K>())
            .unwrap();
        dev.dealloc(d_y_f32, y_f32.len() * 4).unwrap();
        dev.dealloc(d_y_q8_1, y_q8_1_bytes).unwrap();
        dev.dealloc(d_dst, n_rows * 4).unwrap();
    }

    let y_rt = quantize_q8_1_roundtrip(&y_f32);
    let reference = reference_matmul(&x_dequant, &y_rt, n_rows, k);
    (dst, reference)
}

#[test]
fn mmvq_q5_k_small() {
    if !maybe_skip() {
        return;
    }
    let (got, reference) = run(4, QK_K, 0xC0FFEE);
    let err = max_rel_err(&got, &reference);
    let tol = cert_tol(QK_K);
    eprintln!("[mmvq_q5_k 4×{QK}] err={err:.3e}, tol={tol:.3e}");
    assert!(err <= tol);
}

#[test]
fn mmvq_q5_k_qwen_2048() {
    if !maybe_skip() {
        return;
    }
    let k = 2048;
    let (got, reference) = run(8, k, 0xFEEDFACE);
    let err = max_rel_err(&got, &reference);
    let tol = cert_tol(k);
    eprintln!("[mmvq_q5_k 8×{k}] err={err:.3e}, tol={tol:.3e}");
    assert!(err <= tol);
}

#[test]
fn mmvq_q5_k_5120() {
    if !maybe_skip() {
        return;
    }
    let k = 5120;
    let (got, reference) = run(16, k, 0x12345678);
    let err = max_rel_err(&got, &reference);
    let tol = cert_tol(k);
    eprintln!("[mmvq_q5_k 16×{k}] err={err:.3e}, tol={tol:.3e}");
    assert!(err <= tol);
}

#[test]
fn mmvq_q5_k_15360() {
    if !maybe_skip() {
        return;
    }
    let k = 15360;
    let (got, reference) = run(4, k, 0xABCDEF01);
    let err = max_rel_err(&got, &reference);
    let tol = cert_tol(k);
    eprintln!("[mmvq_q5_k 4×{k}] err={err:.3e}, tol={tol:.3e}");
    assert!(err <= tol);
}
