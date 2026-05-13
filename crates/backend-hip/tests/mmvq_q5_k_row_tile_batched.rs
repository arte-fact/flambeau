//! Parity for `mmvq_q5_k_row_tile_batched` vs `mmvq_q5_k_r2_batched` (K3).
//! Both kernels consume identical Q5_K + Q8_1 inputs and must produce
//! identical outputs (mod F32 reduction-order noise).

#![expect(clippy::undocumented_unsafe_blocks, reason = "test fixture; same shape rationale as siblings")]
#![expect(clippy::cast_possible_wrap, reason = "kernel-shape math bounded by GGUF dims")]

use flambeau_backend_hip::{device_count, HipDevice, HipKernel, HipModule, KernelArgs, LaunchCfg};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;
use flambeau_quant::{BlockQ5K, BlockQ8_1, QK8_0, QK_K};
use half::f16;

const QK8: usize = QK8_0;

fn maybe_skip() -> bool {
    matches!(device_count(), Ok(n) if n >= 1)
}

fn seeded_bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 24) as u8
        })
        .collect()
}

fn seeded_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (state >> 32) as u32;
            (u as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn random_q5k_block(seed: u64, idx: usize) -> BlockQ5K {
    let bytes = seeded_bytes(seed ^ ((idx as u64).wrapping_mul(0x9E3779B97F4A7C15)), 176);
    let d_scalar = (bytes[0] as f32 / 255.0) * 0.1 + 0.01;
    let dmin_scalar = (bytes[1] as f32 / 255.0) * 0.05;
    let d = f16::from_f32(d_scalar);
    let dmin = f16::from_f32(dmin_scalar);
    let mut scales = [0u8; 12];
    scales.copy_from_slice(&bytes[4..16]);
    let mut qh = [0u8; QK_K / 8];
    qh.copy_from_slice(&bytes[16..16 + QK_K / 8]);
    let mut qs = [0u8; QK_K / 2];
    qs.copy_from_slice(&bytes[48..48 + QK_K / 2]);
    BlockQ5K { d, dmin, scales, qh, qs }
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

fn copy_back(dev: &HipDevice, src: DevicePtr, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(out.as_mut_ptr() as usize),
            src,
            n * 4,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    out
}

struct Outs {
    k3: Vec<f32>,
    rt: Vec<f32>,
}

fn run_both(n_rows: usize, k: usize, n_slots: usize, seed: u64) -> Outs {
    assert!((2..=4).contains(&n_slots));
    assert_eq!(k % QK_K, 0, "k must be a multiple of QK_K (256)");
    let n_sb_per_row = k / QK_K;

    let dev = HipDevice::new(0).unwrap();
    dev.bind().unwrap();

    let q_module = HipModule::load(0, kernels::hsaco("quantize_q8_1").unwrap()).unwrap();
    let k3_module = HipModule::load(0, kernels::hsaco("mmvq_q5_k_r2_batched").unwrap()).unwrap();
    let rt_module = HipModule::load(0, kernels::hsaco("mmvq_q5_k_row_tile_batched").unwrap()).unwrap();
    let k_quantize: HipKernel<'_> = q_module.kernel("flambeau_quantize_row_q8_1").unwrap();
    let k3_entry = match n_slots {
        2 => "flambeau_mmvq_q5_k_r2_q8_1_batched_n2",
        3 => "flambeau_mmvq_q5_k_r2_q8_1_batched_n3",
        4 => "flambeau_mmvq_q5_k_r2_q8_1_batched_n4",
        _ => unreachable!(),
    };
    let rt_entry = match n_slots {
        2 => "flambeau_mmvq_q5_k_row_tile_q8_1_batched_n2",
        3 => "flambeau_mmvq_q5_k_row_tile_q8_1_batched_n3",
        4 => "flambeau_mmvq_q5_k_row_tile_q8_1_batched_n4",
        _ => unreachable!(),
    };
    let k_k3: HipKernel<'_> = k3_module.kernel(k3_entry).unwrap();
    let k_rt: HipKernel<'_> = rt_module.kernel(rt_entry).unwrap();

    let mut x_blocks: Vec<BlockQ5K> = Vec::with_capacity(n_rows * n_sb_per_row);
    for i in 0..(n_rows * n_sb_per_row) {
        x_blocks.push(random_q5k_block(seed, i));
    }
    let y_f32 = seeded_f32(seed.wrapping_add(31), k * n_slots);

    let d_x = alloc_and_upload(&dev, &x_blocks);
    let d_y_f32 = alloc_and_upload(&dev, &y_f32);
    let y_blocks_per_slot = k / QK8;
    let d_y_q8_1 = dev
        .alloc(n_slots * y_blocks_per_slot * std::mem::size_of::<BlockQ8_1>())
        .unwrap();
    let d_k3 = dev.alloc(n_slots * n_rows * 4).unwrap();
    let d_rt = dev.alloc(n_slots * n_rows * 4).unwrap();

    {
        let stream = dev.default_stream();
        for c in 0..n_slots {
            let n_elems = k as i32;
            let d_y_f32_c: u64 = (d_y_f32.as_usize() + c * k * 4) as u64;
            let d_y_q8_1_c: u64 =
                (d_y_q8_1.as_usize() + c * y_blocks_per_slot * std::mem::size_of::<BlockQ8_1>()) as u64;
            let mut args = KernelArgs::new();
            args.push(&d_y_f32_c);
            args.push(&d_y_q8_1_c);
            args.push(&n_elems);
            let cfg = LaunchCfg::one_d(y_blocks_per_slot as u32, QK8 as u32);
            unsafe { k_quantize.launch(stream, cfg, args).unwrap() };
        }
        stream.synchronize().unwrap();
    }

    let n_rows_i = n_rows as i32;
    let n_sb_i = n_sb_per_row as i32;
    let d_x_ptr: u64 = d_x.as_usize() as u64;
    let d_y_q8_1_ptr: u64 = d_y_q8_1.as_usize() as u64;

    {
        let stream = dev.default_stream();
        let dst: u64 = d_k3.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_x_ptr);
        args.push(&d_y_q8_1_ptr);
        args.push(&dst);
        args.push(&n_rows_i);
        args.push(&n_sb_i);
        let grid = ((n_rows + 1) / 2) as u32;
        let cfg = LaunchCfg::one_d(grid, 64);
        unsafe { k_k3.launch(stream, cfg, args).unwrap() };
        stream.synchronize().unwrap();
    }

    {
        let stream = dev.default_stream();
        let dst: u64 = d_rt.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_x_ptr);
        args.push(&d_y_q8_1_ptr);
        args.push(&dst);
        args.push(&n_rows_i);
        args.push(&n_sb_i);
        let grid = (n_rows as u32).div_ceil(8);
        let cfg = LaunchCfg::one_d(grid, 256);
        unsafe { k_rt.launch(stream, cfg, args).unwrap() };
        stream.synchronize().unwrap();
    }

    let k3 = copy_back(&dev, d_k3, n_slots * n_rows);
    let rt = copy_back(&dev, d_rt, n_slots * n_rows);

    unsafe {
        dev.dealloc(d_x, x_blocks.len() * std::mem::size_of::<BlockQ5K>()).unwrap();
        dev.dealloc(d_y_f32, y_f32.len() * 4).unwrap();
        dev.dealloc(d_y_q8_1, n_slots * y_blocks_per_slot * std::mem::size_of::<BlockQ8_1>()).unwrap();
        dev.dealloc(d_k3, n_slots * n_rows * 4).unwrap();
        dev.dealloc(d_rt, n_slots * n_rows * 4).unwrap();
    }

    Outs { k3, rt }
}

fn assert_close(label: &str, a: &[f32], b: &[f32], k: usize) {
    assert_eq!(a.len(), b.len());
    let mut worst = 0.0f32;
    let mut worst_idx = 0usize;
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let e = (x - y).abs() / x.abs().max(y.abs()).max(1e-3);
        if e > worst {
            worst = e;
            worst_idx = i;
        }
    }
    let tol = 5e-4 * (k as f32 / 128.0).sqrt();
    eprintln!(
        "[{label}] max_rel_err={worst:.3e} at idx={worst_idx} (k3={}, rt={}), tol={tol:.3e}",
        a[worst_idx], b[worst_idx]
    );
    assert!(worst <= tol, "{label}: rel_err {worst:.3e} > {tol:.3e}");
}

#[test]
fn parity_q5k_n2_k256_small() {
    if !maybe_skip() {
        return;
    }
    let outs = run_both(8, 256, 2, 0xC0FFEE);
    assert_close("q5k n2 k256 r8", &outs.k3, &outs.rt, 256);
}

#[test]
fn parity_q5k_n4_k2048() {
    if !maybe_skip() {
        return;
    }
    let outs = run_both(64, 2048, 4, 0x1234_5678);
    assert_close("q5k n4 k2048 r64", &outs.k3, &outs.rt, 2048);
}

#[test]
fn parity_q5k_n3_k4096() {
    if !maybe_skip() {
        return;
    }
    let outs = run_both(40, 4096, 3, 0xFEED_FACE);
    assert_close("q5k n3 k4096 r40", &outs.k3, &outs.rt, 4096);
}

#[test]
fn parity_q5k_n2_rows_non_multiple_8() {
    if !maybe_skip() {
        return;
    }
    // n_rows=13 → row-tile has 1 full block + a partial block (5 rows).
    let outs = run_both(13, 1024, 2, 0xABCDEF);
    assert_close("q5k n2 k1024 r13(tail)", &outs.k3, &outs.rt, 1024);
}
