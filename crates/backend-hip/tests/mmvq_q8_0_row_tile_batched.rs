//! Parity for `mmvq_q8_0_row_tile_batched` vs `mmvq_q8_0_batched`.
//! Both kernels consume identical Q8_0 + Q8_1 inputs and must produce
//! identical outputs (mod F32 reduction-order noise) for all
//! (n_rows, k, N) shapes we ever launch.

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture; same shape rationale as siblings"
)]
#![expect(
    clippy::cast_possible_wrap,
    reason = "kernel-shape math bounded by GGUF dims"
)]

use flambeau_backend_hip::{device_count, HipDevice, HipKernel, HipModule, KernelArgs, LaunchCfg};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;
use flambeau_quant::{BlockQ8_0, BlockQ8_1, QK8_0};
use half::f16;

const QK8: usize = QK8_0;

fn maybe_skip() -> bool {
    match device_count() {
        Ok(n) if n >= 1 => true,
        _ => {
            eprintln!("[skip] no HIP device");
            false
        }
    }
}

fn seeded_bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 24) as u8
        })
        .collect()
}

fn seeded_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (state >> 32) as u32;
            (u as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn random_q8_0_block(seed: u64, idx: usize) -> BlockQ8_0 {
    let bytes = seeded_bytes(
        seed ^ ((idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
        QK8 + 4,
    );
    let d_scalar = (bytes[0] as f32 / 255.0) * 0.1 + 0.01;
    let d = f16::from_f32(d_scalar);
    let mut qs = [0i8; QK8];
    for (i, &b) in bytes[4..4 + QK8].iter().enumerate() {
        qs[i] = b as i8;
    }
    BlockQ8_0 { d, qs }
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
    baseline: Vec<f32>,
    row_tile: Vec<f32>,
}

fn run_both(n_rows: usize, k: usize, n_slots: usize, seed: u64) -> Outs {
    assert!((2..=4).contains(&n_slots), "n_slots ∈ [2, 4]");
    assert_eq!(k % QK8, 0, "k must be a multiple of Q8_0 block size (32)");
    let n_blocks_per_row = k / QK8;

    let dev = HipDevice::new(0).unwrap();
    dev.bind().unwrap();

    let q_bytes = kernels::hsaco("quantize_q8_1").unwrap();
    let base_bytes = kernels::hsaco("mmvq_q8_0_batched").unwrap();
    let rt_bytes = kernels::hsaco("mmvq_q8_0_row_tile_batched").unwrap();
    let q_module = HipModule::load(0, q_bytes).unwrap();
    let base_module = HipModule::load(0, base_bytes).unwrap();
    let rt_module = HipModule::load(0, rt_bytes).unwrap();
    let k_quantize: HipKernel<'_> = q_module.kernel("flambeau_quantize_row_q8_1").unwrap();
    let base_entry = match n_slots {
        2 => "flambeau_mmvq_q8_0_q8_1_batched_n2",
        3 => "flambeau_mmvq_q8_0_q8_1_batched_n3",
        4 => "flambeau_mmvq_q8_0_q8_1_batched_n4",
        _ => unreachable!(),
    };
    let rt_entry = match n_slots {
        2 => "flambeau_mmvq_q8_0_row_tile_dp4a_q8_1_batched_n2",
        3 => "flambeau_mmvq_q8_0_row_tile_dp4a_q8_1_batched_n3",
        4 => "flambeau_mmvq_q8_0_row_tile_dp4a_q8_1_batched_n4",
        _ => unreachable!(),
    };
    let k_base: HipKernel<'_> = base_module.kernel(base_entry).unwrap();
    let k_rt: HipKernel<'_> = rt_module.kernel(rt_entry).unwrap();

    let mut w_blocks: Vec<BlockQ8_0> = Vec::with_capacity(n_rows * n_blocks_per_row);
    for i in 0..(n_rows * n_blocks_per_row) {
        w_blocks.push(random_q8_0_block(seed, i));
    }

    let y_f32 = seeded_f32(seed.wrapping_add(31), k * n_slots);

    let d_w = alloc_and_upload(&dev, &w_blocks);
    let d_y_f32 = alloc_and_upload(&dev, &y_f32);
    let y_blocks_per_slot = k / QK8;
    let d_y_q8_1 = dev
        .alloc(n_slots * y_blocks_per_slot * std::mem::size_of::<BlockQ8_1>())
        .unwrap();
    let d_base = dev.alloc(n_slots * n_rows * 4).unwrap();
    let d_rt = dev.alloc(n_slots * n_rows * 4).unwrap();

    {
        let stream = dev.default_stream();
        for c in 0..n_slots {
            let n_elems = k as i32;
            let d_y_f32_c: u64 = (d_y_f32.as_usize() + c * k * 4) as u64;
            let d_y_q8_1_c: u64 = (d_y_q8_1.as_usize()
                + c * y_blocks_per_slot * std::mem::size_of::<BlockQ8_1>())
                as u64;
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
    let n_blocks_i = n_blocks_per_row as i32;
    let d_w_ptr: u64 = d_w.as_usize() as u64;
    let d_y_q8_1_ptr: u64 = d_y_q8_1.as_usize() as u64;

    {
        let stream = dev.default_stream();
        let out: u64 = d_base.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_w_ptr);
        args.push(&d_y_q8_1_ptr);
        args.push(&out);
        args.push(&n_rows_i);
        args.push(&n_blocks_i);
        let cfg = LaunchCfg::one_d(n_rows as u32, 256);
        unsafe { k_base.launch(stream, cfg, args).unwrap() };
        stream.synchronize().unwrap();
    }

    {
        let stream = dev.default_stream();
        let out: u64 = d_rt.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_w_ptr);
        args.push(&d_y_q8_1_ptr);
        args.push(&out);
        args.push(&n_rows_i);
        args.push(&n_blocks_i);
        let grid = (n_rows as u32).div_ceil(4);
        let cfg = LaunchCfg::one_d(grid, 256);
        unsafe { k_rt.launch(stream, cfg, args).unwrap() };
        stream.synchronize().unwrap();
    }

    let baseline = copy_back(&dev, d_base, n_slots * n_rows);
    let row_tile = copy_back(&dev, d_rt, n_slots * n_rows);

    unsafe {
        dev.dealloc(d_w, w_blocks.len() * std::mem::size_of::<BlockQ8_0>())
            .unwrap();
        dev.dealloc(d_y_f32, y_f32.len() * 4).unwrap();
        dev.dealloc(
            d_y_q8_1,
            n_slots * y_blocks_per_slot * std::mem::size_of::<BlockQ8_1>(),
        )
        .unwrap();
        dev.dealloc(d_base, n_slots * n_rows * 4).unwrap();
        dev.dealloc(d_rt, n_slots * n_rows * 4).unwrap();
    }

    Outs { baseline, row_tile }
}

fn assert_close(label: &str, base: &[f32], rt: &[f32], k: usize) {
    assert_eq!(base.len(), rt.len(), "{label}: length mismatch");
    let mut worst_rel = 0.0f32;
    let mut worst_abs = 0.0f32;
    let mut worst_idx = 0usize;
    for (i, (&a, &b)) in base.iter().zip(rt).enumerate() {
        let abs = (a - b).abs();
        let rel = abs / a.abs().max(b.abs()).max(1e-3);
        if rel > worst_rel {
            worst_rel = rel;
            worst_abs = abs;
            worst_idx = i;
        }
    }
    // Q8_0 outputs can near-cancel (signed i8 × signed i8 products both ways)
    // so a single ULP at the F32 sum scale shows up as a large relative
    // error on tiny outputs. Accept either rel or abs tolerance.
    let rel_tol = 1e-3 * (k as f32 / 128.0).sqrt();
    let abs_tol = 1e-4 * (k as f32 / 128.0).sqrt();
    eprintln!(
        "[{label}] worst rel={worst_rel:.3e} abs={worst_abs:.3e} at idx={worst_idx} \
         (base={}, rt={}), rel_tol={rel_tol:.3e} abs_tol={abs_tol:.3e}",
        base[worst_idx], rt[worst_idx]
    );
    assert!(
        worst_rel <= rel_tol || worst_abs <= abs_tol,
        "{label}: rel={worst_rel:.3e} > {rel_tol:.3e} AND abs={worst_abs:.3e} > {abs_tol:.3e}"
    );
}

#[test]
fn parity_n2_k512_small() {
    if !maybe_skip() {
        return;
    }
    let outs = run_both(8, 512, 2, 0x00C0_FFEE_BEEF);
    assert_close("n2 k512 n_rows=8", &outs.baseline, &outs.row_tile, 512);
}

#[test]
fn parity_n4_k2304_medium() {
    if !maybe_skip() {
        return;
    }
    let outs = run_both(64, 2304, 4, 0x1234_5678);
    assert_close("n4 k2304 n_rows=64", &outs.baseline, &outs.row_tile, 2304);
}

#[test]
fn parity_n3_k4096_asym_rows() {
    if !maybe_skip() {
        return;
    }
    let outs = run_both(37, 4096, 3, 0xFEED_FACE);
    assert_close("n3 k4096 n_rows=37", &outs.baseline, &outs.row_tile, 4096);
}

#[test]
fn parity_n4_k256_single_outer_iter() {
    if !maybe_skip() {
        return;
    }
    // K=256 → n_blocks_per_row=8 = exactly 1 outer iter.
    let outs = run_both(16, 256, 4, 0xFADE_BEEF);
    assert_close("n4 k256 n_rows=16", &outs.baseline, &outs.row_tile, 256);
}

#[test]
fn parity_n4_k2048_gdn_alpha_beta_shape() {
    if !maybe_skip() {
        return;
    }
    // Qwen3.6-27B-Q4_0 / pp2tp2 GDN α/β shape (per-rank slice):
    // n_rows = local_d_inner = 2048-4096 per rank, k = hidden ≈ 2048.
    let outs = run_both(2048, 2048, 4, 0x27B_BEEF);
    assert_close(
        "n4 alpha_beta_2048x2048",
        &outs.baseline,
        &outs.row_tile,
        2048,
    );
}
