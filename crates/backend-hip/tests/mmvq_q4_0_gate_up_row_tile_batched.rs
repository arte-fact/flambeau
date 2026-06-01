//! Parity for `mmvq_q4_0_gate_up_row_tile_batched` vs the existing K5
//! `mmvq_q4_0_gate_up_batched`. Both kernels consume identical
//! Q4_0 + Q8_1 inputs and must produce identical outputs (mod F32
//! reduction-order noise) for all (n_rows_gate, n_rows_up, k, N) shapes
//! we ever launch.

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
use flambeau_quant::{BlockQ4_0, BlockQ8_1, QK8_0};
use half::f16;

const QK8: usize = QK8_0;
const QK4: usize = 32;

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

fn random_q4_0_block(seed: u64, idx: usize) -> BlockQ4_0 {
    let bytes = seeded_bytes(
        seed ^ ((idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
        QK4 / 2 + 4,
    );
    let d_scalar = (bytes[0] as f32 / 255.0) * 0.1 + 0.01;
    let d = f16::from_f32(d_scalar);
    let mut qs = [0u8; QK4 / 2];
    qs.copy_from_slice(&bytes[4..4 + QK4 / 2]);
    BlockQ4_0 { d, qs }
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
    k5_gate: Vec<f32>,
    k5_up: Vec<f32>,
    rt_gate: Vec<f32>,
    rt_up: Vec<f32>,
}

fn run_both(n_rows_gate: usize, n_rows_up: usize, k: usize, n_slots: usize, seed: u64) -> Outs {
    assert!((2..=4).contains(&n_slots), "n_slots ∈ [2, 4]");
    assert_eq!(k % QK4, 0, "k must be a multiple of Q4_0 block size (32)");
    let n_blocks_per_row = k / QK4;

    let dev = HipDevice::new(0).unwrap();
    dev.bind().unwrap();

    let q_bytes = kernels::hsaco("quantize_q8_1").unwrap();
    let k5_bytes = kernels::hsaco("mmvq_q4_0_gate_up_batched").unwrap();
    let rt_bytes = kernels::hsaco("mmvq_q4_0_gate_up_row_tile_batched").unwrap();
    let q_module = HipModule::load(0, q_bytes).unwrap();
    let k5_module = HipModule::load(0, k5_bytes).unwrap();
    let rt_module = HipModule::load(0, rt_bytes).unwrap();
    let k_quantize: HipKernel<'_> = q_module.kernel("flambeau_quantize_row_q8_1").unwrap();
    let k5_entry = match n_slots {
        2 => "flambeau_mmvq_q4_0_gate_up_dp4a_q8_1_batched_n2",
        3 => "flambeau_mmvq_q4_0_gate_up_dp4a_q8_1_batched_n3",
        4 => "flambeau_mmvq_q4_0_gate_up_dp4a_q8_1_batched_n4",
        _ => unreachable!(),
    };
    let rt_entry = match n_slots {
        2 => "flambeau_mmvq_q4_0_gate_up_row_tile_dp4a_q8_1_batched_n2",
        3 => "flambeau_mmvq_q4_0_gate_up_row_tile_dp4a_q8_1_batched_n3",
        4 => "flambeau_mmvq_q4_0_gate_up_row_tile_dp4a_q8_1_batched_n4",
        _ => unreachable!(),
    };
    let k_k5: HipKernel<'_> = k5_module.kernel(k5_entry).unwrap();
    let k_rt: HipKernel<'_> = rt_module.kernel(rt_entry).unwrap();

    let mut g_blocks: Vec<BlockQ4_0> = Vec::with_capacity(n_rows_gate * n_blocks_per_row);
    for i in 0..(n_rows_gate * n_blocks_per_row) {
        g_blocks.push(random_q4_0_block(seed, i));
    }
    let mut u_blocks: Vec<BlockQ4_0> = Vec::with_capacity(n_rows_up * n_blocks_per_row);
    for i in 0..(n_rows_up * n_blocks_per_row) {
        u_blocks.push(random_q4_0_block(seed.wrapping_add(17), i));
    }

    let y_f32 = seeded_f32(seed.wrapping_add(31), k * n_slots);

    let d_g = alloc_and_upload(&dev, &g_blocks);
    let d_u = alloc_and_upload(&dev, &u_blocks);
    let d_y_f32 = alloc_and_upload(&dev, &y_f32);
    let y_blocks_per_slot = k / QK8;
    let d_y_q8_1 = dev
        .alloc(n_slots * y_blocks_per_slot * std::mem::size_of::<BlockQ8_1>())
        .unwrap();
    let d_k5_g = dev.alloc(n_slots * n_rows_gate * 4).unwrap();
    let d_k5_u = dev.alloc(n_slots * n_rows_up * 4).unwrap();
    let d_rt_g = dev.alloc(n_slots * n_rows_gate * 4).unwrap();
    let d_rt_u = dev.alloc(n_slots * n_rows_up * 4).unwrap();

    // Quantise all N slots' activations. Each slot is k long; emit one
    // Q8_1 sub-tensor per slot back-to-back.
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

    let n_rows_g_i = n_rows_gate as i32;
    let n_rows_u_i = n_rows_up as i32;
    let n_blocks_i = n_blocks_per_row as i32;
    let d_g_ptr: u64 = d_g.as_usize() as u64;
    let d_u_ptr: u64 = d_u.as_usize() as u64;
    let d_y_q8_1_ptr: u64 = d_y_q8_1.as_usize() as u64;

    // K5
    {
        let stream = dev.default_stream();
        let g_out: u64 = d_k5_g.as_usize() as u64;
        let u_out: u64 = d_k5_u.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_g_ptr);
        args.push(&d_u_ptr);
        args.push(&d_y_q8_1_ptr);
        args.push(&g_out);
        args.push(&u_out);
        args.push(&n_rows_g_i);
        args.push(&n_rows_u_i);
        args.push(&n_blocks_i);
        let grid = n_rows_gate.max(n_rows_up) as u32;
        let cfg = LaunchCfg::one_d(grid, 256);
        unsafe { k_k5.launch(stream, cfg, args).unwrap() };
        stream.synchronize().unwrap();
    }

    // Row-tile
    {
        let stream = dev.default_stream();
        let g_out: u64 = d_rt_g.as_usize() as u64;
        let u_out: u64 = d_rt_u.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_g_ptr);
        args.push(&d_u_ptr);
        args.push(&d_y_q8_1_ptr);
        args.push(&g_out);
        args.push(&u_out);
        args.push(&n_rows_g_i);
        args.push(&n_rows_u_i);
        args.push(&n_blocks_i);
        let grid = (n_rows_gate.max(n_rows_up) as u32).div_ceil(4);
        let cfg = LaunchCfg::one_d(grid, 256);
        unsafe { k_rt.launch(stream, cfg, args).unwrap() };
        stream.synchronize().unwrap();
    }

    let k5_gate = copy_back(&dev, d_k5_g, n_slots * n_rows_gate);
    let k5_up = copy_back(&dev, d_k5_u, n_slots * n_rows_up);
    let rt_gate = copy_back(&dev, d_rt_g, n_slots * n_rows_gate);
    let rt_up = copy_back(&dev, d_rt_u, n_slots * n_rows_up);

    unsafe {
        dev.dealloc(d_g, g_blocks.len() * std::mem::size_of::<BlockQ4_0>())
            .unwrap();
        dev.dealloc(d_u, u_blocks.len() * std::mem::size_of::<BlockQ4_0>())
            .unwrap();
        dev.dealloc(d_y_f32, y_f32.len() * 4).unwrap();
        dev.dealloc(
            d_y_q8_1,
            n_slots * y_blocks_per_slot * std::mem::size_of::<BlockQ8_1>(),
        )
        .unwrap();
        dev.dealloc(d_k5_g, n_slots * n_rows_gate * 4).unwrap();
        dev.dealloc(d_k5_u, n_slots * n_rows_up * 4).unwrap();
        dev.dealloc(d_rt_g, n_slots * n_rows_gate * 4).unwrap();
        dev.dealloc(d_rt_u, n_slots * n_rows_up * 4).unwrap();
    }

    Outs {
        k5_gate,
        k5_up,
        rt_gate,
        rt_up,
    }
}

fn assert_bit_equal_or_close(label: &str, k5: &[f32], rt: &[f32], k: usize) {
    assert_eq!(k5.len(), rt.len(), "{label}: length mismatch");
    let mut worst = 0.0f32;
    let mut worst_idx = 0usize;
    for (i, (&a, &b)) in k5.iter().zip(rt).enumerate() {
        let e = (a - b).abs() / a.abs().max(b.abs()).max(1e-3);
        if e > worst {
            worst = e;
            worst_idx = i;
        }
    }
    // K5 vs row-tile differ only in F32 accumulation order. Tolerance
    // scales with sqrt(K) to track expected reduction noise.
    let tol = 5e-4 * (k as f32 / 128.0).sqrt();
    eprintln!(
        "[{label}] max_rel_err={worst:.3e} at idx={worst_idx} (k5={}, rt={}), tol={tol:.3e}",
        k5[worst_idx], rt[worst_idx]
    );
    assert!(worst <= tol, "{label}: rel_err {worst:.3e} > {tol:.3e}");
}

#[test]
fn parity_n2_k2304_8x8() {
    if !maybe_skip() {
        return;
    }
    let outs = run_both(8, 8, 2304, 2, 0xC0FFEE_BEEF);
    assert_bit_equal_or_close("n2 k2304 g8 u8 gate", &outs.k5_gate, &outs.rt_gate, 2304);
    assert_bit_equal_or_close("n2 k2304 g8 u8 up", &outs.k5_up, &outs.rt_up, 2304);
}

#[test]
fn parity_n4_k2304_8192x8192() {
    if !maybe_skip() {
        return;
    }
    // Qwen3.6-27B GDN gate/up shape: n_rows=8192 (intermediate=14336/TP2),
    // k=2304 (hidden after some splits). Pick smaller for parity; perf is
    // a separate cert.
    let outs = run_both(64, 64, 2304, 4, 0x1234_5678);
    assert_bit_equal_or_close("n4 k2304 g64 u64 gate", &outs.k5_gate, &outs.rt_gate, 2304);
    assert_bit_equal_or_close("n4 k2304 g64 u64 up", &outs.k5_up, &outs.rt_up, 2304);
}

#[test]
fn parity_n3_k4096_asym() {
    if !maybe_skip() {
        return;
    }
    let outs = run_both(48, 32, 4096, 3, 0xFEED_FACE);
    assert_bit_equal_or_close("n3 k4096 g48 u32 gate", &outs.k5_gate, &outs.rt_gate, 4096);
    assert_bit_equal_or_close("n3 k4096 g48 u32 up", &outs.k5_up, &outs.rt_up, 4096);
}

#[test]
fn parity_n2_k512_tail_unaligned() {
    if !maybe_skip() {
        return;
    }
    // K = 512 → n_blocks_per_row = 16 = exactly 1 outer iter.
    let outs = run_both(4, 4, 512, 2, 0xABCDEF);
    assert_bit_equal_or_close("n2 k512 g4 u4 gate", &outs.k5_gate, &outs.rt_gate, 512);
    assert_bit_equal_or_close("n2 k512 g4 u4 up", &outs.k5_up, &outs.rt_up, 512);
}

#[test]
fn parity_n4_asym_gdn_shape() {
    if !maybe_skip() {
        return;
    }
    // Qwen3.6-35B-A3B-Q4_0 / pp2tp2 / GDN shape: attn_qkv is
    // [local_conv_channels=4096, hidden=2048], attn_gate is
    // [local_d_inner=2048, hidden=2048]. Asymmetric n_rows where
    // n_rows_gate > n_rows_up: rows in [2048, 4096) have do_up=false
    // and must not dereference up_w pointers (which only have 2048
    // valid rows).
    let outs = run_both(4096, 2048, 2048, 4, 0x35B_A3B_F1);
    assert_bit_equal_or_close(
        "n4 gdn-asym g4096 u2048 gate",
        &outs.k5_gate,
        &outs.rt_gate,
        2048,
    );
    assert_bit_equal_or_close("n4 gdn-asym g4096 u2048 up", &outs.k5_up, &outs.rt_up, 2048);
}

#[test]
fn parity_n2_k800_tail_partial() {
    if !maybe_skip() {
        return;
    }
    // K = 800 → n_blocks_per_row = 25 = 2 full outer iters + tail of 9 blocks.
    // Exercises the partial-tail path in the row-tile loader.
    let outs = run_both(4, 4, 800, 2, 0xFADE_BEEF);
    assert_bit_equal_or_close("n2 k800 g4 u4 gate", &outs.k5_gate, &outs.rt_gate, 800);
    assert_bit_equal_or_close("n2 k800 g4 u4 up", &outs.k5_up, &outs.rt_up, 800);
}
