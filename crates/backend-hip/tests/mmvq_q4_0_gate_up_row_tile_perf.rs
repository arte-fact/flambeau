//! Microbench: row-tile vs K5 batched gate+up on representative decode shapes.
//! HIP-event timed, 200 warmup + 1000 measured launches per kernel per shape.
//! Reported as wall-time per launch (µs) and the ratio K5/row-tile (>1 = win).

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture; same shape rationale as siblings"
)]
#![expect(
    clippy::cast_possible_wrap,
    reason = "kernel-shape math bounded by GGUF dims"
)]

use flambeau_backend_hip::{
    device_count, HipDevice, HipEvent, HipKernel, HipModule, KernelArgs, LaunchCfg,
};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;
use flambeau_quant::{BlockQ4_0, BlockQ8_1, QK8_0};
use half::f16;

const QK8: usize = QK8_0;
const QK4: usize = 32;
const WARMUP: u32 = 200;
const ITERS: u32 = 1000;

fn maybe_skip() -> bool {
    matches!(device_count(), Ok(n) if n >= 1)
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

#[derive(Debug)]
struct ShapeResult {
    n_rows: usize,
    k: usize,
    n: usize,
    k5_us: f32,
    rt_us: f32,
}

fn bench_one(dev: &HipDevice, n_rows: usize, k: usize, n_slots: usize) -> ShapeResult {
    assert_eq!(k % QK4, 0);
    let n_blocks_per_row = k / QK4;

    let q_module = HipModule::load(0, kernels::hsaco("quantize_q8_1").unwrap()).unwrap();
    let k5_module =
        HipModule::load(0, kernels::hsaco("mmvq_q4_0_gate_up_batched").unwrap()).unwrap();
    let rt_module = HipModule::load(
        0,
        kernels::hsaco("mmvq_q4_0_gate_up_row_tile_batched").unwrap(),
    )
    .unwrap();
    let k_quantize: HipKernel<'_> = q_module.kernel("flambeau_quantize_row_q8_1").unwrap();
    let k5_entry = match n_slots {
        2 => "flambeau_mmvq_q4_0_gate_up_dp4a_q8_1_batched_n2",
        3 => "flambeau_mmvq_q4_0_gate_up_dp4a_q8_1_batched_n3",
        4 => "flambeau_mmvq_q4_0_gate_up_dp4a_q8_1_batched_n4",
        _ => panic!(),
    };
    let rt_entry = match n_slots {
        2 => "flambeau_mmvq_q4_0_gate_up_row_tile_dp4a_q8_1_batched_n2",
        3 => "flambeau_mmvq_q4_0_gate_up_row_tile_dp4a_q8_1_batched_n3",
        4 => "flambeau_mmvq_q4_0_gate_up_row_tile_dp4a_q8_1_batched_n4",
        _ => panic!(),
    };
    let k_k5: HipKernel<'_> = k5_module.kernel(k5_entry).unwrap();
    let k_rt: HipKernel<'_> = rt_module.kernel(rt_entry).unwrap();

    let mut blocks: Vec<BlockQ4_0> = Vec::with_capacity(n_rows * n_blocks_per_row);
    for i in 0..(n_rows * n_blocks_per_row) {
        blocks.push(random_q4_0_block(0xDEAD_BEEF, i));
    }
    let y_f32 = seeded_f32(0x1234, k * n_slots);

    let d_w = alloc_and_upload(dev, &blocks);
    let d_y_f32 = alloc_and_upload(dev, &y_f32);
    let y_blocks_per_slot = k / QK8;
    let d_y_q8_1 = dev
        .alloc(n_slots * y_blocks_per_slot * std::mem::size_of::<BlockQ8_1>())
        .unwrap();
    let d_g_out = dev.alloc(n_slots * n_rows * 4).unwrap();
    let d_u_out = dev.alloc(n_slots * n_rows * 4).unwrap();

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
    let g_out_ptr: u64 = d_g_out.as_usize() as u64;
    let u_out_ptr: u64 = d_u_out.as_usize() as u64;

    let bench = |k: &HipKernel<'_>, grid: u32, block: u32| -> f32 {
        let stream = dev.default_stream();
        // Warmup
        for _ in 0..WARMUP {
            let mut args = KernelArgs::new();
            args.push(&d_w_ptr);
            args.push(&d_w_ptr);
            args.push(&d_y_q8_1_ptr);
            args.push(&g_out_ptr);
            args.push(&u_out_ptr);
            args.push(&n_rows_i);
            args.push(&n_rows_i);
            args.push(&n_blocks_i);
            let cfg = LaunchCfg::one_d(grid, block);
            unsafe { k.launch(stream, cfg, args).unwrap() };
        }
        stream.synchronize().unwrap();

        let ev_start = HipEvent::new_timing(0).unwrap();
        let ev_end = HipEvent::new_timing(0).unwrap();
        ev_start.record(stream).unwrap();
        for _ in 0..ITERS {
            let mut args = KernelArgs::new();
            args.push(&d_w_ptr);
            args.push(&d_w_ptr);
            args.push(&d_y_q8_1_ptr);
            args.push(&g_out_ptr);
            args.push(&u_out_ptr);
            args.push(&n_rows_i);
            args.push(&n_rows_i);
            args.push(&n_blocks_i);
            let cfg = LaunchCfg::one_d(grid, block);
            unsafe { k.launch(stream, cfg, args).unwrap() };
        }
        ev_end.record(stream).unwrap();
        ev_end.synchronize().unwrap();
        let ms = ev_end.elapsed_ms_since(&ev_start).unwrap();
        (ms * 1000.0) / (ITERS as f32)
    };

    let k5_us = bench(&k_k5, n_rows as u32, 256);
    let rt_us = bench(&k_rt, (n_rows as u32).div_ceil(4), 256);

    unsafe {
        dev.dealloc(d_w, blocks.len() * std::mem::size_of::<BlockQ4_0>())
            .unwrap();
        dev.dealloc(d_y_f32, y_f32.len() * 4).unwrap();
        dev.dealloc(
            d_y_q8_1,
            n_slots * y_blocks_per_slot * std::mem::size_of::<BlockQ8_1>(),
        )
        .unwrap();
        dev.dealloc(d_g_out, n_slots * n_rows * 4).unwrap();
        dev.dealloc(d_u_out, n_slots * n_rows * 4).unwrap();
    }

    ShapeResult {
        n_rows,
        k,
        n: n_slots,
        k5_us,
        rt_us,
    }
}

#[test]
#[ignore = "perf microbench; opt in with --ignored"]
fn perf_row_tile_vs_k5() {
    if !maybe_skip() {
        eprintln!("[skip] no HIP device");
        return;
    }
    let dev = HipDevice::new(0).unwrap();
    dev.bind().unwrap();

    let shapes: Vec<(usize, usize, usize)> = vec![
        // (n_rows, k, n_slots). Qwen3.6-27B GDN intermediate=14336, TP2 → 7168.
        (7168, 2304, 2),
        (7168, 2304, 3),
        (7168, 2304, 4),
        // Qwen3.5-9B-class: smaller hidden, same intermediate split.
        (4096, 2048, 2),
        (4096, 2048, 4),
        // Dense FFN
        (4096, 4096, 2),
        (4096, 4096, 4),
        // Pathological tail / small n_rows
        (128, 2304, 2),
        (128, 2304, 4),
    ];

    eprintln!(
        "\n{:>10} | {:>6} | {:>4} | {:>9} | {:>9} | {:>6}",
        "n_rows", "k", "N", "K5(µs)", "RT(µs)", "K5/RT"
    );
    eprintln!("{}", "-".repeat(60));
    for (n_rows, k, n) in shapes {
        let r = bench_one(&dev, n_rows, k, n);
        let ratio = r.k5_us / r.rt_us;
        eprintln!(
            "{:>10} | {:>6} | {:>4} | {:>9.2} | {:>9.2} | {:>5.2}×",
            r.n_rows, r.k, r.n, r.k5_us, r.rt_us, ratio
        );
    }
}
