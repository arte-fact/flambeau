//! Head-to-head perf microbench: `mmvq_q5_k_dp4a` vs `mmvq_q5_k_r2`.
//!
//! Times both kernels back-to-back at the production shape
//! (Qwen3.6-27B `ssm_out`: k=6144, n=5120, m=1) using `HipEvent` timing
//! over 1000 iterations after a warm-up. Reports avg µs/launch and
//! the relative speedup.
//!
//! Run with: `cargo test --release -p flambeau-bench --test
//! mmvq_q5_k_dp4a_vs_r2_bench -- --nocapture --ignored`
//!
//! Ignored by default so it doesn't run on every `cargo test` (needs a
//! gfx906 device + ~50 MB free per benched shape).

#![cfg(feature = "hip")]

use anyhow::Result;
use flambeau_backend_hip::{HipDevice, HipEvent, HipKernel, HipModule, KernelArgs, LaunchCfg};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;
use flambeau_quant::{BlockQ5K, BlockQ8_1, QK_K};

const QK8: usize = 32;

const N_WARMUP: usize = 50;
const N_ITERS: usize = 1000;

struct KernelSpec {
    stem: &'static str,
    entry: &'static str,
    threads: u32,
    rows_per_block: u32,
}

fn run_one(dev: &HipDevice, spec: &KernelSpec, m: usize, k: usize) -> Result<(f32, f32)> {
    let bytes = kernels::hsaco(spec.stem)
        .ok_or_else(|| anyhow::anyhow!("{} not compiled", spec.stem))?;
    let module = HipModule::load(dev.id(), bytes)?;
    let kernel: HipKernel = module.kernel(spec.entry)?;

    let n_super_per_row = k / QK_K;
    let weights_blocks = m * n_super_per_row;
    let weights_bytes = weights_blocks * std::mem::size_of::<BlockQ5K>();
    let d_x = dev.alloc(weights_bytes)?;
    let y_q8_1_blocks = k / QK8;
    let y_q8_1_bytes = y_q8_1_blocks * std::mem::size_of::<BlockQ8_1>();
    let d_y_q8_1 = dev.alloc(y_q8_1_bytes)?;
    let d_dst = dev.alloc(m * 4)?;

    // Fill with deterministic bytes so the kernel does real work.
    let xs = vec![0x55u8; weights_bytes];
    let ys = vec![0x33u8; y_q8_1_bytes];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d_x,
            DevicePtr(xs.as_ptr() as usize),
            weights_bytes,
        )?;
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d_y_q8_1,
            DevicePtr(ys.as_ptr() as usize),
            y_q8_1_bytes,
        )?;
    }
    dev.default_stream().synchronize()?;

    let m_i = m as i32;
    let n_units = (k / QK_K) as i32;
    let d_x_ptr: u64 = d_x.as_usize() as u64;
    let d_y_ptr: u64 = d_y_q8_1.as_usize() as u64;
    let d_dst_ptr: u64 = d_dst.as_usize() as u64;
    let grid = (m as u32).div_ceil(spec.rows_per_block);
    let cfg = LaunchCfg::one_d(grid, spec.threads);

    let stream = dev.default_stream();

    // Warmup.
    for _ in 0..N_WARMUP {
        let mut args = KernelArgs::new();
        args.push(&d_x_ptr);
        args.push(&d_y_ptr);
        args.push(&d_dst_ptr);
        args.push(&m_i);
        args.push(&n_units);
        unsafe { kernel.launch(stream, cfg, args)?; }
    }
    stream.synchronize()?;

    // Timed.
    let t_start = HipEvent::new_timing(dev.id())?;
    let t_stop = HipEvent::new_timing(dev.id())?;
    t_start.record(stream)?;
    for _ in 0..N_ITERS {
        let mut args = KernelArgs::new();
        args.push(&d_x_ptr);
        args.push(&d_y_ptr);
        args.push(&d_dst_ptr);
        args.push(&m_i);
        args.push(&n_units);
        unsafe { kernel.launch(stream, cfg, args)?; }
    }
    t_stop.record(stream)?;
    t_stop.synchronize()?;
    let total_ms = t_stop.elapsed_ms_since(&t_start)?;
    let per_us = total_ms * 1000.0 / N_ITERS as f32;

    unsafe {
        dev.dealloc(d_x, weights_bytes)?;
        dev.dealloc(d_y_q8_1, y_q8_1_bytes)?;
        dev.dealloc(d_dst, m * 4)?;
    }
    Ok((total_ms, per_us))
}

#[test]
#[ignore]
fn mmvq_q5_k_dp4a_vs_r2() -> Result<()> {
    let dev = HipDevice::new(0)?;

    // Qwen3.6-27B ssm_out shape: k=6144, n=5120 (per rank if split — at
    // single-GPU bench we use the full output dim).
    let shapes: &[(&str, usize, usize)] = &[
        ("ssm_out_full (n=5120, k=6144)", 5120, 6144),
        ("ssm_out_pp4_quarter (n=1280, k=6144)", 1280, 6144),
        ("small (n=64, k=2048)", 64, 2048),
    ];

    let r2 = KernelSpec {
        stem: "mmvq_q5_k_r2",
        entry: "flambeau_mmvq_q5_k_r2_q8_1",
        threads: 64,
        rows_per_block: 2,
    };
    let dp4a = KernelSpec {
        stem: "mmvq_q5_k_dp4a",
        entry: "flambeau_mmvq_q5_k_dp4a_q8_1",
        threads: 256,
        rows_per_block: 1,
    };

    println!("\n=== mmvq_q5_k: r2 (nw1 64-thread) vs dp4a (256-thread SDOT4) ===");
    println!("    iters={N_ITERS}, warmup={N_WARMUP}\n");
    for (label, n_rows, k) in shapes {
        let (_t_r2_ms, r2_us) = run_one(&dev, &r2, *n_rows, *k)?;
        let (_t_dp4a_ms, dp4a_us) = run_one(&dev, &dp4a, *n_rows, *k)?;
        let speedup = r2_us / dp4a_us;
        println!(
            "  {label:<46} r2={r2_us:7.2}us  dp4a={dp4a_us:7.2}us  speedup={speedup:.2}x"
        );
    }
    Ok(())
}
