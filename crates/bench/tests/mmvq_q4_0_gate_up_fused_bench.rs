//! Head-to-head perf microbench: fused `mmvq_q4_0_gate_up_t128_dp4a`
//! vs two separate `mmvq_q4_0_q8_1` launches.
//!
//! Validates whether the single-launch fused gate+up kernel beats two
//! sequential `qmatmul()` calls at decode (m=1). Shapes pulled from
//! Qwen3.6-27B-Q4_0 dense FFN: hidden=5120, intermediate=17408.
//!
//! Run with: `cargo test --release -p flambeau-bench --test
//! mmvq_q4_0_gate_up_fused_bench -- --nocapture --ignored`

#![cfg(feature = "hip")]

use anyhow::Result;
use flambeau_backend_hip::{HipDevice, HipEvent, HipModule, KernelArgs, LaunchCfg};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;
use flambeau_quant::{BlockQ4_0, BlockQ8_1};

const QK8: usize = 32;
const N_WARMUP: usize = 50;
const N_ITERS: usize = 1000;

fn alloc_and_fill_bytes(dev: &HipDevice, nbytes: usize, fill: u8) -> Result<DevicePtr> {
    let p = dev.alloc(nbytes)?;
    let xs = vec![fill; nbytes];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            p,
            DevicePtr(xs.as_ptr() as usize),
            nbytes,
        )?;
    }
    dev.default_stream().synchronize()?;
    Ok(p)
}

fn run_split(dev: &HipDevice, n_rows: usize, k: usize) -> Result<f32> {
    let bytes_w = n_rows * (k / 32) * std::mem::size_of::<BlockQ4_0>();
    let bytes_y = (k / QK8) * std::mem::size_of::<BlockQ8_1>();
    let bytes_dst = n_rows * 4;
    let d_gw = alloc_and_fill_bytes(dev, bytes_w, 0x55)?;
    let d_uw = alloc_and_fill_bytes(dev, bytes_w, 0x66)?;
    let d_y = alloc_and_fill_bytes(dev, bytes_y, 0x33)?;
    let d_g = dev.alloc(bytes_dst)?;
    let d_u = dev.alloc(bytes_dst)?;

    let bytes = kernels::hsaco("mmvq_q4_0").ok_or_else(|| anyhow::anyhow!("mmvq_q4_0 not compiled"))?;
    let module = HipModule::load(dev.id(), bytes)?;
    let kernel = module.kernel("flambeau_mmvq_q4_0_q8_1")?;

    let n_rows_i = n_rows as i32;
    let n_blocks = (k / 32) as i32;
    let stream = dev.default_stream();
    let cfg = LaunchCfg::one_d(n_rows as u32, 256);

    let launch = |w: DevicePtr, dst: DevicePtr| -> Result<()> {
        let wp: u64 = w.as_usize() as u64;
        let yp: u64 = d_y.as_usize() as u64;
        let dp: u64 = dst.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&wp); args.push(&yp); args.push(&dp); args.push(&n_rows_i); args.push(&n_blocks);
        unsafe { kernel.launch(stream, cfg, args)?; }
        Ok(())
    };

    for _ in 0..N_WARMUP { launch(d_gw, d_g)?; launch(d_uw, d_u)?; }
    stream.synchronize()?;

    let s = HipEvent::new_timing(dev.id())?;
    let e = HipEvent::new_timing(dev.id())?;
    s.record(stream)?;
    for _ in 0..N_ITERS { launch(d_gw, d_g)?; launch(d_uw, d_u)?; }
    e.record(stream)?;
    e.synchronize()?;
    let per_us = e.elapsed_ms_since(&s)? * 1000.0 / N_ITERS as f32;

    unsafe {
        dev.dealloc(d_gw, bytes_w)?; dev.dealloc(d_uw, bytes_w)?;
        dev.dealloc(d_y, bytes_y)?; dev.dealloc(d_g, bytes_dst)?; dev.dealloc(d_u, bytes_dst)?;
    }
    Ok(per_us)
}

fn run_fused_t128(dev: &HipDevice, n_rows: usize, k: usize) -> Result<f32> {
    let bytes_w = n_rows * (k / 32) * std::mem::size_of::<BlockQ4_0>();
    let bytes_y = (k / QK8) * std::mem::size_of::<BlockQ8_1>();
    let bytes_dst = n_rows * 4;
    let d_gw = alloc_and_fill_bytes(dev, bytes_w, 0x55)?;
    let d_uw = alloc_and_fill_bytes(dev, bytes_w, 0x66)?;
    let d_y = alloc_and_fill_bytes(dev, bytes_y, 0x33)?;
    let d_g = dev.alloc(bytes_dst)?;
    let d_u = dev.alloc(bytes_dst)?;

    let bytes = kernels::hsaco("mmvq_q4_0_gate_up_t128_dp4a")
        .ok_or_else(|| anyhow::anyhow!("mmvq_q4_0_gate_up_t128_dp4a not compiled"))?;
    let module = HipModule::load(dev.id(), bytes)?;
    let kernel = module.kernel("flambeau_mmvq_q4_0_gate_up_t128_dp4a_q8_1")?;

    let n_rows_g = n_rows as i32;
    let n_rows_u = n_rows as i32;
    let n_blocks = (k / 32) as i32;
    let stream = dev.default_stream();
    let cfg = LaunchCfg::one_d(n_rows as u32, 128);

    let launch = || -> Result<()> {
        let gw: u64 = d_gw.as_usize() as u64; let uw: u64 = d_uw.as_usize() as u64;
        let yp: u64 = d_y.as_usize() as u64; let gp: u64 = d_g.as_usize() as u64; let up: u64 = d_u.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&gw); args.push(&uw); args.push(&yp); args.push(&gp); args.push(&up);
        args.push(&n_rows_g); args.push(&n_rows_u); args.push(&n_blocks);
        unsafe { kernel.launch(stream, cfg, args)?; }
        Ok(())
    };

    for _ in 0..N_WARMUP { launch()?; }
    stream.synchronize()?;

    let s = HipEvent::new_timing(dev.id())?;
    let e = HipEvent::new_timing(dev.id())?;
    s.record(stream)?;
    for _ in 0..N_ITERS { launch()?; }
    e.record(stream)?;
    e.synchronize()?;
    let per_us = e.elapsed_ms_since(&s)? * 1000.0 / N_ITERS as f32;

    unsafe {
        dev.dealloc(d_gw, bytes_w)?; dev.dealloc(d_uw, bytes_w)?;
        dev.dealloc(d_y, bytes_y)?; dev.dealloc(d_g, bytes_dst)?; dev.dealloc(d_u, bytes_dst)?;
    }
    Ok(per_us)
}

#[test]
#[ignore]
fn mmvq_q4_0_gate_up_fused_vs_split() -> Result<()> {
    let dev = HipDevice::new(0)?;
    // Qwen3.6-27B: hidden=5120, intermediate=17408. Also test a smaller
    // shape to capture launch-overhead-dominated regime.
    let shapes: &[(&str, usize, usize)] = &[
        ("qwen36_27b_ffn (n=17408, k=5120)", 17408, 5120),
        ("qwen36_27b_attn_qkv (n=3328, k=5120)", 3328, 5120),
        ("small (n=2048, k=2048)", 2048, 2048),
    ];

    println!("\n=== mmvq_q4_0 gate+up: 2 split launches vs 1 fused t128_dp4a ===");
    println!("    iters={N_ITERS}, warmup={N_WARMUP}\n");
    for (label, n_rows, k) in shapes {
        let split_us = run_split(&dev, *n_rows, *k)?;
        let fused_us = run_fused_t128(&dev, *n_rows, *k)?;
        let speedup = split_us / fused_us;
        println!(
            "  {label:<46} split={split_us:7.2}us  fused={fused_us:7.2}us  speedup={speedup:.2}x"
        );
    }
    Ok(())
}
