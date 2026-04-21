//! V2.1 step 1 — measure the full Rust-side launch path cost per kernel
//! invocation, compared against `rocprof`'s host-side `hipModuleLaunchKernel`
//! timing. The delta is our Rust-FFI overhead — the hypothesised residual
//! ~1.5-3 ms/token gap to llama.cpp per the V1.7.6 cert.
//!
//! Usage:
//!   cargo run --release -p flambeau-backend-hip --example launch_overhead_bench --features hip
//!
//! Then run it under rocprof:
//!   rocprof --hip-trace -o /tmp/lob/out.csv target/release/examples/launch_overhead_bench
//!   # inspect /tmp/lob/out.hip_stats.csv → hipModuleLaunchKernel avg_ns
//!   # compare that to "avg per-call" printed by this binary's wall-clock
//!
//! The kernel is the cheapest we have (`scale_f32`, 1-element buffer, no
//! real work) so kernel-side time is ~kernel-launch-latency and the
//! measurement is dominated by the launch path.

use std::time::Instant;

use flambeau_backend_hip::{device_count, HipDevice, HipModule, KernelArgs, LaunchCfg};
#[allow(unused_imports)]
use flambeau_core::{Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;

fn main() -> anyhow::Result<()> {
    let n_iters: usize = std::env::var("FLAMBEAU_LOB_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);
    let sync_every: usize = std::env::var("FLAMBEAU_LOB_SYNC_EVERY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0); // 0 = sync only at end (best ILP)

    if device_count().unwrap_or(0) < 1 {
        eprintln!("no HIP device — skipping");
        return Ok(());
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;
    let module_bytes = kernels::hsaco("scale_f32").expect("scale_f32 not compiled");
    let module = HipModule::load(0, module_bytes)?;
    let kernel = module.kernel("flambeau_scale_f32")?;

    // 1-element scratch buffer; scale by 1.0 = no observable work.
    let d_ptr = dev.alloc(4)?;
    let stream = dev.default_stream();
    let n_elems: i32 = 1;
    let scale: f32 = 1.0f32;
    let dst_u64: u64 = d_ptr.as_usize() as u64;

    // Warm-up: 1k launches so first-launch JIT/cache effects don't skew.
    for _ in 0..1000 {
        let mut args = KernelArgs::new();
        args.push(&dst_u64);      // x
        args.push(&dst_u64);      // y (same buffer, 1.0 scale = no-op)
        args.push(&n_elems);      // n
        args.push(&scale);        // scale
        let cfg = LaunchCfg::one_d(1, 32);
        unsafe { kernel.launch(stream, cfg, args)? };
    }
    stream.synchronize()?;

    // Measured window.
    let t0 = Instant::now();
    for i in 0..n_iters {
        let mut args = KernelArgs::new();
        args.push(&dst_u64);      // x
        args.push(&dst_u64);      // y (same buffer, 1.0 scale = no-op)
        args.push(&n_elems);      // n
        args.push(&scale);        // scale
        let cfg = LaunchCfg::one_d(1, 32);
        unsafe { kernel.launch(stream, cfg, args)? };
        if sync_every > 0 && (i + 1) % sync_every == 0 {
            stream.synchronize()?;
        }
    }
    stream.synchronize()?;
    let dt = t0.elapsed();

    let per_call_ns = dt.as_nanos() as f64 / n_iters as f64;
    let per_call_us = per_call_ns / 1000.0;
    eprintln!(
        "[lob] {} iterations, sync_every={}, total {:.3} ms → {:.3} µs/launch",
        n_iters,
        sync_every,
        dt.as_secs_f64() * 1000.0,
        per_call_us
    );
    eprintln!("[lob] (rocprof hipModuleLaunchKernel baseline from V1.7.6 cert: 3.67 µs avg)");

    unsafe { dev.dealloc(d_ptr, 4)? };
    Ok(())
}
