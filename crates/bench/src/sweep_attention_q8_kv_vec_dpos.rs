//! Correctness sweep + microbench for `attention_decode_q8_kv_vec_dpos`, the
//! 128-thread warp-per-token Q8_0-KV decode attention (CUDA-only). Global
//! layers: the KV tail is rows [0, position+1), position read on device.
//!
//! Each case runs the kernel at its `n_par` (gridDim.y; above 1 the shared
//! split-K combine reduces the parts, some of which may hold no token) and
//! compares against an F32 softmax over the dequantised rows the kernel reads.
//! The qwen35 cases (GQA 24/4, head_dim 256) also time the kernel against the
//! split-K `_dpos` chunk + combine it would replace, at the exact grid and at
//! the 32-chunk grid a captured decode bakes; the timings are logged, the cert
//! carries correctness only.

#![cfg(any(feature = "hip", feature = "cuda"))]

use std::path::Path;

use anyhow::{anyhow, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr, Event, LaunchCfg, Stream};
use flambeau_quant::{BlockQ8_0, QK8_0};
use half::f16;

use crate::backend::{BenchBackend, LaunchArgs};
use crate::cert::{now_utc_iso8601, Cert, ShapeResult, SCHEMA_VERSION};
use crate::harness::{alloc_and_upload, max_rel_err_with_floor, rig_for, seeded_f32_range};
use crate::sweep_attention_q8_kv_splitk_dpos::{dequantize_row_q8_0, quantize_row_q8_0};

const TOL: f32 = 5e-2;
const TOL_FORMULA: &str =
    "|err| <= 5e-2 * max(|ref|, sqrt(head_dim))  (correctness; delta-ppl quality cert)";
const MAX_SPLITK_CHUNKS: usize = 32;
const SPLITK_MIN_TOKENS_KV: usize = 32;
const BENCH_WARMUP: usize = 20;
const BENCH_ITERS: usize = 200;

struct Case {
    head_dim: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    position: usize,
    n_par: usize,
    bench: bool,
}

const fn case(head_dim: usize, n_heads_q: usize, n_heads_kv: usize, position: usize, n_par: usize, bench: bool) -> Case {
    Case { head_dim, n_heads_q, n_heads_kv, position, n_par, bench }
}

const CASES: &[Case] = &[
    // qwen35 27B full attention (GQA 24/4, head_dim 256), one block per head,
    // from the first decode token to a full 8k context.
    case(256, 24, 4, 0, 1, false),
    case(256, 24, 4, 31, 1, false),
    case(256, 24, 4, 32, 1, true),
    case(256, 24, 4, 180, 1, true),
    case(256, 24, 4, 1023, 1, true),
    case(256, 24, 4, 1024, 1, false),
    case(256, 24, 4, 4000, 1, true),
    case(256, 24, 4, 8191, 1, true),
    // Split over parts + combine, including parts with no token in the tail.
    case(256, 24, 4, 180, 4, false),
    case(256, 24, 4, 1023, 4, true),
    case(256, 24, 4, 4000, 8, true),
    case(256, 24, 4, 8191, 8, true),
    // Other GQA groupings / head_dim 128.
    case(256, 16, 2, 1641, 1, false),
    case(128, 4, 1, 300, 2, false),
];

/// The split-K chunk size the non-graph decode picks for a `n`-token tail.
fn splitk_chunk_size(n: usize) -> usize {
    if n <= SPLITK_MIN_TOKENS_KV {
        return n;
    }
    n.div_ceil(MAX_SPLITK_CHUNKS).max(32).next_multiple_of(4)
}

struct Kernels<'m, B: BenchBackend> {
    vec_d128: B::Kernel<'m>,
    vec_d256: B::Kernel<'m>,
    splitk_chunk: B::Kernel<'m>,
    combine: B::Kernel<'m>,
}

/// Backend-generic entry point.
/// # Errors
/// Propagates device / launch / cert-write failures.
pub fn run_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let vec_bytes = B::image("attention_decode_q8_kv_vec_dpos")
        .ok_or_else(|| anyhow!("attention_decode_q8_kv_vec_dpos not compiled (CUDA-only)"))?;
    let vec_module = B::load(&dev, vec_bytes)?;
    let splitk_bytes = B::image("attention_decode_q8_kv_splitk_dpos")
        .ok_or_else(|| anyhow!("attention_decode_q8_kv_splitk_dpos not compiled (CUDA-only)"))?;
    let splitk_module = B::load(&dev, splitk_bytes)?;
    let combine_bytes = B::image("attention_decode_q8_kv_splitk")
        .ok_or_else(|| anyhow!("attention_decode_q8_kv_splitk not compiled"))?;
    let combine_module = B::load(&dev, combine_bytes)?;
    let kernels = Kernels::<B> {
        vec_d128: B::kernel(&vec_module, "flambeau_attention_decode_q8_kv_vec_dpos_d128")?,
        vec_d256: B::kernel(&vec_module, "flambeau_attention_decode_q8_kv_vec_dpos_d256")?,
        splitk_chunk: B::kernel(&splitk_module, "flambeau_attention_decode_q8_kv_splitk_dpos_chunk")?,
        combine: B::kernel(&combine_module, "flambeau_attention_decode_q8_kv_splitk_combine")?,
    };
    let pmc = B::attributes(&kernels.vec_d256)?;

    let mut results = Vec::new();
    for case in CASES {
        let seed = 0x7EC0_D905 ^ (case.head_dim as u64 * 7919) ^ (case.position as u64 * 101)
            ^ (case.n_par as u64 * 13) ^ (case.n_heads_q as u64 * 3);
        let (got, reference) = run_case::<B>(&dev, &kernels, case, seed)?;
        let max_rel = max_rel_err_with_floor(&got, &reference, (case.head_dim as f32).sqrt() * 0.01);
        tracing::info!(
            target: "flambeau_bench::sweep_attention_q8_kv_vec_dpos",
            head_dim = case.head_dim, n_heads_q = case.n_heads_q, n_heads_kv = case.n_heads_kv,
            position = case.position, n_par = case.n_par, max_rel, "q8 kv vec decode shape"
        );
        results.push(ShapeResult {
            m: case.n_heads_q,
            k: case.position + 1,
            n: case.head_dim,
            seed,
            max_rel_err: max_rel,
            tolerance: TOL,
            pass: max_rel <= TOL,
        });
    }

    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: format!("attention_decode_q8_kv_vec_dpos_{arch}"),
        backend: dev.backend().to_string(),
        arch: arch.to_string(),
        op: "attention_decode_q8_kv".to_string(),
        dtype_weight: "Q8_0".to_string(),
        dtype_activation: "F16".to_string(),
        tolerance_formula: TOL_FORMULA.to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_for(arch),
        pmc: Some(pmc),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

/// Device buffers of one case.
struct Bufs {
    q: DevicePtr,
    k: DevicePtr,
    v: DevicePtr,
    pos: DevicePtr,
    out: DevicePtr,
    m: DevicePtr,
    s: DevicePtr,
    o: DevicePtr,
}

#[allow(clippy::undocumented_unsafe_blocks, clippy::too_many_lines)]
fn run_case<B: BenchBackend>(
    dev: &B::Device,
    kernels: &Kernels<'_, B>,
    case: &Case,
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let Case { head_dim, n_heads_q, n_heads_kv, position, n_par, bench } = *case;
    let group = n_heads_q / n_heads_kv;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_width = n_heads_kv * head_dim;
    let nb_per_row = head_dim / QK8_0;
    let n_kv = position + 1;

    let q_host: Vec<f16> = seeded_f32_range(seed, n_heads_q * head_dim, -0.5, 0.5)
        .iter().map(|v| f16::from_f32(*v)).collect();
    let cache_n = n_kv * kv_width;
    let k_f32 = seeded_f32_range(seed ^ 0xA1, cache_n, -0.5, 0.5);
    let v_f32 = seeded_f32_range(seed ^ 0xA2, cache_n, -0.5, 0.5);
    let n_block_rows = n_kv * n_heads_kv;
    let mut k_blocks: Vec<BlockQ8_0> = Vec::with_capacity(n_block_rows * nb_per_row);
    let mut v_blocks: Vec<BlockQ8_0> = Vec::with_capacity(n_block_rows * nb_per_row);
    for row in 0..n_block_rows {
        k_blocks.extend(quantize_row_q8_0(&k_f32[row * head_dim..(row + 1) * head_dim]));
        v_blocks.extend(quantize_row_q8_0(&v_f32[row * head_dim..(row + 1) * head_dim]));
    }

    let out_len = n_heads_q * head_dim;
    // Partials sized for the larger of this case's parts and the 32-chunk
    // split-K grid the microbench runs.
    let part_ms = n_heads_q * n_par.max(MAX_SPLITK_CHUNKS);
    let bufs = Bufs {
        q: alloc_and_upload(dev, &q_host),
        k: alloc_and_upload(dev, &k_blocks),
        v: alloc_and_upload(dev, &v_blocks),
        pos: alloc_and_upload(dev, &[position as i32]),
        out: dev.alloc(out_len * 2)?,
        m: dev.alloc(part_ms * 4)?,
        s: dev.alloc(part_ms * 4)?,
        o: dev.alloc(part_ms * head_dim * 4)?,
    };
    let vec_kernel = if head_dim == 128 { &kernels.vec_d128 } else { &kernels.vec_d256 };
    let stream = dev.default_stream();

    launch_vec::<B>(vec_kernel, &kernels.combine, stream, &bufs, case, scale)?;
    stream.synchronize()?;
    let mut out_f16: Vec<f16> = vec![f16::from_f32(0.0); out_len];
    unsafe {
        dev.memcpy_async(stream, CopyDirection::DeviceToHost, DevicePtr(out_f16.as_mut_ptr() as usize), bufs.out, out_len * 2)?;
    }
    stream.synchronize()?;
    let got: Vec<f32> = out_f16.iter().map(|v| v.to_f32()).collect();

    if bench {
        let vec_us = time_us::<B>(dev, || launch_vec::<B>(vec_kernel, &kernels.combine, stream, &bufs, case, scale))?;
        let exact_chunk = splitk_chunk_size(n_kv).max(1);
        let exact_chunks = n_kv.div_ceil(exact_chunk);
        let exact_us = time_us::<B>(dev, || {
            launch_splitk::<B>(kernels, stream, &bufs, case, scale, exact_chunks, exact_chunk)
        })?;
        let baked_chunk = splitk_chunk_size(n_kv).max(SPLITK_MIN_TOKENS_KV);
        let baked_us = time_us::<B>(dev, || {
            launch_splitk::<B>(kernels, stream, &bufs, case, scale, MAX_SPLITK_CHUNKS, baked_chunk)
        })?;
        tracing::info!(
            target: "flambeau_bench::sweep_attention_q8_kv_vec_dpos",
            head_dim, n_heads_q, n_heads_kv, position, n_par,
            vec_us, splitk_exact_us = exact_us, splitk_exact_chunks = exact_chunks,
            splitk_baked32_us = baked_us,
            speedup_vs_baked32 = baked_us / vec_us,
            "q8 kv vec decode microbench"
        );
        println!(
            "microbench D={head_dim} {n_heads_q}/{n_heads_kv} pos={position} n_par={n_par}: \
             vec {vec_us:.2} us | split-K exact ({exact_chunks} chunks) {exact_us:.2} us | \
             split-K baked 32 {baked_us:.2} us | x{:.2} vs baked, x{:.2} vs exact",
            baked_us / vec_us,
            exact_us / vec_us
        );
    }

    unsafe {
        dev.dealloc(bufs.q, q_host.len() * 2)?;
        dev.dealloc(bufs.k, k_blocks.len() * std::mem::size_of::<BlockQ8_0>())?;
        dev.dealloc(bufs.v, v_blocks.len() * std::mem::size_of::<BlockQ8_0>())?;
        dev.dealloc(bufs.pos, 4)?;
        dev.dealloc(bufs.out, out_len * 2)?;
        dev.dealloc(bufs.m, part_ms * 4)?;
        dev.dealloc(bufs.s, part_ms * 4)?;
        dev.dealloc(bufs.o, part_ms * head_dim * 4)?;
    }

    // Reference: F32 softmax over the dequantised rows the kernel reads.
    let mut k_rt = vec![0.0f32; cache_n];
    let mut v_rt = vec![0.0f32; cache_n];
    for row in 0..n_block_rows {
        let bk = &k_blocks[row * nb_per_row..(row + 1) * nb_per_row];
        let bv = &v_blocks[row * nb_per_row..(row + 1) * nb_per_row];
        k_rt[row * head_dim..(row + 1) * head_dim].copy_from_slice(&dequantize_row_q8_0(bk));
        v_rt[row * head_dim..(row + 1) * head_dim].copy_from_slice(&dequantize_row_q8_0(bv));
    }
    let q_inputs: Vec<f32> = q_host.iter().map(|v| v.to_f32()).collect();
    let mut reference = vec![0.0f32; out_len];
    for qh in 0..n_heads_q {
        let kvh = qh / group;
        let q_off = qh * head_dim;
        let scores: Vec<f32> = (0..n_kv)
            .map(|t| {
                let base = (t * n_heads_kv + kvh) * head_dim;
                let dot: f64 = (0..head_dim).map(|d| (q_inputs[q_off + d] * k_rt[base + d]) as f64).sum();
                (dot as f32) * scale
            })
            .collect();
        let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let w: Vec<f32> = scores.iter().map(|&sc| (sc - m).exp()).collect();
        let inv = 1.0f32 / (w.iter().map(|&e| e as f64).sum::<f64>() as f32);
        for d in 0..head_dim {
            let acc: f64 = (0..n_kv)
                .map(|t| (w[t] * inv * v_rt[(t * n_heads_kv + kvh) * head_dim + d]) as f64)
                .sum();
            reference[q_off + d] = f16::from_f32(acc as f32).to_f32();
        }
    }
    Ok((got, reference))
}

fn launch_vec<B: BenchBackend>(
    kernel: &B::Kernel<'_>,
    combine: &B::Kernel<'_>,
    stream: &<B::Device as Device>::Stream,
    bufs: &Bufs,
    case: &Case,
    scale: f32,
) -> Result<()> {
    let ptrs = [bufs.q, bufs.k, bufs.v, bufs.out, bufs.m, bufs.s, bufs.o].map(|p| p.as_usize() as u64);
    let pos_ptr = bufs.pos.as_usize() as u64;
    let n_heads_q_i = case.n_heads_q as i32;
    let n_heads_kv_i = case.n_heads_kv as i32;
    let mut a = <B::Args<'_>>::default();
    for p in &ptrs {
        a.push_arg(p);
    }
    a.push_arg(&n_heads_q_i);
    a.push_arg(&n_heads_kv_i);
    a.push_arg(&scale);
    a.push_arg(&pos_ptr);
    let cfg = LaunchCfg { grid: (case.n_heads_q as u32, case.n_par as u32, 1), block: (128, 1, 1), shared_bytes: 0 };
    B::launch(kernel, stream, cfg, a)?;
    if case.n_par > 1 {
        launch_combine::<B>(combine, stream, bufs, case.n_heads_q, case.n_par, case.head_dim)?;
    }
    Ok(())
}

fn launch_splitk<B: BenchBackend>(
    kernels: &Kernels<'_, B>,
    stream: &<B::Device as Device>::Stream,
    bufs: &Bufs,
    case: &Case,
    scale: f32,
    n_chunks: usize,
    chunk_size: usize,
) -> Result<()> {
    let ptrs = [bufs.q, bufs.k, bufs.v, bufs.m, bufs.s, bufs.o].map(|p| p.as_usize() as u64);
    let pos_ptr = bufs.pos.as_usize() as u64;
    let ints = [case.n_heads_q, case.n_heads_kv, case.head_dim, n_chunks, chunk_size].map(|v| v as i32);
    let zero = 0i32;
    let mut a = <B::Args<'_>>::default();
    for p in &ptrs {
        a.push_arg(p);
    }
    for v in &ints {
        a.push_arg(v);
    }
    a.push_arg(&scale);
    a.push_arg(&zero);
    a.push_arg(&zero);
    a.push_arg(&zero);
    a.push_arg(&pos_ptr);
    let cfg = LaunchCfg {
        grid: (case.n_heads_q as u32, n_chunks as u32, 1),
        block: ((case.head_dim / 4) as u32, 1, 1),
        shared_bytes: 0,
    };
    B::launch(&kernels.splitk_chunk, stream, cfg, a)?;
    launch_combine::<B>(&kernels.combine, stream, bufs, case.n_heads_q, n_chunks, case.head_dim)
}

fn launch_combine<B: BenchBackend>(
    combine: &B::Kernel<'_>,
    stream: &<B::Device as Device>::Stream,
    bufs: &Bufs,
    n_heads_q: usize,
    n_parts: usize,
    head_dim: usize,
) -> Result<()> {
    let ptrs = [bufs.m, bufs.s, bufs.o, bufs.out].map(|p| p.as_usize() as u64);
    let ints = [n_heads_q, n_parts, head_dim].map(|v| v as i32);
    let mut a = <B::Args<'_>>::default();
    for p in &ptrs {
        a.push_arg(p);
    }
    for v in &ints {
        a.push_arg(v);
    }
    let cfg = LaunchCfg { grid: (n_heads_q as u32, 1, 1), block: (head_dim as u32, 1, 1), shared_bytes: 0 };
    B::launch(combine, stream, cfg, a)?;
    Ok(())
}

/// Mean device time of one `launch` call over `BENCH_ITERS` back-to-back
/// launches, after `BENCH_WARMUP`, in microseconds.
fn time_us<B: BenchBackend>(dev: &B::Device, mut launch: impl FnMut() -> Result<()>) -> Result<f32> {
    let stream = dev.default_stream();
    for _ in 0..BENCH_WARMUP {
        launch()?;
    }
    let start = dev.new_timing_event()?;
    let end = dev.new_timing_event()?;
    start.record(stream)?;
    for _ in 0..BENCH_ITERS {
        launch()?;
    }
    end.record(stream)?;
    end.synchronize()?;
    Ok(end.elapsed_ms_since(&start)? * 1000.0 / BENCH_ITERS as f32)
}
