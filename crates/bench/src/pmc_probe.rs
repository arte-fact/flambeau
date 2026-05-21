//! One-shot kernel launcher used by `flambeau pmc-probe` under rocprofv3.
//! Runs exactly one kernel launch + one `hipStreamSynchronize` so the
//! profiler sees a clean single-dispatch trace.

#![cfg(feature = "hip")]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "sweep harness — every unsafe block is a kernel launch or a memcpy_async \
              over buffers allocated locally in the same function and freed before \
              return; invariant is uniform across all sites."
)]

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::{HipDevice, HipKernel, HipModule, KernelArgs, LaunchCfg};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;
use flambeau_quant::{BlockQ4_1, BlockQ8_0, BlockQ8_1, BlockQ8_1Mmq, QK8_0, QK8_1_MMQ};
use half::f16;

/// Launch a single `(kernel_stem, shape)` combination on device 0, return
/// the kernel entry name that was launched (useful so the PMC wrapper knows
/// what Kernel_Name string to filter by in rocprofv3's CSV).
pub fn run_one(kernel_stem: &str, m: usize, k: usize, n: usize) -> Result<String> {
    let (entry, threads) = match kernel_stem {
        "mmvq_q8_0" => ("flambeau_mmvq_q8_0_q8_1", 256u32),
        "mmvq_q4_0" => ("flambeau_mmvq_q4_0_q8_1", 256),
        "mmvq_q4_k" => ("flambeau_mmvq_q4_k_q8_1", 64),
        "mmvq_q5_k" => ("flambeau_mmvq_q5_k_q8_1", 64),
        "mmvq_q6_k" => ("flambeau_mmvq_q6_k_q8_1", 64),
        "mmvq_q4_k_r2" => ("flambeau_mmvq_q4_k_r2_q8_1", 64),
        "mmvq_q5_k_r2" => ("flambeau_mmvq_q5_k_r2_q8_1", 64),
        "mmvq_q6_k_r4" => ("flambeau_mmvq_q6_k_r4_q8_1", 64),
        "mmq_q8_0_oracle" => ("flambeau_mmq_q8_0_oracle_q8_1", 256),
        "mmq_q8_0_4warp" => ("flambeau_mmq_q8_0_4warp_q8_1", 256),
        // candle port: 2D block (64, 4, 1), DS4 Q8_1 activation, 9 args, 30336 B LDS.
        "mmq_q4_1_4warp_lds" => ("flambeau_mmq_q4_1_4warp_lds_q8_1", 0),
        // / K-quant wave64 MMQ ports. All three share the same
        // 8-arg signature and launch shape; only weight block size differs.
        "mmq_q4_K_wave64" => ("flambeau_mmq_q4_K_wave64_q8_1", 64),
        "mmq_q5_K_wave64" => ("flambeau_mmq_q5_K_wave64_q8_1", 64),
        "mmq_q6_K_wave64" => ("flambeau_mmq_q6_K_wave64_q8_1", 64),
        "mmq_q8_0_wave64" => ("flambeau_mmq_q8_0_wave64_q8_1", 64),
        "mmq_q8_0_wave64_tile16" => ("flambeau_mmq_q8_0_wave64_tile16_q8_1", 64),
        other => bail!("unknown kernel {other}"),
    };

    // Variants that live in the same .cu as another stem must resolve hsaco
    // via the parent stem's filename, not their own.
    let dev = HipDevice::new(0)?;
    dev.bind()?;
    let bytes = kernels::hsaco(kernel_stem).context("hsaco")?;
    let module = HipModule::load(dev.id(), bytes)?;
    let kernel: HipKernel<'_> = module.kernel(entry)?;

    // Branch on kernel family. Each family has distinct layout / arg shape /
    // launch geometry; one `if is_mmq` enum isn't enough anymore.
    match kernel_stem {
        "mmq_q4_1_4warp_lds" => run_mmq_q4_1_4warp_lds(&dev, &kernel, m, k, n)?,
        "mmq_q4_K_wave64" | "mmq_q5_K_wave64" | "mmq_q6_K_wave64" => {
            run_mmq_k_wave64(&dev, &kernel, kernel_stem, m, k, n)?
        }
        "mmq_q8_0_wave64" => run_mmq_q8_0_wave64(&dev, &kernel, m, k, n, 8)?,
        "mmq_q8_0_wave64_tile16" => run_mmq_q8_0_wave64(&dev, &kernel, m, k, n, 16)?,
        "mmq_q8_0_oracle" | "mmq_q8_0_4warp" => {
            run_mmq_q8_0(&dev, &kernel, kernel_stem, threads, m, k, n)?
        }
        _ => run_mmvq(&dev, &kernel, kernel_stem, threads, m, k, n)?,
    }

    // Suppress unused-import warnings in release builds.
    let _ = &f16::from_f32;

    Ok(entry.to_string())
}

fn run_mmq_q8_0(
    dev: &HipDevice,
    kernel: &HipKernel<'_>,
    kernel_stem: &str,
    threads: u32,
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    let qk = QK8_0;
    assert!(k % qk == 0);
    let nb = k / qk;
    let w_blocks = n * nb;
    let w_bytes = w_blocks * std::mem::size_of::<BlockQ8_0>();
    let w_raw = vec![0u8; w_bytes];
    let d_x = dev.alloc(w_bytes)?;
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d_x,
            DevicePtr(w_raw.as_ptr() as usize),
            w_bytes,
        )?;
    }
    dev.default_stream().synchronize()?;

    let y_bytes = m * nb * std::mem::size_of::<BlockQ8_1>();
    let d_y = dev.alloc(y_bytes)?;
    let zero = vec![0u8; y_bytes];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d_y,
            DevicePtr(zero.as_ptr() as usize),
            y_bytes,
        )?;
    }
    dev.default_stream().synchronize()?;

    let d_dst = dev.alloc(m * n * 4)?;

    let (grid_x, grid_y) = if kernel_stem == "mmq_q8_0_4warp" {
        ((n as u32).div_ceil(32), (m as u32).div_ceil(8))
    } else {
        (n as u32, m as u32)
    };
    let d_x_ptr: u64 = d_x.as_usize() as u64;
    let d_y_ptr: u64 = d_y.as_usize() as u64;
    let d_dst_ptr: u64 = d_dst.as_usize() as u64;
    let n_rows_i = n as i32;
    let n_batches_i = m as i32;
    let nb_i = nb as i32;
    let mut args = KernelArgs::new();
    args.push(&d_x_ptr);
    args.push(&d_y_ptr);
    args.push(&d_dst_ptr);
    args.push(&n_rows_i);
    args.push(&n_batches_i);
    args.push(&nb_i);
    let cfg = LaunchCfg {
        grid: (grid_x, grid_y, 1),
        block: (threads, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(dev.default_stream(), cfg, args)? };
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_x, w_bytes)?;
        dev.dealloc(d_y, y_bytes)?;
        dev.dealloc(d_dst, m * n * 4)?;
    }
    Ok(())
}

/// Q4_1 4-warp LDS-tiled MMQ probe (candle port).
/// Args: (vx, vy, dst, ncols_x, nrows_x, ncols_y, stride_col_y, stride_row_x, nrows_dst)
/// Block = (64, 4, 1), shared = 30336 B.
/// Y layout: block_q8_1_mmq × (n_big_blocks_k, ncols_y), 144 B each, where
/// n_big_blocks_k = k / QK8_1_MMQ (QK8_1_MMQ = 128).
fn run_mmq_q4_1_4warp_lds(
    dev: &HipDevice,
    kernel: &HipKernel<'_>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    const QK4_1: usize = 32;
    assert!(k % QK8_1_MMQ == 0, "k={k} must be multiple of {QK8_1_MMQ}");
    let n_blocks_per_row = k / QK4_1; // weight row stride in Q4_1 blocks
    let n_big_blocks_k = k / QK8_1_MMQ; // Y row count in big blocks

    // Weights: Q4_1 * [n_rows=N, cols=n_blocks_per_row] row-major.
    let w_blocks = n * n_blocks_per_row;
    let w_bytes = w_blocks * std::mem::size_of::<BlockQ4_1>();
    let w_raw = vec![0u8; w_bytes];
    let d_x = dev.alloc(w_bytes)?;
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d_x,
            DevicePtr(w_raw.as_ptr() as usize),
            w_bytes,
        )?;
    }
    dev.default_stream().synchronize()?;

    // Y: block_q8_1_mmq * [n_big_blocks_k, ncols_y]. 144 B each.
    let y_blocks = n_big_blocks_k * m;
    let y_bytes = y_blocks * std::mem::size_of::<BlockQ8_1Mmq>();
    let d_y = dev.alloc(y_bytes)?;
    let zero_y = vec![0u8; y_bytes];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d_y,
            DevicePtr(zero_y.as_ptr() as usize),
            y_bytes,
        )?;
    }
    dev.default_stream().synchronize()?;

    let d_dst = dev.alloc(m * n * 4)?;

    let ncols_x = k as i32;
    let nrows_x = n as i32;
    let ncols_y = m as i32;
    let stride_col_y = m as i32; // ncols_y
    let stride_row_x = n_blocks_per_row as i32;
    let nrows_dst = n as i32;

    let d_x_ptr: u64 = d_x.as_usize() as u64;
    let d_y_ptr: u64 = d_y.as_usize() as u64;
    let d_dst_ptr: u64 = d_dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&d_x_ptr);
    args.push(&d_y_ptr);
    args.push(&d_dst_ptr);
    args.push(&ncols_x);
    args.push(&nrows_x);
    args.push(&ncols_y);
    args.push(&stride_col_y);
    args.push(&stride_row_x);
    args.push(&nrows_dst);

    const MMQ_Y: u32 = 128;
    const MMQ_X: u32 = 64;
    const SHARED_BYTES: u32 = 7584 * 4; // matches ops/src/hip/qmatmul.rs
    let grid_x = (n as u32).div_ceil(MMQ_Y);
    let grid_y = (m as u32).div_ceil(MMQ_X);
    let cfg = LaunchCfg {
        grid: (grid_x, grid_y, 1),
        block: (64, 4, 1),
        shared_bytes: SHARED_BYTES,
    };
    unsafe { kernel.launch(dev.default_stream(), cfg, args)? };
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_x, w_bytes)?;
        dev.dealloc(d_y, y_bytes)?;
        dev.dealloc(d_dst, m * n * 4)?;
    }
    Ok(())
}

/// / K-quant wave64 MMQ probe (candle ports). Q4_K and Q5_K
/// share launch shape; only weight block size differs.
/// Args: (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst)
/// Block = (64, 1, 1).
fn run_mmq_k_wave64(
    dev: &HipDevice,
    kernel: &HipKernel<'_>,
    kernel_stem: &str,
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    use flambeau_quant::{BlockQ4K, BlockQ5K, BlockQ6K, QK_K};
    const QK8_1: usize = 32;
    assert!(k % QK_K == 0, "k={k} must be multiple of {QK_K}");
    let weight_block_bytes = match kernel_stem {
        "mmq_q4_K_wave64" => std::mem::size_of::<BlockQ4K>(),
        "mmq_q5_K_wave64" => std::mem::size_of::<BlockQ5K>(),
        "mmq_q6_K_wave64" => std::mem::size_of::<BlockQ6K>(),
        other => bail!("unexpected K-quant wave64 stem {other}"),
    };
    let w_super_blocks = n * (k / QK_K);
    let w_bytes = w_super_blocks * weight_block_bytes;
    let w_raw = vec![0u8; w_bytes];
    let d_x = dev.alloc(w_bytes)?;
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d_x,
            DevicePtr(w_raw.as_ptr() as usize),
            w_bytes,
        )?;
    }
    dev.default_stream().synchronize()?;

    let y_blocks = m * (k / QK8_1);
    let y_bytes = y_blocks * std::mem::size_of::<BlockQ8_1>();
    let d_y = dev.alloc(y_bytes)?;
    let zero_y = vec![0u8; y_bytes];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d_y,
            DevicePtr(zero_y.as_ptr() as usize),
            y_bytes,
        )?;
    }
    dev.default_stream().synchronize()?;

    let d_dst = dev.alloc(m * n * 4)?;

    let ncols_x = k as i32;
    let nrows_x = n as i32;
    let ncols_y = m as i32;
    let nrows_y = k as i32;
    let nrows_dst = n as i32;

    let d_x_ptr: u64 = d_x.as_usize() as u64;
    let d_y_ptr: u64 = d_y.as_usize() as u64;
    let d_dst_ptr: u64 = d_dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&d_x_ptr);
    args.push(&d_y_ptr);
    args.push(&d_dst_ptr);
    args.push(&ncols_x);
    args.push(&nrows_x);
    args.push(&ncols_y);
    args.push(&nrows_y);
    args.push(&nrows_dst);

    const TILE_N: u32 = 8;
    let grid_x = (n as u32).div_ceil(64);
    let grid_y = (m as u32).div_ceil(TILE_N);
    let cfg = LaunchCfg {
        grid: (grid_x, grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(dev.default_stream(), cfg, args)? };
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_x, w_bytes)?;
        dev.dealloc(d_y, y_bytes)?;
        dev.dealloc(d_dst, m * n * 4)?;
    }
    Ok(())
}

fn run_mmvq(
    dev: &HipDevice,
    kernel: &HipKernel<'_>,
    kernel_stem: &str,
    threads: u32,
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    let _ = m;
    let qk = QK8_0;
    assert!(k % qk == 0);
    let nb = k / qk;
    let n_rows = n;
    let w_blocks = n_rows * nb;
    let w_bytes = w_blocks * block_size_of(kernel_stem)?;
    let d_x = dev.alloc(w_bytes)?;
    let w_raw = synthetic_weights(kernel_stem, w_blocks);
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d_x,
            DevicePtr(w_raw.as_ptr() as usize),
            w_bytes,
        )?;
    }
    dev.default_stream().synchronize()?;

    let y_bytes = nb * std::mem::size_of::<BlockQ8_1>();
    let d_y = dev.alloc(y_bytes)?;
    let zero = vec![0u8; y_bytes];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d_y,
            DevicePtr(zero.as_ptr() as usize),
            y_bytes,
        )?;
    }
    dev.default_stream().synchronize()?;

    let d_dst = dev.alloc(n_rows * 4)?;

    let rows_per_block: u32 = match kernel_stem {
        "mmvq_q4_k_r2" | "mmvq_q5_k_r2" => 2,
        "mmvq_q6_k_r4" => 4,
        _ => 1,
    };
    let grid_x = (n_rows as u32).div_ceil(rows_per_block);
    let d_x_ptr: u64 = d_x.as_usize() as u64;
    let d_y_ptr: u64 = d_y.as_usize() as u64;
    let d_dst_ptr: u64 = d_dst.as_usize() as u64;
    let n_rows_i = n_rows as i32;
    let n_bpr_i = nb as i32;
    let mut args = KernelArgs::new();
    args.push(&d_x_ptr);
    args.push(&d_y_ptr);
    args.push(&d_dst_ptr);
    args.push(&n_rows_i);
    args.push(&n_bpr_i);
    let cfg = LaunchCfg::one_d(grid_x, threads);
    unsafe { kernel.launch(dev.default_stream(), cfg, args)? };
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_x, w_bytes)?;
        dev.dealloc(d_y, y_bytes)?;
        dev.dealloc(d_dst, n_rows * 4)?;
    }
    Ok(())
}

fn block_size_of(kernel_stem: &str) -> Result<usize> {
    Ok(match kernel_stem {
        "mmvq_q8_0" => std::mem::size_of::<BlockQ8_0>(),
        "mmvq_q4_0" => std::mem::size_of::<flambeau_quant::BlockQ4_0>(),
        "mmvq_q4_k" | "mmvq_q4_k_r2" => std::mem::size_of::<flambeau_quant::BlockQ4K>(),
        "mmvq_q5_k" | "mmvq_q5_k_r2" => std::mem::size_of::<flambeau_quant::BlockQ5K>(),
        "mmvq_q6_k" | "mmvq_q6_k_r4" => std::mem::size_of::<flambeau_quant::BlockQ6K>(),
        other => bail!("block size unknown for {other}"),
    })
}

fn synthetic_weights(kernel_stem: &str, n_blocks: usize) -> Vec<u8> {
    // Fill with zeros for K-quants (the d/dmin fields need to be valid f16
    // with bit pattern 0x0000 which is +0.0 — safe). For Q8_0, 0x0000 is
    // also +0.0 for `d`. Kernel produces zero output; irrelevant to PMC.
    let _ = kernel_stem;
    vec![0u8; n_blocks * block_size_of(kernel_stem).unwrap_or(0)]
}

/// Q8_0 wave64 MMQ probe (separate from K-quant wave64 because
/// Q8_0 uses 34-byte-per-block layout with QK8_0=32, not QK_K=256).
fn run_mmq_q8_0_wave64(
    dev: &HipDevice,
    kernel: &HipKernel<'_>,
    m: usize,
    k: usize,
    n: usize,
    tile_n: u32,
) -> Result<()> {
    const QK8_0: usize = 32;
    const QK8_1: usize = 32;
    assert!(k % QK8_0 == 0, "k={k} must be multiple of {QK8_0}");

    // Weight: n rows × (k/QK8_0) Q8_0 blocks each.
    let w_blocks = n * (k / QK8_0);
    let w_bytes = w_blocks * std::mem::size_of::<BlockQ8_0>();
    let w_raw = vec![0u8; w_bytes];
    let d_x = dev.alloc(w_bytes)?;
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d_x,
            DevicePtr(w_raw.as_ptr() as usize),
            w_bytes,
        )?;
    }
    dev.default_stream().synchronize()?;

    // Activation: m cols × (k/QK8_1) Q8_1 blocks each.
    let y_blocks = m * (k / QK8_1);
    let y_bytes = y_blocks * std::mem::size_of::<BlockQ8_1>();
    let d_y = dev.alloc(y_bytes)?;
    let zero_y = vec![0u8; y_bytes];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d_y,
            DevicePtr(zero_y.as_ptr() as usize),
            y_bytes,
        )?;
    }
    dev.default_stream().synchronize()?;

    let d_dst = dev.alloc(m * n * 4)?;

    let ncols_x = k as i32;
    let nrows_x = n as i32;
    let ncols_y = m as i32;
    let nrows_y = k as i32;
    let nrows_dst = n as i32;

    let d_x_ptr: u64 = d_x.as_usize() as u64;
    let d_y_ptr: u64 = d_y.as_usize() as u64;
    let d_dst_ptr: u64 = d_dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&d_x_ptr);
    args.push(&d_y_ptr);
    args.push(&d_dst_ptr);
    args.push(&ncols_x);
    args.push(&nrows_x);
    args.push(&ncols_y);
    args.push(&nrows_y);
    args.push(&nrows_dst);

    let grid_x = (n as u32).div_ceil(64);
    let grid_y = (m as u32).div_ceil(tile_n);
    let cfg = LaunchCfg {
        grid: (grid_x, grid_y, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(dev.default_stream(), cfg, args)? };
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_x, w_bytes)?;
        dev.dealloc(d_y, y_bytes)?;
        dev.dealloc(d_dst, m * n * 4)?;
    }
    Ok(())
}
