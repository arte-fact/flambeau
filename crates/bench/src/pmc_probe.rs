//! One-shot kernel launcher used by `flambeau pmc-probe` under rocprofv3.
//! Runs exactly one kernel launch + one `hipStreamSynchronize` so the
//! profiler sees a clean single-dispatch trace.

#![cfg(feature = "hip")]

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::{HipDevice, HipKernel, HipModule, KernelArgs, LaunchCfg};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;
use flambeau_quant::{BlockQ8_0, BlockQ8_1, QK8_0};
use half::f16;

/// Launch a single `(kernel_stem, shape)` combination on device 0, return
/// the kernel entry name that was launched (useful so the PMC wrapper knows
/// what Kernel_Name string to filter by in rocprofv3's CSV).
pub fn run_one(kernel_stem: &str, m: usize, k: usize, n: usize) -> Result<String> {
    let (entry, threads) = match kernel_stem {
        "mmvq_q8_0" => ("flambeau_mmvq_q8_0_q8_1", 256u32),
        "mmvq_q4_k" => ("flambeau_mmvq_q4_k_q8_1", 64),
        "mmvq_q5_k" => ("flambeau_mmvq_q5_k_q8_1", 64),
        "mmvq_q6_k" => ("flambeau_mmvq_q6_k_q8_1", 64),
        "mmvq_q4_k_r2" => ("flambeau_mmvq_q4_k_r2_q8_1", 64),
        "mmvq_q5_k_r2" => ("flambeau_mmvq_q5_k_r2_q8_1", 64),
        "mmvq_q6_k_r4" => ("flambeau_mmvq_q6_k_r4_q8_1", 64),
        "mmq_q8_0_oracle" => ("flambeau_mmq_q8_0_oracle_q8_1", 256),
        "mmq_q8_0_4warp" => ("flambeau_mmq_q8_0_4warp_q8_1", 256),
        other => bail!("unknown kernel {other}"),
    };

    let dev = HipDevice::new(0)?;
    dev.bind()?;
    let bytes = kernels::hsaco(kernel_stem).context("hsaco")?;
    let module = HipModule::load(dev.id(), bytes)?;
    let kernel: HipKernel<'_> = module.kernel(entry)?;

    // Keep shapes small + inputs synthetic. The PMC wrapper captures the
    // *shape* of the counter response, not absolute timing quality — we just
    // need one representative launch.
    let is_mmq = kernel_stem.starts_with("mmq_");
    let qk = QK8_0;

    if is_mmq {
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

        // Activation: m * nb Q8_1 blocks, filled with zeros — the kernel just
        // needs valid pointers for the profiler to see one full dispatch.
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
            (((n as u32) + 31) / 32, ((m as u32) + 7) / 8)
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
    } else {
        // MMVQ shape — m is batch (always 1 for the probe), weights are
        // [rows=n, cols=k], activation is a single Q8_1 row.
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
        let grid_x = ((n_rows as u32) + rows_per_block - 1) / rows_per_block;
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
    }

    // Suppress unused-import warnings in release builds.
    let _ = &f16::from_f32;

    Ok(entry.to_string())
}

fn block_size_of(kernel_stem: &str) -> Result<usize> {
    Ok(match kernel_stem {
        "mmvq_q8_0" => std::mem::size_of::<BlockQ8_0>(),
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
