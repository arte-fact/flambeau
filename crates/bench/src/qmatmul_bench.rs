//! `flambeau qmatmul-bench`: launch QMatMul implementations over a grid of
//! (output rows, k) shapes and activation counts `m`, for a kernel profiler
//! (nsys `cuda_gpu_kern_sum`, rocprofv3 `--kernel-trace`) to time. Weights
//! and activations are zero: kernel time does not depend on the values. Both
//! activation layouts are passed; the line says which one the implementation
//! reads (its quantisation is timed apart). The printed wall time
//! per launch includes launch overhead; the profiler's kernel time is the
//! measurement.

use std::time::Instant;

use anyhow::{bail, Context, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr, KernelBackend, QDtype, Stream};
use flambeau_ops::qmatmul::{qmatmul_bench, qmatmul_bench_act_layout, QMATMUL_BENCH_PRODUCTION};
use flambeau_ops::ActLayout;
use flambeau_ops::{MatmulShape, QmatmulBuffers, Registry};

/// Activation columns the DS4 buffer is padded to (the widest MMQ tile).
const MMQ_COL_PAD: usize = 64;

pub struct QmatmulBenchSpec<'a> {
    pub dtype: &'a str,
    /// `(output rows, k)` pairs.
    pub shapes: &'a [(usize, usize)],
    pub ms: &'a [usize],
    /// `C0`..`C4` (see [`impl_id_for`]) or literal impl ids.
    pub impls: &'a [String],
    pub iters: usize,
}

/// Weight dtype and bytes per 256-element super-block.
fn weight_dtype(dtype: &str) -> Result<(QDtype, usize)> {
    use flambeau_quant::{BlockQ3K, BlockQ4K, BlockQ5K, BlockQ6K};
    Ok(match dtype {
        "Q3_K" => (QDtype::Q3_K, std::mem::size_of::<BlockQ3K>()),
        "Q4_K" => (QDtype::Q4_K, std::mem::size_of::<BlockQ4K>()),
        "Q5_K" => (QDtype::Q5_K, std::mem::size_of::<BlockQ5K>()),
        "Q6_K" => (QDtype::Q6_K, std::mem::size_of::<BlockQ6K>()),
        other => bail!("qmatmul-bench: --dtype {other} not wired (Q3_K / Q4_K / Q5_K / Q6_K)"),
    })
}

/// Candidate names of coord/mmq-few-plan.md: C0 the production path, C1..C4
/// the LDS-tiled MMQ at 8 / 16 / 64 columns and the wave64 MMQ, C5 the Q4_K
/// r4 MMVQ of the output head (HIP only).
pub fn impl_id_for(name: &str, dtype: &str) -> String {
    let d = dtype.to_ascii_lowercase();
    match name {
        "C0" => QMATMUL_BENCH_PRODUCTION.to_string(),
        "C1" => format!("bench:{d}_lds_x8"),
        "C2" => format!("bench:{d}_lds_x16"),
        "C3" => format!("bench:{d}_lds_x64"),
        "C4" => format!("bench:{d}_wave64"),
        "C5" => format!("bench:{d}_mmvq_r4"),
        other => other.to_string(),
    }
}

fn upload_zeros<D: Device>(dev: &D, bytes: usize) -> Result<DevicePtr> {
    let ptr = dev.alloc(bytes)?;
    let zeros = vec![0u8; bytes];
    // SAFETY: ptr owns `bytes`; `zeros` outlives the sync below.
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            ptr,
            DevicePtr(zeros.as_ptr() as usize),
            bytes,
        )?;
    }
    dev.default_stream().synchronize()?;
    Ok(ptr)
}

pub fn run<K: KernelBackend>(
    dev: &K::Device,
    reg: &Registry<K>,
    arch: &str,
    spec: &QmatmulBenchSpec<'_>,
) -> Result<()> {
    let (qdtype, block_bytes) = weight_dtype(spec.dtype)?;
    let m_max = spec.ms.iter().copied().max().context("qmatmul-bench: empty --m")?;
    let stream = dev.default_stream();
    println!(
        "qmatmul-bench {arch} {}: iters={} m={:?} impls={:?}",
        spec.dtype, spec.iters, spec.ms, spec.impls
    );
    for &(rows, k) in spec.shapes {
        if k % 256 != 0 {
            bail!("qmatmul-bench: k={k} is not a multiple of 256");
        }
        let w_bytes = rows * (k / 256) * block_bytes;
        let act_bytes = m_max * (k / 32) * 36;
        let mmq_bytes = k.div_ceil(128) * m_max.next_multiple_of(MMQ_COL_PAD) * 144;
        let dst_bytes = m_max.next_multiple_of(MMQ_COL_PAD) * rows * 4;
        let weights = upload_zeros(dev, w_bytes)?;
        let act = upload_zeros(dev, act_bytes)?;
        let act_mmq = upload_zeros(dev, mmq_bytes)?;
        let dst = upload_zeros(dev, dst_bytes)?;
        for &m in spec.ms {
            let shape = MatmulShape { m, k, n: rows };
            for name in spec.impls {
                let id = impl_id_for(name, spec.dtype);
                let layout = match qmatmul_bench_act_layout::<K>(&id, shape, qdtype) {
                    Ok(layout) => layout,
                    Err(e) => {
                        println!("  {name:<3} {id:<26} skipped: {e:#}");
                        continue;
                    }
                };
                let buf = QmatmulBuffers {
                    weights,
                    act_q8_1: act,
                    act_q8_1_mmq: act_mmq,
                    dst,
                };
                let launch = || qmatmul_bench::<K>(reg, stream, &id, buf, shape, qdtype);
                if let Err(e) = launch() {
                    println!("  {name:<3} {id:<26} rows={rows:<6} k={k:<6} m={m:<3} skipped: {e:#}");
                    continue;
                }
                for _ in 0..2 {
                    launch().with_context(|| format!("{id} rows={rows} k={k} m={m}"))?;
                }
                stream.synchronize()?;
                let t = Instant::now();
                for _ in 0..spec.iters {
                    launch()?;
                }
                stream.synchronize()?;
                let us = t.elapsed().as_secs_f64() * 1e6 / spec.iters as f64;
                println!(
                    "  {name:<3} {id:<26} rows={rows:<6} k={k:<6} m={m:<3} act={} wall_us/launch={us:8.1}",
                    match layout {
                        ActLayout::Mmq => "ds4",
                        ActLayout::Q8_1 => "q8_1",
                        ActLayout::Few(_) => "few",
                    }
                );
            }
        }
        // SAFETY: each pointer was allocated above with these sizes and no
        // launch is pending (synchronized).
        unsafe {
            dev.dealloc(weights, w_bytes)?;
            dev.dealloc(act, act_bytes)?;
            dev.dealloc(act_mmq, mmq_bytes)?;
            dev.dealloc(dst, dst_bytes)?;
        }
    }
    Ok(())
}

/// [`run`] on HIP device `device` (gfx906).
#[cfg(feature = "hip")]
pub fn run_hip(device: i32, spec: &QmatmulBenchSpec<'_>) -> Result<()> {
    use flambeau_backend_hip::{HipDevice, HipKernels};
    let dev = HipDevice::new(device).map_err(|e| anyhow::anyhow!("HipDevice {device}: {e:?}"))?;
    let reg = Registry::<HipKernels>::new(&dev).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    run(&dev, &reg, "gfx906", spec)
}

/// [`run`] on CUDA device `device` (sm_86).
#[cfg(feature = "cuda")]
pub fn run_cuda(device: i32, spec: &QmatmulBenchSpec<'_>) -> Result<()> {
    use flambeau_backend_cuda::{CudaDevice, CudaKernels};
    let dev = CudaDevice::new(device).map_err(|e| anyhow::anyhow!("CudaDevice {device}: {e:?}"))?;
    let reg = Registry::<CudaKernels>::new(&dev).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    run(&dev, &reg, "sm_86", spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_names_map_to_bench_ids() {
        assert_eq!(impl_id_for("C0", "Q5_K"), QMATMUL_BENCH_PRODUCTION);
        assert_eq!(impl_id_for("C1", "Q5_K"), "bench:q5_k_lds_x8");
        assert_eq!(impl_id_for("C4", "Q3_K"), "bench:q3_k_wave64");
        assert_eq!(impl_id_for("C5", "Q4_K"), "bench:q4_k_mmvq_r4");
        assert_eq!(impl_id_for("qmatmul_x", "Q4_K"), "qmatmul_x");
    }
}
