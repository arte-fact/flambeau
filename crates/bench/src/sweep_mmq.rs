//! MMQ correctness sweep.
//! MMQ = matrix × matrix (prefill path). Inputs:
//! weights: [N, K] Q-format blocks
//! act: [M, K] F32 → quantised to Q8_1 on device
//! output: [M, N] F32
//! The roadmap cert grid is `M ∈ {128, 512, 2048}` × Qwen3.6 attention /
//! MoE projection widths (K, N). This sweep runs a compact sub-grid; full
//! coverage lands with the first-class 4-warp LDS-tiled port next session.
//! Current impl: `qmatmul_q8_0_mmq_oracle_gfx906` — the `mmq_q8_0_oracle`
//! kernel (one output element per single-wave block). Slow but correct.

#![cfg(feature = "hip")]

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "sweep harness — every unsafe block is a kernel launch or a memcpy_async \
              over buffers allocated locally in the same function and freed before \
              return; invariant is uniform across all sites."
)]

use std::path::Path;

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::{
    device_count, FuncAttributes, HipDevice, HipKernel, HipModule, KernelArgs, LaunchCfg,
};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;
use flambeau_quant::{
    dequantize_into, BlockQ2K, BlockQ3K, BlockQ4_0, BlockQ4_1, BlockQ4K, BlockQ5_0, BlockQ5_1, BlockQ5K,
    BlockQ6K, BlockQ8K, BlockQ8_0, BlockQ8_1, GgmlDType, QK4_1, QK5_1, QK8_0, QK_K,
};
use half::f16;

use crate::cert::{now_utc_iso8601, Cert, PmcSnapshot, ShapeResult, SCHEMA_VERSION};
use crate::harness::{alloc_and_upload, max_rel_err_with_floor, rig, seeded_f32_range};

const QK8: usize = QK8_0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Q8_0Oracle,
    /// 4-warp LDS-tiled MMQ, MMQ_Y=32, MMQ_X=8, 256 threads.
    Q8_04Warp,
    /// Wave64 Q8_0 MMQ (). MMQ_Y=64, TILE_N=8, 64 threads.
    Q8_0Wave64,
    /// Wave64 Q8_0 MMQ TILE_N=16 (). MMQ_Y=64, TILE_N=16, 64 threads.
    /// Halves weight HBM fetches vs TILE_N=8 (each weight tile reused across
    /// 16 activations instead of 8). Targets the 97 % MemBusy bottleneck
    /// measured on Qwen3.6-35B Mesh<2> pp=512.
    Q8_0Wave64Tile16,
    /// 4-warp LDS-tiled Q4_1 MMQ, MMQ_Y=32, MMQ_X=8, 256 threads.
    Q4_14Warp,
    /// Wave64 Q4_1 MMQ (3.a). MMQ_Y=64, TILE_N=8, 64 threads. DP4A inner.
    Q4_1Wave64,
    /// Wave64 Q4_0 MMQ (8.a). MMQ_Y=64, TILE_N=8, 64 threads. DP4A inner
    /// with the `(q-8)·y = dp4a(q,y) - 8·y_s` bias-correction identity.
    Q4_0Wave64,
    /// 4-warp LDS-tiled Q4_0 MMQ. MMQ_Y=128, MMQ_X=64, 256 threads
    /// (4 warps × 64). Direct port of Q4_1 4warp_lds with bias-correction in
    /// the dot. Closes the 2.5× dense-prefill gap on 27B-Q4_0 vs Q4_1.
    Q4_04Warp,
    /// Wave64 Q5_0 MMQ (0.a). MMQ_Y=64, TILE_N=8, 64 threads. DP4A inner
    /// with the `(q5-16)·y = dp4a(nibble,y) + 16·dp4a(bit,y) - 16·y_s` identity
    /// (5th-bit side from 3 mmvq_q5_0 grafted into the 8.a tile shape).
    Q5_0Wave64,
    /// Wave64 Q5_1 MMQ. Combines the Q5_0 5th-bit ladder with the Q4_1
    /// `d·d_y·sumi + m·y_s` reduction; Q5_1 elements are unsigned so no
    /// −16·y_s correction.
    Q5_1Wave64,
    /// 4-warp LDS-tiled Q4_K MMQ, MMQ_Y=16, MMQ_X=8, 128 threads.
    Q4K4Warp,
    /// 4.b — llamacpp-turbo Q4_K MMQ port. 256 threads (4 warps × 64),
    /// MMQ_Y=16, MMQ_X=16, double-buffered Y LDS per super-block. DS4 Q8_1
    /// activation layout.
    Q4KTurbo,
    /// Wave64 Q4_K MMQ (, candle port). MMQ_Y=64, TILE_N=8, 64 threads.
    Q4KWave64,
    /// Wave64 Q5_K MMQ (, candle port). MMQ_Y=64, TILE_N=8, 64 threads.
    Q5KWave64,
    /// 4-warp LDS-tiled Q6_K MMQ, MMQ_Y=16, MMQ_X=8, 128 threads.
    Q6K4Warp,
    /// Wave64 Q6_K MMQ (flambeau-authored, no candle source).
    /// MMQ_Y=64, TILE_N=8, 64 threads. DP4A inner.
    Q6KWave64,
    /// Wave64 Q8_K MMQ. MMQ_Y=64, TILE_N=8, 64 threads. DP4A inner.
    /// Simplest K-quant MMQ: no nibble unpack, no sub-block scale, no min.
    Q8KWave64,
    /// Wave64 Q2_K MMQ. Affine quant; bias correction via -m·Σy.
    Q2KWave64,
    /// Wave64 Q3_K MMQ. Byte-wise u32 loads on qs/hmask/scales.
    Q3KWave64,
}

/// Shape of the output tile a single thread block produces for `dtype`.
/// `(rows_per_block, batches_per_block)` — the grid is
/// `(ceil(n / rows_per_block), ceil(m / batches_per_block))`.
fn tile_shape(dtype: Dtype) -> (u32, u32) {
    match dtype {
        Dtype::Q8_0Oracle => (1, 1),   // one output element per block
        Dtype::Q8_04Warp => (32, 8),   // MMQ_Y × MMQ_X
        Dtype::Q8_0Wave64 => (64, 8),  // MMQ_Y × TILE_N — wave64 ()
        Dtype::Q8_0Wave64Tile16 => (64, 16), // MMQ_Y × TILE_N — wave64 ()
        Dtype::Q4_14Warp => (128, 64), // 5 — candle turbo tile shape
        Dtype::Q4_1Wave64 => (64, 8),  // 3.a wave64 MMQ for Q4_1
        Dtype::Q4_0Wave64 => (64, 8),  // 8.a wave64 MMQ for Q4_0
        Dtype::Q4_04Warp => (128, 64), // C1 — 4warp_lds tile, MMQ_Y=128, MMQ_X=64
        Dtype::Q5_0Wave64 => (64, 8),  // 0.a wave64 MMQ for Q5_0
        Dtype::Q5_1Wave64 => (64, 8),
        Dtype::Q4K4Warp => (16, 8),    // MMQ_Y × MMQ_X (Q4_K uses 16 rows to fit LDS)
        Dtype::Q4KTurbo => (128, 16),  // 4.b turbo-ported Q4_K (MMQ_Y=128, MMQ_X=16)
        Dtype::Q4KWave64 => (64, 8),   // MMQ_Y × TILE_N — wave64 MMQ for Q4_K (candle port)
        Dtype::Q5KWave64 => (64, 8),   // MMQ_Y × TILE_N — wave64 MMQ for Q5_K (candle port)
        Dtype::Q6K4Warp => (16, 8),    // MMQ_Y × MMQ_X (Q6_K same tile as Q4_K)
        Dtype::Q6KWave64 => (64, 8),   // MMQ_Y × TILE_N — wave64 MMQ for Q6_K ()
        Dtype::Q8KWave64 => (64, 8),
        Dtype::Q2KWave64 => (64, 8),
        Dtype::Q3KWave64 => (64, 8),
    }
}

impl Dtype {
    pub fn name(self) -> &'static str {
        match self {
            Dtype::Q8_0Oracle
            | Dtype::Q8_04Warp
            | Dtype::Q8_0Wave64
            | Dtype::Q8_0Wave64Tile16 => "Q8_0",
            Dtype::Q4_14Warp | Dtype::Q4_1Wave64 => "Q4_1",
            Dtype::Q4_0Wave64 | Dtype::Q4_04Warp => "Q4_0",
            Dtype::Q5_0Wave64 => "Q5_0",
            Dtype::Q5_1Wave64 => "Q5_1",
            Dtype::Q4K4Warp | Dtype::Q4KWave64 | Dtype::Q4KTurbo => "Q4_K",
            Dtype::Q5KWave64 => "Q5_K",
            Dtype::Q6K4Warp | Dtype::Q6KWave64 => "Q6_K",
            Dtype::Q8KWave64 => "Q8_K",
            Dtype::Q2KWave64 => "Q2_K",
            Dtype::Q3KWave64 => "Q3_K",
        }
    }

    pub fn ggml(self) -> GgmlDType {
        match self {
            Dtype::Q8_0Oracle
            | Dtype::Q8_04Warp
            | Dtype::Q8_0Wave64
            | Dtype::Q8_0Wave64Tile16 => GgmlDType::Q8_0,
            Dtype::Q4_14Warp | Dtype::Q4_1Wave64 => GgmlDType::Q4_1,
            Dtype::Q4_0Wave64 | Dtype::Q4_04Warp => GgmlDType::Q4_0,
            Dtype::Q5_0Wave64 => GgmlDType::Q5_0,
            Dtype::Q5_1Wave64 => GgmlDType::Q5_1,
            Dtype::Q4K4Warp | Dtype::Q4KWave64 | Dtype::Q4KTurbo => GgmlDType::Q4K,
            Dtype::Q5KWave64 => GgmlDType::Q5K,
            Dtype::Q6K4Warp | Dtype::Q6KWave64 => GgmlDType::Q6K,
            Dtype::Q8KWave64 => GgmlDType::Q8K,
            Dtype::Q2KWave64 => GgmlDType::Q2K,
            Dtype::Q3KWave64 => GgmlDType::Q3K,
        }
    }

    fn impl_id(self) -> &'static str {
        match self {
            Dtype::Q8_0Oracle => "qmatmul_q8_0_mmq_oracle_gfx906",
            Dtype::Q8_04Warp => "qmatmul_q8_0_mmq_4warp_lds_gfx906",
            Dtype::Q8_0Wave64 => "qmatmul_q8_0_mmq_wave64_gfx906",
            Dtype::Q8_0Wave64Tile16 => "qmatmul_q8_0_mmq_wave64_tile16_gfx906",
            Dtype::Q4_14Warp => "qmatmul_q4_1_mmq_4warp_lds_gfx906",
            Dtype::Q4_1Wave64 => "qmatmul_q4_1_mmq_wave64_gfx906",
            Dtype::Q4_0Wave64 => "qmatmul_q4_0_mmq_wave64_gfx906",
            Dtype::Q4_04Warp => "qmatmul_q4_0_mmq_4warp_lds_gfx906",
            Dtype::Q5_0Wave64 => "qmatmul_q5_0_mmq_wave64_gfx906",
            Dtype::Q5_1Wave64 => "qmatmul_q5_1_mmq_wave64_gfx906",
            Dtype::Q4K4Warp => "qmatmul_q4_K_mmq_4warp_lds_gfx906",
            Dtype::Q4KTurbo => "qmatmul_q4_K_mmq_turbo_gfx906",
            Dtype::Q4KWave64 => "qmatmul_q4_K_mmq_wave64_gfx906",
            Dtype::Q5KWave64 => "qmatmul_q5_K_mmq_wave64_gfx906",
            Dtype::Q6K4Warp => "qmatmul_q6_K_mmq_4warp_lds_gfx906",
            Dtype::Q6KWave64 => "qmatmul_q6_K_mmq_wave64_gfx906",
            Dtype::Q8KWave64 => "qmatmul_q8_K_mmq_wave64_gfx906",
            Dtype::Q2KWave64 => "qmatmul_q2_K_mmq_wave64_gfx906",
            Dtype::Q3KWave64 => "qmatmul_q3_K_mmq_wave64_gfx906",
        }
    }

    fn kernel_stem(self) -> &'static str {
        match self {
            Dtype::Q8_0Oracle => "mmq_q8_0_oracle",
            Dtype::Q8_04Warp => "mmq_q8_0_4warp",
            Dtype::Q8_0Wave64 => "mmq_q8_0_wave64",
            Dtype::Q8_0Wave64Tile16 => "mmq_q8_0_wave64_tile16",
            Dtype::Q4_14Warp => "mmq_q4_1_4warp_lds",
            Dtype::Q4_1Wave64 => "mmq_q4_1_wave64",
            Dtype::Q4_0Wave64 => "mmq_q4_0_wave64",
            Dtype::Q4_04Warp => "mmq_q4_0_4warp_lds",
            Dtype::Q5_0Wave64 => "mmq_q5_0_wave64",
            Dtype::Q5_1Wave64 => "mmq_q5_1_wave64",
            Dtype::Q4K4Warp => "mmq_q4_K_4warp",
            Dtype::Q4KTurbo => "mmq_q4_K_turbo",
            Dtype::Q4KWave64 => "mmq_q4_K_wave64",
            Dtype::Q5KWave64 => "mmq_q5_K_wave64",
            Dtype::Q6K4Warp => "mmq_q6_K_4warp",
            Dtype::Q6KWave64 => "mmq_q6_K_wave64",
            Dtype::Q8KWave64 => "mmq_q8_K_wave64",
            Dtype::Q2KWave64 => "mmq_q2_K_wave64",
            Dtype::Q3KWave64 => "mmq_q3_K_wave64",
        }
    }

    fn kernel_entry(self) -> &'static str {
        match self {
            Dtype::Q8_0Oracle => "flambeau_mmq_q8_0_oracle_q8_1",
            Dtype::Q8_04Warp => "flambeau_mmq_q8_0_4warp_q8_1",
            Dtype::Q8_0Wave64 => "flambeau_mmq_q8_0_wave64_q8_1",
            Dtype::Q8_0Wave64Tile16 => "flambeau_mmq_q8_0_wave64_tile16_q8_1",
            Dtype::Q4_14Warp => "flambeau_mmq_q4_1_4warp_lds_q8_1",
            Dtype::Q4_1Wave64 => "flambeau_mmq_q4_1_wave64_q8_1",
            Dtype::Q4_0Wave64 => "flambeau_mmq_q4_0_wave64_q8_1",
            Dtype::Q4_04Warp => "flambeau_mmq_q4_0_4warp_lds_q8_1",
            Dtype::Q5_0Wave64 => "flambeau_mmq_q5_0_wave64_q8_1",
            Dtype::Q5_1Wave64 => "flambeau_mmq_q5_1_wave64_q8_1",
            Dtype::Q4K4Warp => "flambeau_mmq_q4_K_4warp_q8_1",
            Dtype::Q4KTurbo => "flambeau_mmq_q4_K_turbo_q8_1",
            Dtype::Q4KWave64 => "flambeau_mmq_q4_K_wave64_q8_1",
            Dtype::Q5KWave64 => "flambeau_mmq_q5_K_wave64_q8_1",
            Dtype::Q6K4Warp => "flambeau_mmq_q6_K_4warp_q8_1",
            Dtype::Q6KWave64 => "flambeau_mmq_q6_K_wave64_q8_1",
            Dtype::Q8KWave64 => "flambeau_mmq_q8_K_wave64_q8_1",
            Dtype::Q2KWave64 => "flambeau_mmq_q2_K_wave64_q8_1",
            Dtype::Q3KWave64 => "flambeau_mmq_q3_K_wave64_q8_1",
        }
    }

    fn launch_threads(self) -> u32 {
        match self {
            Dtype::Q8_0Oracle | Dtype::Q8_04Warp | Dtype::Q4_14Warp | Dtype::Q4_04Warp => 256,
            Dtype::Q8_0Wave64 | Dtype::Q8_0Wave64Tile16 | Dtype::Q4_1Wave64 | Dtype::Q4_0Wave64 | Dtype::Q5_0Wave64 | Dtype::Q5_1Wave64 => 64,
            Dtype::Q4K4Warp | Dtype::Q6K4Warp => 128,
            Dtype::Q4KTurbo => 256,  // 4 warps × 64
            Dtype::Q4KWave64 | Dtype::Q5KWave64 | Dtype::Q6KWave64 | Dtype::Q8KWave64 | Dtype::Q2KWave64 | Dtype::Q3KWave64 => 64,
        }
    }

    fn block_elem_count(self) -> usize {
        self.ggml().block_size()
    }

    fn weight_block_bytes(self) -> usize {
        match self {
            Dtype::Q8_0Oracle
            | Dtype::Q8_04Warp
            | Dtype::Q8_0Wave64
            | Dtype::Q8_0Wave64Tile16 => std::mem::size_of::<BlockQ8_0>(),
            Dtype::Q4_14Warp | Dtype::Q4_1Wave64 => std::mem::size_of::<BlockQ4_1>(),
            Dtype::Q4_0Wave64 | Dtype::Q4_04Warp => std::mem::size_of::<BlockQ4_0>(),
            Dtype::Q5_0Wave64 => std::mem::size_of::<BlockQ5_0>(),
            Dtype::Q5_1Wave64 => std::mem::size_of::<BlockQ5_1>(),
            Dtype::Q4K4Warp | Dtype::Q4KWave64 | Dtype::Q4KTurbo => std::mem::size_of::<BlockQ4K>(),
            Dtype::Q5KWave64 => std::mem::size_of::<BlockQ5K>(),
            Dtype::Q6K4Warp | Dtype::Q6KWave64 => std::mem::size_of::<BlockQ6K>(),
            Dtype::Q8KWave64 => std::mem::size_of::<BlockQ8K>(),
            Dtype::Q2KWave64 => std::mem::size_of::<BlockQ2K>(),
            Dtype::Q3KWave64 => std::mem::size_of::<BlockQ3K>(),
        }
    }

    /// Short tag for the CLI to log alongside the dtype.
    pub fn impl_id_short(self) -> &'static str {
        match self {
            Dtype::Q8_0Oracle => "oracle",
            Dtype::Q8_04Warp => "4warp_lds",
            Dtype::Q4_14Warp => "4warp_lds",
            Dtype::Q4_1Wave64 => "wave64",
            Dtype::Q4_0Wave64 => "wave64",
            Dtype::Q4_04Warp => "4warp_lds",
            Dtype::Q5_0Wave64 => "wave64",
            Dtype::Q5_1Wave64 => "wave64",
            Dtype::Q4K4Warp => "4warp_lds",
            Dtype::Q4KTurbo => "turbo",
            Dtype::Q4KWave64 | Dtype::Q5KWave64 | Dtype::Q6KWave64 | Dtype::Q8KWave64 | Dtype::Q2KWave64 | Dtype::Q3KWave64 | Dtype::Q8_0Wave64 => "wave64",
            Dtype::Q8_0Wave64Tile16 => "wave64_tile16",
            Dtype::Q6K4Warp => "4warp_lds",
        }
    }
}

/// One shape in the grid. `m` = batch rows, `k` = contracted dim, `n` = output features.
#[derive(Debug, Clone, Copy)]
pub struct Shape {
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

#[derive(Debug, Clone)]
pub struct SweepSpec {
    pub dtype: Dtype,
    pub shapes: Vec<Shape>,
    pub seed: u64,
}

impl SweepSpec {
    /// oracle grid — small enough that the single-wave-per-output kernel
    /// finishes in a reasonable wall-clock. Once the 4-warp LDS-tiled kernel
    /// lands, the grid grows to `M ∈ {128, 512, 2048}` × full K × N.
    pub fn v1_4_oracle(dtype: Dtype) -> Self {
        // K, N chosen from Qwen3.6 attention + FFN widths; M small so the
        // oracle's M×N single-wave launches stay under a few seconds total.
        let shapes = vec![
            Shape { m: 4,  k: 2048, n: 2048 },
            Shape { m: 8,  k: 2048, n: 2048 },
            Shape { m: 16, k: 2048, n: 2048 },
            Shape { m: 4,  k: 5120, n: 5120 },
            Shape { m: 8,  k: 5120, n: 5120 },
            Shape { m: 4,  k: 5120, n: 15360 }, // FFN up
            Shape { m: 4,  k: 15360, n: 5120 }, // FFN down
        ];
        Self { dtype, shapes, seed: 0xC0FFEE }
    }

    /// Prefill-scale grid for the first-class 4-warp LDS-tiled kernel.
    /// Matches the roadmap's `M ∈ {128, 512, 2048}` × Qwen3.6 attention /
    /// MoE widths.
    pub fn v1_4_prefill(dtype: Dtype) -> Self {
        let shapes = vec![
            Shape { m: 128,  k: 2048, n: 2048 },
            Shape { m: 128,  k: 5120, n: 5120 },
            Shape { m: 128,  k: 5120, n: 15360 },
            Shape { m: 128,  k: 15360, n: 5120 },
            Shape { m: 512,  k: 2048, n: 2048 },
            Shape { m: 512,  k: 5120, n: 5120 },
            Shape { m: 2048, k: 2048, n: 2048 },
        ];
        Self { dtype, shapes, seed: 0xC0FFEE }
    }
}

pub fn run_sweep(spec: &SweepSpec, repo_root: &Path) -> Result<Cert> {
    let n = device_count().context("hipGetDeviceCount")?;
    if n < 1 {
        bail!("no HIP devices on this host — sweep needs gfx906");
    }

    let dev = HipDevice::new(0)?;
    dev.bind()?;

    let pmc = capture_static_pmc(&dev, spec.dtype)?;

    let mut results = Vec::new();
    for sh in &spec.shapes {
        let seed = spec
            .seed
            .wrapping_add((sh.m as u64).wrapping_mul(0x1234567))
            .wrapping_add((sh.k as u64).wrapping_mul(0x9E3779B97F4A7C15))
            .wrapping_add((sh.n as u64).wrapping_mul(0xDEADBEEFCAFEBABE));
        let (got, reference) = run_shape(&dev, spec.dtype, sh.m, sh.k, sh.n, seed)?;
        let max_rel_err = max_rel_err_with_floor(&got, &reference, (sh.k as f32).sqrt());
        let tolerance = cert_tol(spec.dtype, sh.k);
        results.push(ShapeResult {
            m: sh.m,
            k: sh.k,
            n: sh.n,
            seed,
            max_rel_err,
            tolerance,
            pass: max_rel_err <= tolerance,
        });
        tracing::info!(
            target: "flambeau_bench::sweep_mmq",
            dtype = spec.dtype.name(),
            m = sh.m, k = sh.k, n = sh.n,
            max_rel_err, tolerance,
            "shape result"
        );
    }

    let pass = results.iter().all(|r| r.pass);
    let rig = rig();

    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: spec.dtype.impl_id().to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "qmatmul_mmq".to_string(),
        dtype_weight: spec.dtype.name().to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula: match spec.dtype {
            Dtype::Q4KWave64 | Dtype::Q5KWave64 | Dtype::Q6KWave64 | Dtype::Q8KWave64 | Dtype::Q2KWave64 | Dtype::Q3KWave64 => "|err| <= 5e-2 * max(|ref|, sqrt(k))".to_string(),
            _ => "|err| <= 3e-2 * max(|ref|, sqrt(k))".to_string(),
        },
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig,
        pmc: Some(pmc),
    };

    let written = cert.write_to_disk(repo_root)?;
    tracing::info!(
        target: "flambeau_bench::sweep_mmq",
        cert = %written.display(),
        pass = cert.pass,
        "cert written"
    );
    Ok(cert)
}

fn capture_static_pmc(dev: &HipDevice, dtype: Dtype) -> Result<PmcSnapshot> {
    let bytes = kernels::hsaco(dtype.kernel_stem())
        .ok_or_else(|| anyhow::anyhow!("{} kernel not compiled", dtype.kernel_stem()))?;
    let module = HipModule::load(dev.id(), bytes)?;
    let kernel: HipKernel<'_> = module.kernel(dtype.kernel_entry())?;
    let attrs: FuncAttributes = kernel.attributes()?;
    Ok(PmcSnapshot {
        vgpr_count: Some(attrs.num_regs),
        sgpr_count: None,
        waves_per_simd: Some(attrs.gfx906_waves_per_simd()),
        mem_busy_pct: None,
        valu_busy_pct: None,
    })
}

fn run_shape(
    dev: &HipDevice,
    dtype: Dtype,
    m: usize,
    k: usize,
    n: usize,
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>)> {
    assert_eq!(k % dtype.block_elem_count(), 0);
    let nb_per_row = k / dtype.block_elem_count();

    // Q4_1 MMQ uses the DS4 / BlockQ8_1Mmq activation layout (series
    // port of candle). Every other MMQ variant still uses the per-row
    // BlockQ8_1 layout. The fork happens here.
    let uses_mmq_layout = dtype == Dtype::Q4_14Warp
        || dtype == Dtype::Q4_04Warp
        || dtype == Dtype::Q4KTurbo;

    // Kernels.
    let q_stem = if uses_mmq_layout { "quantize_q8_1_mmq" } else { "quantize_q8_1" };
    let q_entry = if uses_mmq_layout { "flambeau_quantize_q8_1_mmq" } else { "flambeau_quantize_row_q8_1" };
    let q_bytes = kernels::hsaco(q_stem).unwrap();
    let m_bytes = kernels::hsaco(dtype.kernel_stem()).unwrap();
    let q_module = HipModule::load(dev.id(), q_bytes)?;
    let m_module = HipModule::load(dev.id(), m_bytes)?;
    let k_quantize: HipKernel<'_> = q_module.kernel(q_entry)?;
    let k_mmq: HipKernel<'_> = m_module.kernel(dtype.kernel_entry())?;

    // Weights: [n, k] weight blocks (size depends on dtype).
    let weights_raw = seeded_bytes(seed, n * nb_per_row * dtype.weight_block_bytes());
    let weights_raw = tame_weight_scales(dtype, weights_raw);

    // Activation: [m, k] F32.
    let act_f32 = seeded_f32_range(seed.wrapping_add(0xA5A5A5A5), m * k, -1.0, 1.0);

    // CPU dequantise weights for the reference.
    let mut w_dequant = vec![0.0f32; n * k];
    dequantize_into(dtype.ggml(), &weights_raw, &mut w_dequant)?;

    // Device allocs + upload.
    let d_x = alloc_and_upload(dev, weights_raw.as_slice());
    let d_act = alloc_and_upload(dev, act_f32.as_slice());

    // Allocate activation Q8_1 buffer sized for whichever layout we use.
    let (d_y_q8_1, y_bytes_total) = if uses_mmq_layout {
        // BlockQ8_1Mmq: 128 elements per block, 144 B each.
        // Layout: (k_big_blocks, m) row-major. k_big_blocks = k / 128.
        assert_eq!(k % 128, 0);
        let k_big_blocks = k / 128;
        let total = k_big_blocks * m;
        let bytes = total * std::mem::size_of::<flambeau_quant::BlockQ8_1Mmq>();
        (dev.alloc(bytes)?, bytes)
    } else {
        assert_eq!(k % QK8, 0);
        let total = m * k / QK8;
        let bytes = total * std::mem::size_of::<BlockQ8_1>();
        (dev.alloc(bytes)?, bytes)
    };

    // Quantise.
    {
        let stream = dev.default_stream();
        let d_act_ptr: u64 = d_act.as_usize() as u64;
        let d_y_q8_1_ptr: u64 = d_y_q8_1.as_usize() as u64;
        if uses_mmq_layout {
            // flambeau_quantize_q8_1_mmq(x, vy, ncols, total_b).
            // grid = (k/128, m), block = 128.
            let ncols_i = k as i32;
            let total_b_i = m as i32;
            let mut args = KernelArgs::new();
            args.push(&d_act_ptr);
            args.push(&d_y_q8_1_ptr);
            args.push(&ncols_i);
            args.push(&total_b_i);
            let cfg = LaunchCfg {
                grid: ((k / 128) as u32, m as u32, 1),
                block: (128, 1, 1),
                shared_bytes: 0,
            };
            unsafe { k_quantize.launch(stream, cfg, args)? };
        } else {
            // flambeau_quantize_row_q8_1(x, vy, n_elems). Grid = total_blocks.
            let total_y_blocks = (m * k / QK8) as u32;
            let n_elems = (m * k) as i32;
            let mut args = KernelArgs::new();
            args.push(&d_act_ptr);
            args.push(&d_y_q8_1_ptr);
            args.push(&n_elems);
            let cfg = LaunchCfg::one_d(total_y_blocks, QK8 as u32);
            unsafe { k_quantize.launch(stream, cfg, args)? };
        }
        stream.synchronize()?;
    }

    // MMQ launch.
    let d_dst = dev.alloc(m * n * 4)?;
    {
        let stream = dev.default_stream();
        let d_x_ptr: u64 = d_x.as_usize() as u64;
        let d_y_q8_1_ptr: u64 = d_y_q8_1.as_usize() as u64;
        let d_dst_ptr: u64 = d_dst.as_usize() as u64;

        if uses_mmq_layout {
            // flambeau_mmq_q4_1_4warp_lds_q8_1 / flambeau_mmq_q4_K_turbo_q8_1
            // take 9 scalar args + 3 ptrs and a 2D block (64, 4, 1) with
            // dynamic LDS.
            let ncols_x = k as i32;
            let nrows_x = n as i32;
            let ncols_y = m as i32;
            let stride_col_y = m as i32;
            let stride_row_x = nb_per_row as i32;
            let nrows_dst = n as i32;
            let mut args = KernelArgs::new();
            args.push(&d_x_ptr);
            args.push(&d_y_q8_1_ptr);
            args.push(&d_dst_ptr);
            args.push(&ncols_x);
            args.push(&nrows_x);
            args.push(&ncols_y);
            args.push(&stride_col_y);
            args.push(&stride_row_x);
            args.push(&nrows_dst);

            let (rows_per_block, batches_per_block) = tile_shape(dtype);
            let grid_x = (n as u32).div_ceil(rows_per_block);
            let grid_y = (m as u32).div_ceil(batches_per_block);
            // LDS budget depends on kernel:
            // Q4_1 4warp_lds: 7584*4 = 30336 B (matches mmq_q4_1_4warp_lds.cu)
            // Q4_K turbo: tile_y (MMQ_X × 36) + tile_x_qs (MMQ_Y × (32+1))
            // + tile_x_dm (MMQ_Y half2) + tile_x_sc (MMQ_Y×4 + MMQ_Y/8) ints
            // = 576 + 4224 + 128 + 528 = 5456 ints = 21824 B (mmq_y=128)
            let shared_bytes: u32 = match dtype {
                Dtype::Q4_14Warp | Dtype::Q4_04Warp => 7584 * 4,  // same MMQ_Y/MMQ_X tile
                Dtype::Q4KTurbo  => 22528,  // 22 KiB, rounded up
                _ => 0,  // unreachable (uses_mmq_layout gate above)
            };
            let cfg = LaunchCfg {
                grid: (grid_x, grid_y, 1),
                block: (64, 4, 1),
                shared_bytes,
            };
            unsafe { k_mmq.launch(stream, cfg, args)? };
        } else if dtype == Dtype::Q4KWave64 || dtype == Dtype::Q5KWave64 || dtype == Dtype::Q6KWave64 || dtype == Dtype::Q8KWave64 || dtype == Dtype::Q2KWave64 || dtype == Dtype::Q3KWave64 || dtype == Dtype::Q8_0Wave64 || dtype == Dtype::Q8_0Wave64Tile16 || dtype == Dtype::Q4_1Wave64 || dtype == Dtype::Q4_0Wave64 || dtype == Dtype::Q5_0Wave64 || dtype == Dtype::Q5_1Wave64 {
            // flambeau_mmq_{q4_K,q5_K,q6_K,q8_0}_wave64_q8_1: 8 scalar args + 3 ptrs, wave64
            // block. Args: vx, vy, dst, ncols_x=K, nrows_x=N, ncols_y=M,
            // nrows_y=K, nrows_dst=N.
            let ncols_x = k as i32;
            let nrows_x = n as i32;
            let ncols_y = m as i32;
            let nrows_y = k as i32;
            let nrows_dst = n as i32;
            let mut args = KernelArgs::new();
            args.push(&d_x_ptr);
            args.push(&d_y_q8_1_ptr);
            args.push(&d_dst_ptr);
            args.push(&ncols_x);
            args.push(&nrows_x);
            args.push(&ncols_y);
            args.push(&nrows_y);
            args.push(&nrows_dst);
            let (rows_per_block, batches_per_block) = tile_shape(dtype);
            let grid_x = (n as u32).div_ceil(rows_per_block);
            let grid_y = (m as u32).div_ceil(batches_per_block);
            let cfg = LaunchCfg {
                grid: (grid_x, grid_y, 1),
                block: (dtype.launch_threads(), 1, 1),
                shared_bytes: 0,
            };
            unsafe { k_mmq.launch(stream, cfg, args)? };
        } else {
            let n_rows_i = n as i32;
            let n_batches_i = m as i32;
            let n_bpr_i = nb_per_row as i32;
            let mut args = KernelArgs::new();
            args.push(&d_x_ptr);
            args.push(&d_y_q8_1_ptr);
            args.push(&d_dst_ptr);
            args.push(&n_rows_i);
            args.push(&n_batches_i);
            args.push(&n_bpr_i);
            let (rows_per_block, batches_per_block) = tile_shape(dtype);
            let grid_x = (n as u32).div_ceil(rows_per_block);
            let grid_y = (m as u32).div_ceil(batches_per_block);
            let cfg = LaunchCfg {
                grid: (grid_x, grid_y, 1),
                block: (dtype.launch_threads(), 1, 1),
                shared_bytes: 0,
            };
            unsafe { k_mmq.launch(stream, cfg, args)? };
        }
        stream.synchronize()?;
    }

    let _ = y_bytes_total;

    // Download result [m, n].
    let mut dst = vec![0.0f32; m * n];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(dst.as_mut_ptr() as usize),
            d_dst,
            m * n * 4,
        )?;
    }
    dev.default_stream().synchronize()?;

    unsafe {
        dev.dealloc(d_x, weights_raw.len())?;
        dev.dealloc(d_act, act_f32.len() * 4)?;
        dev.dealloc(d_y_q8_1, y_bytes_total)?;
        dev.dealloc(d_dst, m * n * 4)?;
    }

    // Reference: dequant_weights × (Q8_1-round-trip activation). Shape [m, n].
    let act_rt = quantize_q8_1_roundtrip(&act_f32);
    let reference = reference_matmul(&w_dequant, &act_rt, m, k, n);
    Ok((dst, reference))
}

// ---- helpers (mostly copied from sweep_mmvq; dedupe into a util module
// once more than one sweep lands) ----

fn seeded_bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 24) as u8
        })
        .collect()
}

fn tame_weight_scales(dtype: Dtype, mut raw: Vec<u8>) -> Vec<u8> {
    let bs = dtype.weight_block_bytes();
    let nblocks = raw.len() / bs;
    for i in 0..nblocks {
        let block = &mut raw[i * bs..(i + 1) * bs];
        match dtype {
            Dtype::Q8_0Oracle
            | Dtype::Q8_04Warp
            | Dtype::Q8_0Wave64
            | Dtype::Q8_0Wave64Tile16 => {
                let d = f16::from_f32((block[0] as f32 / 255.0) * 0.1 + 0.01);
                block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
            }
            Dtype::Q4_14Warp | Dtype::Q4_1Wave64 => {
                // Q4_1 block: d(fp16), m(fp16), qs[16]. Tame both scales so
                // the reference stays well-conditioned and the bias term
                // (m * s_y) doesn't dominate the Q4_1 × Q8_1 dot.
                let d = f16::from_f32((block[0] as f32 / 255.0) * 0.1 + 0.01);
                let m = f16::from_f32((block[1] as f32 / 255.0) * 0.05 - 0.025);
                block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
                block[2..4].copy_from_slice(&m.to_bits().to_le_bytes());
                let _ = QK4_1;
            }
            Dtype::Q4_0Wave64 | Dtype::Q4_04Warp => {
                // Q4_0 block: d(fp16), qs[16]. No `m`. Tame d so the reference
                // stays well-conditioned (the `-8·d·y_s` bias term is bounded
                // by d alone since Q4_0 is symmetric quant).
                let d = f16::from_f32((block[0] as f32 / 255.0) * 0.1 + 0.01);
                block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
            }
            Dtype::Q5_0Wave64 => {
                // Q5_0 block: d(fp16), qh[4], qs[16]. Tame d only. 5th-bit qh
                // field is raw random bytes from the seed — exercises the
                // expand_bits4 DP4A path across all 32 bits naturally.
                let d = f16::from_f32((block[0] as f32 / 255.0) * 0.1 + 0.01);
                block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
            }
            Dtype::Q5_1Wave64 => {
                // Q5_1 block: d(fp16), m(fp16), qh[4], qs[16]. Tame d + m
                // for the same reasons as Q4_1.
                let d = f16::from_f32((block[0] as f32 / 255.0) * 0.1 + 0.01);
                let m = f16::from_f32((block[1] as f32 / 255.0) * 0.05 - 0.025);
                block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
                block[2..4].copy_from_slice(&m.to_bits().to_le_bytes());
                let _ = QK5_1;
            }
            Dtype::Q4K4Warp | Dtype::Q4KWave64 | Dtype::Q4KTurbo => {
                // Q4_K: d(fp16), dmin(fp16), scales[12], qs[128]. Tame d/dmin
                // so the reference stays well-conditioned (same shaping the
                // MMVQ sweep uses).
                let d = f16::from_f32((block[0] as f32 / 255.0) * 0.1 + 0.01);
                let dmin = f16::from_f32((block[1] as f32 / 255.0) * 0.05);
                block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
                block[2..4].copy_from_slice(&dmin.to_bits().to_le_bytes());
            }
            Dtype::Q5KWave64 => {
                // Q5_K: d(fp16), dmin(fp16), scales[12], qh[32], qs[128].
                // Same scale taming shape as Q4_K — d/dmin at offset 0..4.
                let d = f16::from_f32((block[0] as f32 / 255.0) * 0.1 + 0.01);
                let dmin = f16::from_f32((block[1] as f32 / 255.0) * 0.05);
                block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
                block[2..4].copy_from_slice(&dmin.to_bits().to_le_bytes());
            }
            Dtype::Q6K4Warp | Dtype::Q6KWave64 => {
                // Q6_K: ql[128], qh[64], scales[16] (i8), d(fp16) at offset 208.
                // Clamp scales into a small signed range and set d modest.
                for s in 0..16 {
                    let raw = block[192 + s] as i8;
                    // Scale down into ±32 so scale × (raw_q ∈ -32..31) stays
                    // well-behaved before d applies.
                    block[192 + s] = ((raw as i32) >> 2) as u8;
                }
                let d = f16::from_f32((block[208] as f32 / 255.0) * 0.05 + 0.005);
                block[208..210].copy_from_slice(&d.to_bits().to_le_bytes());
            }
            Dtype::Q8KWave64 => {
                // Q8_K: d(f32) at offset 0, qs[256] i8, bsums[16] i16. Tame d.
                let d = (block[0] as f32 / 255.0) * 0.02 + 0.002;
                block[0..4].copy_from_slice(&d.to_le_bytes());
            }
            Dtype::Q2KWave64 => {
                // Q2_K: scales[16] + qs[64] + d (f16 @ 80) + dmin (f16 @ 82).
                let d = f16::from_f32((block[80] as f32 / 255.0) * 0.05 + 0.005);
                let dmin = f16::from_f32((block[81] as f32 / 255.0) * 0.02);
                block[80..82].copy_from_slice(&d.to_bits().to_le_bytes());
                block[82..84].copy_from_slice(&dmin.to_bits().to_le_bytes());
            }
            Dtype::Q3KWave64 => {
                // Q3_K: hmask[32] + qs[64] + scales[12] + d (f16 @ 108).
                // Scales bytes are signed 6-bit packed; mod-32 keeps them
                // in-range. d tamed small.
                let d_off = QK_K / 8 + QK_K / 4 + 12;
                for s in &mut block[(QK_K / 8 + QK_K / 4)..(QK_K / 8 + QK_K / 4 + 12)] {
                    *s = (*s as i32 % 32) as u8;
                }
                let d = f16::from_f32((block[d_off] as f32 / 255.0) * 0.05 + 0.005);
                block[d_off..d_off + 2].copy_from_slice(&d.to_bits().to_le_bytes());
            }
        }
    }
    let _ = QK_K;
    raw
}

fn quantize_q8_1_roundtrip(xs: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; xs.len()];
    for i in 0..(xs.len() / QK8) {
        let block = &xs[i * QK8..(i + 1) * QK8];
        let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        for (j, &v) in block.iter().enumerate() {
            let q = (v * id).round().clamp(-127.0, 127.0) as i32;
            out[i * QK8 + j] = (q as f32) * d;
        }
    }
    out
}

fn reference_matmul(weights: &[f32], act: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for b in 0..m {
        let act_row = &act[b * k..(b + 1) * k];
        for row in 0..n {
            let w_row = &weights[row * k..(row + 1) * k];
            let mut acc = 0.0f64;
            for j in 0..k {
                acc += (w_row[j] * act_row[j]) as f64;
            }
            out[b * n + row] = acc as f32;
        }
    }
    out
}

fn cert_tol(dtype: Dtype, _k: usize) -> f32 {
    match dtype {
        // Q5_K has compound super-block × sub-block scale errors; at large M the
        // extreme-value distribution of per-element errors crosses the 3e-2 bar
        // even for a correct kernel. Candle's Q5_K MMQ accepts ~5e-2 at m=2048.
        // Q4_K wave64 shares the same super/sub-block decomposition as Q5_K.
        Dtype::Q4KWave64 | Dtype::Q5KWave64 | Dtype::Q6KWave64 | Dtype::Q4KTurbo => 5e-2,
        _ => 3e-2,
    }
}

