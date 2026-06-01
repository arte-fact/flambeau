//! kernel-launched BAR1 P2P AllReduce, wired against [`HipCluster`].
//! `BarP2pAllReduce` is the device-side AllReduce primitive used on the TP
//! decode hot path. Each call launches one
//! `flambeau_p2p_allreduce_*_tp{2,4}` kernel per rank, simultaneously, on
//! per-rank streams. The kernels read peer GPUs' partial buffers directly
//! through PCIe BAR1 (authorised at `HipCluster::new` time, see
//! `cluster::probe_and_enable_peer_access`).
//! Two operating modes:
//! - **residual** (`residual_tpN`) — `hidden += partial_local + Σ peers`
//! on each rank. Used after attention output proj and FFN down proj
//! where the sharded result must be folded into the residual stream.
//! - **sum** (`sum_tpN`) — `partial_local += Σ peers` (no residual add).
//! Used at the LM-head output-AR boundary where the post-AR value
//! *is* the final result, no residual to fold.
//! ## Ordering contract (caller responsibility)
//! The kernel reads peer partials at launch time. The producer GEMV that
//! wrote each rank's `partial[r]` must have completed *and been visible
//! to peer ranks' streams* before this AllReduce launches. The standard
//! pattern is:
//! ```text
//! rank r's GEMV stream AR stream r reads peer partials
//! produces partial[r] → hipEventRecord(event_r)
//! ↓
//! AR stream r waits ←─ hipStreamWaitEvent(stream_ar_r, event_(r+1)%N)
//! ←─ hipStreamWaitEvent(stream_ar_r, event_(r+2)%N)
//! ←─ hipStreamWaitEvent(stream_ar_r, event_(r+3)%N)
//! AR launch: BarP2pAllReduce::residual_tp4(...)
//! ```
//! The forward path will set up these waits explicitly. For
//! same-stream usage (producer + AR on the same stream per rank) HIP
//! guarantees serial execution within the stream, so only the
//! cross-rank waits are strictly required.

use std::sync::Arc;

use flambeau_core::{Device, DeviceError, DevicePtr, DeviceResult};

use crate::cluster::HipCluster;
use crate::module::{HipKernel, HipModule, KernelArgs, LaunchCfg};
use crate::HipStream;
use flambeau_kernels_hip as kernels;

const HSACO_NAME: &str = "p2p_allreduce_residual";
const HSACO_NAME_FUSED_NORM: &str = "p2p_allreduce_residual_rmsnorm";
const HSACO_NAME_FUSED_NORM_Q8_1: &str = "p2p_allreduce_residual_rmsnorm_q8_1";
const FN_RESIDUAL_TP4: &str = "flambeau_p2p_allreduce_residual_tp4";
const FN_RESIDUAL_TP2: &str = "flambeau_p2p_allreduce_residual_tp2";
const FN_SUM_TP4: &str = "flambeau_p2p_allreduce_sum_tp4";
const FN_SUM_TP2: &str = "flambeau_p2p_allreduce_sum_tp2";
const FN_SUM_TP4_F32: &str = "flambeau_p2p_allreduce_sum_tp4_f32";
const FN_SUM_TP2_F32: &str = "flambeau_p2p_allreduce_sum_tp2_f32";
const FN_RESIDUAL_RMSNORM_TP4: &str = "flambeau_p2p_allreduce_residual_rmsnorm_tp4";
const FN_RESIDUAL_RMSNORM_TP2: &str = "flambeau_p2p_allreduce_residual_rmsnorm_tp2";
const FN_RESIDUAL_RMSNORM_Q8_1_TP4: &str = "flambeau_p2p_allreduce_residual_rmsnorm_q8_1_tp4";
const FN_RESIDUAL_RMSNORM_Q8_1_TP2: &str = "flambeau_p2p_allreduce_residual_rmsnorm_q8_1_tp2";
const FN_POSTATTN_RESIDUAL_RMSNORM_F32_TO_F16_TP2: &str =
    "flambeau_p2p_allreduce_postattn_residual_rmsnorm_f32_to_f16_tp2";
const FN_POSTATTN_RESIDUAL_RMSNORM_F32_TO_F16_TP4: &str =
    "flambeau_p2p_allreduce_postattn_residual_rmsnorm_f32_to_f16_tp4";
/// 256 threads/block × 2 fp16 elements/thread. Pointwise kernel, low
/// VGPR pressure — gfx906 occupancy is bounded by the launch grid size,
/// not by per-thread resource use.
const BLOCK_THREADS: u32 = 256;

/// Per-rank kernel grid for an `elem_count`-element AllReduce.
fn launch_cfg(elem_count: u32) -> LaunchCfg {
    let blocks = elem_count.div_ceil(BLOCK_THREADS * 2);
    LaunchCfg::one_d(blocks, BLOCK_THREADS)
}

/// Type for the well-known kernel names exposed by this module.
#[derive(Debug, Clone, Copy)]
enum ArKind {
    ResidualTp4,
    ResidualTp2,
    SumTp4,
    SumTp2,
    SumTp4F32,
    SumTp2F32,
    ResidualRmsNormTp4,
    ResidualRmsNormTp2,
    ResidualRmsNormQ8_1Tp4,
    ResidualRmsNormQ8_1Tp2,
    PostAttnResidualRmsNormF32ToF16Tp2,
    PostAttnResidualRmsNormF32ToF16Tp4,
}

impl ArKind {
    fn fn_name(self) -> &'static str {
        match self {
            ArKind::ResidualTp4 => FN_RESIDUAL_TP4,
            ArKind::ResidualTp2 => FN_RESIDUAL_TP2,
            ArKind::SumTp4 => FN_SUM_TP4,
            ArKind::SumTp2 => FN_SUM_TP2,
            ArKind::SumTp4F32 => FN_SUM_TP4_F32,
            ArKind::SumTp2F32 => FN_SUM_TP2_F32,
            ArKind::ResidualRmsNormTp4 => FN_RESIDUAL_RMSNORM_TP4,
            ArKind::ResidualRmsNormTp2 => FN_RESIDUAL_RMSNORM_TP2,
            ArKind::ResidualRmsNormQ8_1Tp4 => FN_RESIDUAL_RMSNORM_Q8_1_TP4,
            ArKind::ResidualRmsNormQ8_1Tp2 => FN_RESIDUAL_RMSNORM_Q8_1_TP2,
            ArKind::PostAttnResidualRmsNormF32ToF16Tp2 => {
                FN_POSTATTN_RESIDUAL_RMSNORM_F32_TO_F16_TP2
            }
            ArKind::PostAttnResidualRmsNormF32ToF16Tp4 => {
                FN_POSTATTN_RESIDUAL_RMSNORM_F32_TO_F16_TP4
            }
        }
    }

    /// Elements processed per thread. F16 kernels pack via `half2`
    /// (2 elements / thread); F32 kernels run scalar (1 elem / thread).
    /// The fused post-attn norm kernels launch one block per row and
    /// don't use this for grid sizing — the value here is irrelevant
    /// for those variants.
    fn elems_per_thread(self) -> u32 {
        match self {
            ArKind::SumTp4F32 | ArKind::SumTp2F32 => 1,
            ArKind::PostAttnResidualRmsNormF32ToF16Tp2
            | ArKind::PostAttnResidualRmsNormF32ToF16Tp4 => 1,
            _ => 2,
        }
    }
}

fn launch_cfg_for(kind: ArKind, elem_count: u32) -> LaunchCfg {
    let elems_per_block = BLOCK_THREADS * kind.elems_per_thread();
    let blocks = elem_count.div_ceil(elems_per_block);
    LaunchCfg::one_d(blocks, BLOCK_THREADS)
}

/// BAR1 P2P AllReduce primitive.
/// Holds one [`HipModule`] per rank — all loaded from the same hsaco —
/// because `HipModule` is bound to a specific HIP device at load time.
/// Kernel handles are resolved on demand via `HipModule::kernel`'s
/// internal cache (uncontended `RwLock` read on the hot path).
/// Construction requires a fully-connected peer-access matrix
/// ([`HipCluster::peer_access_full`]); a partial matrix means at least
/// one rank can't read at least one peer through BAR1, and the caller
/// should fall back to host-bounce AllReduce instead.
pub struct BarP2pAllReduce {
    // **drop-order**: Rust drops fields in declaration order, so
    // anything that holds a per-device resource MUST be declared
    // before the `cluster` Arc. Otherwise the Arc drops first; if it
    // was the last holder the devices get freed; then HipModule drops
    // try to unload from dangling device handles → SIGSEGV at process
    // exit. Same pattern as HipDevice's blas-before-stream rule.
    /// `modules[r]` is the AR hsaco loaded onto rank `r`'s device.
    modules: Vec<HipModule>,
    /// fused AR + residual + RMSNorm hsaco loaded onto each
    /// rank's device. Separate module from the plain-AR one because
    /// build.rs emits one hsaco per `.cu` file.
    modules_fused_norm: Vec<HipModule>,
    /// fused AR + residual + RMSNorm + Q8_1 quantize
    /// hsaco. Used at the cross-layer FFN boundary so the next layer's
    /// first mmvq sees the AR'd-and-quantized x_q8_1 directly.
    modules_fused_norm_q8_1: Vec<HipModule>,
    cluster: Arc<HipCluster>,
}

impl std::fmt::Debug for BarP2pAllReduce {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BarP2pAllReduce")
            .field("ranks", &self.cluster.ranks())
            .finish()
    }
}

impl BarP2pAllReduce {
    /// Load the AllReduce hsaco onto every rank.
    /// # Errors
    /// - The cluster's peer-access matrix isn't fully connected
    /// (`HipCluster::peer_access_full` returned `false`). The
    /// kernel-launched AR path is unsafe to engage in this case;
    /// the caller must fall back to host-bounce.
    /// - The `p2p_allreduce_residual` hsaco wasn't compiled into
    /// `flambeau-kernels-hip` (build.rs failure or `HIP_SKIP_BUILD=1`).
    /// - Any per-rank `hipDeviceBind` / `hipModuleLoadData` failure.
    pub fn new(cluster: Arc<HipCluster>) -> DeviceResult<Self> {
        if !cluster.peer_access_full() {
            return Err(DeviceError::Backend {
                backend: "hip",
                code: -1,
                message: "BarP2pAllReduce requires a fully-connected peer-access matrix; \
                          fall back to host-bounce AllReduce on this cluster"
                    .into(),
            });
        }
        let hsaco = kernels::hsaco(HSACO_NAME).ok_or(DeviceError::Backend {
            backend: "hip",
            code: -1,
            message: "p2p_allreduce_residual hsaco not compiled in flambeau-kernels-hip".into(),
        })?;
        let hsaco_fused = kernels::hsaco(HSACO_NAME_FUSED_NORM).ok_or(DeviceError::Backend {
            backend: "hip",
            code: -1,
            message: "p2p_allreduce_residual_rmsnorm hsaco not compiled in flambeau-kernels-hip"
                .into(),
        })?;
        let hsaco_fused_q8_1 =
            kernels::hsaco(HSACO_NAME_FUSED_NORM_Q8_1).ok_or(DeviceError::Backend {
                backend: "hip",
                code: -1,
                message:
                    "p2p_allreduce_residual_rmsnorm_q8_1 hsaco not compiled in flambeau-kernels-hip"
                        .into(),
            })?;
        let mut modules = Vec::with_capacity(cluster.ranks());
        let mut modules_fused_norm = Vec::with_capacity(cluster.ranks());
        let mut modules_fused_norm_q8_1 = Vec::with_capacity(cluster.ranks());
        for r in 0..cluster.ranks() {
            let dev = cluster.device(r);
            dev.bind()?;
            modules.push(HipModule::load(dev.id(), hsaco)?);
            modules_fused_norm.push(HipModule::load(dev.id(), hsaco_fused)?);
            modules_fused_norm_q8_1.push(HipModule::load(dev.id(), hsaco_fused_q8_1)?);
        }
        Ok(Self {
            cluster,
            modules,
            modules_fused_norm,
            modules_fused_norm_q8_1,
        })
    }

    /// Number of ranks this AllReduce was constructed for.
    pub fn ranks(&self) -> usize {
        self.cluster.ranks()
    }

    /// HIP device id for `rank`. Used by callers that need to allocate
    /// per-rank ordering primitives (e.g. `HipEvent`) bound to the
    /// rank's device without re-acquiring the cluster handle.
    pub fn device_id(&self, rank: usize) -> i32 {
        self.cluster.device(rank).id()
    }

    /// TP=4 residual: on every rank simultaneously,
    /// `hidden[r] += partial[r] + Σ_{k≠r} partial[k]`.
    /// `hidden[r]` and `partial[r]` are device pointers on rank `r`'s
    /// HIP device, each addressing at least `elem_count * 2` bytes
    /// (fp16). Each rank's launch goes on `streams[r]`.
    /// # Safety
    /// - `hidden[r]` and `partial[r]` must be valid for `elem_count`
    /// `__half` values on rank `r`'s device.
    /// - Producer writes to `partial[k]` must be ordered against
    /// `streams[r]` for every peer `k ≠ r` (typically via
    /// `hipEventRecord` on the producer stream + `hipStreamWaitEvent`
    /// on `streams[r]`); see the module-level "Ordering contract".
    /// - `streams[r]` must be on rank `r`'s device.
    pub unsafe fn residual_tp4(
        &self,
        hidden: &[DevicePtr; 4],
        partial: &[DevicePtr; 4],
        elem_count: u32,
        streams: &[&HipStream; 4],
    ) -> DeviceResult<()> {
        self.expect_ranks(4)?;
        let cfg = launch_cfg(elem_count);
        for r in 0..4 {
            // SAFETY: forwarded from the public-method contract.
            // **B5 fix** — canonical-order partials (always (p[0], p[1], p[2], p[3]))
            // so all ranks compute the same FP32 add order. Same reasoning as
            // residual_tp2: non-canonical order causes divergent F16 hidden
            // across ranks → MoE router picks different top-k experts.
            unsafe {
                self.launch_one(
                    ArKind::ResidualTp4,
                    r,
                    cfg,
                    streams[r],
                    ArArgs::Residual {
                        hidden: hidden[r],
                        partial_local: partial[0],
                        peers: [partial[1], partial[2], partial[3]],
                    },
                    elem_count,
                )?;
            }
        }
        Ok(())
    }

    /// TP=2 residual. Mirrors `residual_tp4` for two ranks.
    /// **B5 fix** — partial pointers are passed in CANONICAL (rank-id sorted)
    /// order to both ranks, so both compute `hidden + partial[0] + partial[1]`
    /// in the same FP32 add order. Without this, rank 0 sums `h + p[0] + p[1]`
    /// and rank 1 sums `h + p[1] + p[0]` — non-associative F32 → divergent
    /// F16 cast → divergent post-AR hidden → MoE router picks DIFFERENT top-k
    /// experts on each rank → AR sums mismatched experts → garbage tokens.
    /// Dense-FFN paths happen to round to the same argmax despite the drift,
    /// so this bug only surfaced on qwen35moe (35B-A3B parity drift).
    /// # Safety
    /// Same per-pointer + ordering contract as `residual_tp4`, with
    /// only one peer per rank.
    pub unsafe fn residual_tp2(
        &self,
        hidden: &[DevicePtr; 2],
        partial: &[DevicePtr; 2],
        elem_count: u32,
        streams: &[&HipStream; 2],
    ) -> DeviceResult<()> {
        self.expect_ranks(2)?;
        let cfg = launch_cfg(elem_count);
        for r in 0..2 {
            // SAFETY: forwarded from the public-method contract.
            // Canonical order: partial[0] is "local" arg, partial[1] is "peer"
            // — for rank 0, partial[0] is local-device read and partial[1] is
            // a BAR1 P2P read; for rank 1 the roles swap (partial[0] becomes
            // BAR1 P2P, partial[1] becomes local-device). The kernel just sees
            // two pointers and sums in the same order on both ranks.
            unsafe {
                self.launch_one(
                    ArKind::ResidualTp2,
                    r,
                    cfg,
                    streams[r],
                    ArArgs::Residual {
                        hidden: hidden[r],
                        partial_local: partial[0],
                        peers: [partial[1], DevicePtr(0), DevicePtr(0)],
                    },
                    elem_count,
                )?;
            }
        }
        Ok(())
    }

    /// TP=4 sum (no residual): on every rank,
    /// `partial[r] = Σ partial[k]` for k in 0..4.
    /// # Safety
    /// Same per-pointer + ordering contract as `residual_tp4`.
    pub unsafe fn sum_tp4(
        &self,
        partial: &[DevicePtr; 4],
        elem_count: u32,
        streams: &[&HipStream; 4],
    ) -> DeviceResult<()> {
        self.expect_ranks(4)?;
        let cfg = launch_cfg(elem_count);
        for r in 0..4 {
            // Per-rank local write target. See `sum_tp2` for the
            // rationale: `partial_local` is the kernel's write target,
            // so it must be rank r's own buffer; otherwise rank r's
            // local partial stays at the pre-AR value while the sum
            // gets BAR1-written into rank 0's buffer.
            let peers = [
                partial[(r + 1) % 4],
                partial[(r + 2) % 4],
                partial[(r + 3) % 4],
            ];
            // SAFETY: forwarded from the public-method contract.
            unsafe {
                self.launch_one(
                    ArKind::SumTp4,
                    r,
                    cfg,
                    streams[r],
                    ArArgs::Sum {
                        partial_local: partial[r],
                        peers,
                    },
                    elem_count,
                )?;
            }
        }
        Ok(())
    }

    /// fused TP=4 AllReduce + residual-add + RMSNorm.
    /// On every rank, in one launch:
    /// `hidden[r] += partial[r] + Σ_{k≠r} partial[k]`
    /// `out_norm[r] = hidden[r] * rsqrt(mean(hidden²) + eps) * rms_weight`
    /// `n` is the hidden_size; must satisfy `n % 256 == 0` (single-block
    /// kernel; per-thread element count is `n / 256`).
    /// # Safety
    /// - `hidden[r]`, `partial[r]`, `out_norm[r]` valid for `n` `__half`
    /// elements on rank `r`'s device.
    /// - `rms_weight[r]` valid for `n` `__half` elements (Replicated
    /// per the layout table).
    /// - Producer-stream ordering as in `residual_tp4`.
    pub unsafe fn residual_rmsnorm_tp4(
        &self,
        hidden: &[DevicePtr; 4],
        partial: &[DevicePtr; 4],
        rms_weight: &[DevicePtr; 4],
        out_norm: &[DevicePtr; 4],
        n: u32,
        eps: f32,
        streams: &[&HipStream; 4],
    ) -> DeviceResult<()> {
        self.expect_ranks(4)?;
        if n % BLOCK_THREADS != 0 {
            return Err(DeviceError::Backend {
                backend: "hip",
                code: -1,
                message: format!(
                    "residual_rmsnorm_tp4: n={n} not divisible by BLOCK_THREADS={BLOCK_THREADS}"
                ),
            });
        }
        // Single-block kernel: one block per rank.
        let cfg = LaunchCfg::one_d(1, BLOCK_THREADS);
        for r in 0..4 {
            // SAFETY: forwarded from public-method contract.
            // **B5 fix** — canonical-order partials. See residual_tp2 / tp4.
            unsafe {
                self.launch_fused_rmsnorm(
                    ArKind::ResidualRmsNormTp4,
                    r,
                    cfg,
                    streams[r],
                    hidden[r],
                    partial[0],
                    [partial[1], partial[2], partial[3]],
                    rms_weight[r],
                    out_norm[r],
                    n,
                    eps,
                )?;
            }
        }
        Ok(())
    }

    /// fused TP=4 AR + residual + RMSNorm + Q8_1 quantize.
    /// Cross-layer FFN-boundary lever: produces the next layer's
    /// `x_q8_1` directly so the next forward call's first mmvq doesn't
    /// run a separate `rmsnorm_quant_q8_1`.
    /// `out_q8_1[r]` must point to at least `n / QK8_1` `flambeau_block_q8_1`
    /// entries on rank `r`'s device (= `n / 32 * sizeof(BlockQ8_1)` bytes).
    /// `n % 256 == 0` and `n % QK8_1 == 0` (covers Qwen3.5/3.6 hidden sizes).
    /// # Safety
    /// Same per-pointer + ordering contract as `residual_rmsnorm_tp4`,
    /// with `out_q8_1` replacing `out_norm`.
    pub unsafe fn residual_rmsnorm_q8_1_tp4(
        &self,
        hidden: &[DevicePtr; 4],
        partial: &[DevicePtr; 4],
        rms_weight: &[DevicePtr; 4],
        out_q8_1: &[DevicePtr; 4],
        n: u32,
        eps: f32,
        streams: &[&HipStream; 4],
    ) -> DeviceResult<()> {
        self.expect_ranks(4)?;
        if n % BLOCK_THREADS != 0 || n % 32 != 0 {
            return Err(DeviceError::Backend {
                backend: "hip",
                code: -1,
                message: format!(
                    "residual_rmsnorm_q8_1_tp4: n={n} must be divisible by 256 (block) and 32 (Q8_1)"
                ),
            });
        }
        let cfg = LaunchCfg::one_d(1, BLOCK_THREADS);
        for r in 0..4 {
            // SAFETY: forwarded from public-method contract.
            // **B5 fix** — canonical-order partials. See residual_tp2 / tp4.
            unsafe {
                self.launch_fused_rmsnorm_q8_1(
                    ArKind::ResidualRmsNormQ8_1Tp4,
                    r,
                    cfg,
                    streams[r],
                    hidden[r],
                    partial[0],
                    [partial[1], partial[2], partial[3]],
                    rms_weight[r],
                    out_q8_1[r],
                    n,
                    eps,
                )?;
            }
        }
        Ok(())
    }

    /// fused TP=2 AR + residual + RMSNorm + Q8_1 quantize.
    /// # Safety
    /// Same per-pointer + ordering contract as `residual_rmsnorm_q8_1_tp4`.
    pub unsafe fn residual_rmsnorm_q8_1_tp2(
        &self,
        hidden: &[DevicePtr; 2],
        partial: &[DevicePtr; 2],
        rms_weight: &[DevicePtr; 2],
        out_q8_1: &[DevicePtr; 2],
        n: u32,
        eps: f32,
        streams: &[&HipStream; 2],
    ) -> DeviceResult<()> {
        self.expect_ranks(2)?;
        if n % BLOCK_THREADS != 0 || n % 32 != 0 {
            return Err(DeviceError::Backend {
                backend: "hip",
                code: -1,
                message: format!(
                    "residual_rmsnorm_q8_1_tp2: n={n} must be divisible by 256 (block) and 32 (Q8_1)"
                ),
            });
        }
        let cfg = LaunchCfg::one_d(1, BLOCK_THREADS);
        for r in 0..2 {
            // SAFETY: forwarded from public-method contract.
            // **B5 fix** — canonical-order partials. See residual_tp2.
            unsafe {
                self.launch_fused_rmsnorm_q8_1(
                    ArKind::ResidualRmsNormQ8_1Tp2,
                    r,
                    cfg,
                    streams[r],
                    hidden[r],
                    partial[0],
                    [partial[1], DevicePtr(0), DevicePtr(0)],
                    rms_weight[r],
                    out_q8_1[r],
                    n,
                    eps,
                )?;
            }
        }
        Ok(())
    }

    /// fused TP=2 AllReduce + residual-add + RMSNorm.
    /// # Safety
    /// Same per-pointer + ordering contract as `residual_rmsnorm_tp4`.
    pub unsafe fn residual_rmsnorm_tp2(
        &self,
        hidden: &[DevicePtr; 2],
        partial: &[DevicePtr; 2],
        rms_weight: &[DevicePtr; 2],
        out_norm: &[DevicePtr; 2],
        n: u32,
        eps: f32,
        streams: &[&HipStream; 2],
    ) -> DeviceResult<()> {
        self.expect_ranks(2)?;
        if n % BLOCK_THREADS != 0 {
            return Err(DeviceError::Backend {
                backend: "hip",
                code: -1,
                message: format!(
                    "residual_rmsnorm_tp2: n={n} not divisible by BLOCK_THREADS={BLOCK_THREADS}"
                ),
            });
        }
        let cfg = LaunchCfg::one_d(1, BLOCK_THREADS);
        for r in 0..2 {
            // SAFETY: forwarded from public-method contract.
            // **B5 fix** — canonical-order partials (partial[0] then partial[1])
            // so both ranks compute the same FP32 sum order. See residual_tp2.
            unsafe {
                self.launch_fused_rmsnorm(
                    ArKind::ResidualRmsNormTp2,
                    r,
                    cfg,
                    streams[r],
                    hidden[r],
                    partial[0],
                    [partial[1], DevicePtr(0), DevicePtr(0)],
                    rms_weight[r],
                    out_norm[r],
                    n,
                    eps,
                )?;
            }
        }
        Ok(())
    }

    /// TP=2 F32 sum: per-rank `partial[r] += partial[1-r]` over `n`
    /// F32 elements. Sibling of [`Self::sum_tp2`] for callers that
    /// keep partials in F32 (gemma4 attention output projection —
    /// the F32→F16 cast at the row-parallel matmul output saturates
    /// on V-spike inputs; F32 AR preserves the magnitudes until the
    /// next rmsnorm normalises them).
    /// # Safety
    /// Same per-pointer + ordering contract as [`Self::sum_tp2`].
    pub unsafe fn sum_tp2_f32(
        &self,
        partial: &[DevicePtr; 2],
        elem_count: u32,
        streams: &[&HipStream; 2],
    ) -> DeviceResult<()> {
        self.expect_ranks(2)?;
        let cfg = launch_cfg_for(ArKind::SumTp2F32, elem_count);
        for r in 0..2 {
            let peer = partial[1 - r];
            // SAFETY: forwarded from the public-method contract.
            unsafe {
                self.launch_one(
                    ArKind::SumTp2F32,
                    r,
                    cfg,
                    streams[r],
                    ArArgs::Sum {
                        partial_local: partial[r],
                        peers: [peer, DevicePtr(0), DevicePtr(0)],
                    },
                    elem_count,
                )?;
            }
        }
        Ok(())
    }

    /// TP=4 F32 sum: per-rank `partial[r] += Σ_{p≠r} partial[p]` over
    /// `n` F32 elements. Sibling of [`Self::sum_tp4`].
    /// # Safety
    /// Same per-pointer + ordering contract as [`Self::sum_tp4`].
    pub unsafe fn sum_tp4_f32(
        &self,
        partial: &[DevicePtr; 4],
        elem_count: u32,
        streams: &[&HipStream; 4],
    ) -> DeviceResult<()> {
        self.expect_ranks(4)?;
        let cfg = launch_cfg_for(ArKind::SumTp4F32, elem_count);
        for r in 0..4 {
            let peers = [
                partial[(r + 1) % 4],
                partial[(r + 2) % 4],
                partial[(r + 3) % 4],
            ];
            // SAFETY: forwarded from the public-method contract.
            unsafe {
                self.launch_one(
                    ArKind::SumTp4F32,
                    r,
                    cfg,
                    streams[r],
                    ArArgs::Sum {
                        partial_local: partial[r],
                        peers,
                    },
                    elem_count,
                )?;
            }
        }
        Ok(())
    }

    /// TP=2 sum.
    /// # Safety
    /// Same per-pointer + ordering contract as `residual_tp4`.
    pub unsafe fn sum_tp2(
        &self,
        partial: &[DevicePtr; 2],
        elem_count: u32,
        streams: &[&HipStream; 2],
    ) -> DeviceResult<()> {
        self.expect_ranks(2)?;
        let cfg = launch_cfg(elem_count);
        for r in 0..2 {
            // The `partial_local` arg is *written to* by the kernel
            // (`partial_local = partial_local + peer`). For both ranks
            // to end up with the sum locally, rank r must point
            // `partial_local` at `partial[r]` — passing the same
            // partial[0] on every rank means rank 1 would write via
            // BAR1 to rank 0's buffer and rank 1's own buffer would
            // stay at the pre-AR value. The peer is the OTHER rank's
            // partial.
            let peer = partial[1 - r];
            // SAFETY: forwarded from the public-method contract.
            unsafe {
                self.launch_one(
                    ArKind::SumTp2,
                    r,
                    cfg,
                    streams[r],
                    ArArgs::Sum {
                        partial_local: partial[r],
                        peers: [peer, DevicePtr(0), DevicePtr(0)],
                    },
                    elem_count,
                )?;
            }
        }
        Ok(())
    }

    /// Rank-local F32 AllReduce-sum for TP=2. Each rank's worker
    /// calls this for its own rank; caller exchanges peer pointers via
    /// an out-of-band barrier and sync-drains the producer stream
    /// before invocation (see `feedback_bar_p2p_sum_write_target`).
    /// # Safety
    /// - `partial_local` is on `rank`'s device and valid for
    ///   `elem_count` F32 elements.
    /// - `peer` is on the other rank's device, mapped via BAR1, and
    ///   valid for `elem_count` F32 elements.
    /// - Producer writes to BOTH partials have been ordered against
    ///   `stream` (typically `Stream::synchronize` on every rank
    ///   before all-ranks-publish-then-launch).
    pub unsafe fn sum_tp2_f32_rank(
        &self,
        rank: usize,
        partial_local: DevicePtr,
        peer: DevicePtr,
        elem_count: u32,
        stream: &HipStream,
    ) -> DeviceResult<()> {
        self.expect_ranks(2)?;
        let cfg = launch_cfg_for(ArKind::SumTp2F32, elem_count);
        // SAFETY: forwarded from public-method contract.
        unsafe {
            self.launch_one(
                ArKind::SumTp2F32,
                rank,
                cfg,
                stream,
                ArArgs::Sum {
                    partial_local,
                    peers: [peer, DevicePtr(0), DevicePtr(0)],
                },
                elem_count,
            )
        }
    }

    /// Rank-local F16 AllReduce-sum for TP=2. F16 sibling of
    /// [`Self::sum_tp2_f32_rank`]. Halves the BAR1 payload at the
    /// price of F16 saturation on overflowing partials — gated by
    /// caller (e.g. `output_proj_safe_for_f16_ar` on gemma4).
    /// # Safety
    /// Same per-pointer + ordering contract as
    /// [`Self::sum_tp2_f32_rank`], but partials are F16.
    pub unsafe fn sum_tp2_rank(
        &self,
        rank: usize,
        partial_local: DevicePtr,
        peer: DevicePtr,
        elem_count: u32,
        stream: &HipStream,
    ) -> DeviceResult<()> {
        self.expect_ranks(2)?;
        let cfg = launch_cfg_for(ArKind::SumTp2, elem_count);
        // SAFETY: forwarded from public-method contract.
        unsafe {
            self.launch_one(
                ArKind::SumTp2,
                rank,
                cfg,
                stream,
                ArArgs::Sum {
                    partial_local,
                    peers: [peer, DevicePtr(0), DevicePtr(0)],
                },
                elem_count,
            )
        }
    }

    /// Rank-local fused AR + residual-add for TP=2 (F16). Caller
    /// exchanges peer partial pointer out-of-band; both ranks pass
    /// partial in canonical order (rank0's then rank1's) so the
    /// F32 reduce order matches across ranks.
    /// # Safety
    /// Same per-pointer + ordering contract as
    /// [`Self::sum_tp2_f32_rank`]. `hidden` and `partial`s are F16.
    pub unsafe fn residual_tp2_rank(
        &self,
        rank: usize,
        hidden: DevicePtr,
        partial_canonical_rank0: DevicePtr,
        partial_canonical_rank1: DevicePtr,
        elem_count: u32,
        stream: &HipStream,
    ) -> DeviceResult<()> {
        self.expect_ranks(2)?;
        let cfg = launch_cfg(elem_count);
        let _ = rank;
        // SAFETY: forwarded from public-method contract.
        unsafe {
            self.launch_one(
                ArKind::ResidualTp2,
                rank,
                cfg,
                stream,
                ArArgs::Residual {
                    hidden,
                    partial_local: partial_canonical_rank0,
                    peers: [partial_canonical_rank1, DevicePtr(0), DevicePtr(0)],
                },
                elem_count,
            )
        }
    }

    /// Rank-local fused AR + residual-add + RMSNorm for TP=2 (F16).
    /// `out_norm` receives `rmsnorm(hidden + Σ partials, rms_weight)`;
    /// `hidden` is also updated in-place. Same canonical-order /
    /// producer-sync contract as [`Self::residual_tp2_rank`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn residual_rmsnorm_tp2_rank(
        &self,
        rank: usize,
        hidden: DevicePtr,
        partial_canonical_rank0: DevicePtr,
        partial_canonical_rank1: DevicePtr,
        rms_weight: DevicePtr,
        out_norm: DevicePtr,
        n: u32,
        eps: f32,
        stream: &HipStream,
    ) -> DeviceResult<()> {
        self.expect_ranks(2)?;
        if n % BLOCK_THREADS != 0 {
            return Err(DeviceError::Backend {
                backend: "hip",
                code: -1,
                message: format!(
                    "residual_rmsnorm_tp2_rank: n={n} not divisible by BLOCK_THREADS={BLOCK_THREADS}"
                ),
            });
        }
        let cfg = LaunchCfg::one_d(1, BLOCK_THREADS);
        // SAFETY: forwarded from public-method contract.
        unsafe {
            self.launch_fused_rmsnorm(
                ArKind::ResidualRmsNormTp2,
                rank,
                cfg,
                stream,
                hidden,
                partial_canonical_rank0,
                [partial_canonical_rank1, DevicePtr(0), DevicePtr(0)],
                rms_weight,
                out_norm,
                n,
                eps,
            )
        }
    }

    /// Rank-local F32 AllReduce-sum for TP=4. Caller exchanges all
    /// three peer pointers out-of-band.
    /// # Safety
    /// Same per-pointer + ordering contract as [`Self::sum_tp2_f32_rank`].
    pub unsafe fn sum_tp4_f32_rank(
        &self,
        rank: usize,
        partial_local: DevicePtr,
        peers: [DevicePtr; 3],
        elem_count: u32,
        stream: &HipStream,
    ) -> DeviceResult<()> {
        self.expect_ranks(4)?;
        let cfg = launch_cfg_for(ArKind::SumTp4F32, elem_count);
        // SAFETY: forwarded from public-method contract.
        unsafe {
            self.launch_one(
                ArKind::SumTp4F32,
                rank,
                cfg,
                stream,
                ArArgs::Sum {
                    partial_local,
                    peers,
                },
                elem_count,
            )
        }
    }

    /// Rank-local F16 AllReduce-sum for TP=4. F16 sibling of
    /// [`Self::sum_tp4_f32_rank`].
    /// # Safety
    /// Same per-pointer + ordering contract as
    /// [`Self::sum_tp2_f32_rank`], but partials are F16.
    pub unsafe fn sum_tp4_rank(
        &self,
        rank: usize,
        partial_local: DevicePtr,
        peers: [DevicePtr; 3],
        elem_count: u32,
        stream: &HipStream,
    ) -> DeviceResult<()> {
        self.expect_ranks(4)?;
        let cfg = launch_cfg_for(ArKind::SumTp4, elem_count);
        // SAFETY: forwarded from public-method contract.
        unsafe {
            self.launch_one(
                ArKind::SumTp4,
                rank,
                cfg,
                stream,
                ArArgs::Sum {
                    partial_local,
                    peers,
                },
                elem_count,
            )
        }
    }

    /// Rank-local fused AR + post-attn-norm + residual-add for TP=2,
    /// F32-input projection → F16 output residual. Replaces a 2-launch
    /// `ar_sum_f32 + rmsnorm_f32_to_f16_add_residual` sequence on the
    /// gemma4 post-attn / post-ffn paths. `n_rows` blocks of
    /// [`BLOCK_THREADS`] threads; per-row hidden `n` must satisfy
    /// `n <= 32 * BLOCK_THREADS = 8192` (per-thread AR-sum register
    /// cache).
    /// # Safety
    /// - `proj_local`, `peer` are F32 on respective devices, each
    ///   valid for `n_rows * n` elements.
    /// - `post_norm_w` is F16 on rank's device, valid for `n` elems.
    /// - `resid_in`, `resid_out` are F16 on rank's device, each valid
    ///   for `n_rows * n` elements. `resid_out` may NOT alias
    ///   `resid_in` (kernel reads resid_in and writes resid_out in the
    ///   same pass; aliasing would race).
    /// - Producer-stream ordering as in [`Self::sum_tp2_f32_rank`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn postattn_residual_rmsnorm_f32_to_f16_tp2_rank(
        &self,
        rank: usize,
        proj_local: DevicePtr,
        peer: DevicePtr,
        post_norm_w: DevicePtr,
        resid_in: DevicePtr,
        resid_out: DevicePtr,
        n_rows: u32,
        n: u32,
        eps: f32,
        stream: &HipStream,
    ) -> DeviceResult<()> {
        self.expect_ranks(2)?;
        if n > 8192 {
            return Err(DeviceError::Backend {
                backend: "hip",
                code: -1,
                message: format!(
                    "postattn_residual_rmsnorm_f32_to_f16_tp2_rank: n={n} > 8192 \
                     (per-thread register-cache cap)"
                ),
            });
        }
        let cfg = LaunchCfg::one_d(n_rows, BLOCK_THREADS);
        // SAFETY: forwarded from public-method contract.
        unsafe {
            self.launch_postattn_norm_f32_to_f16(
                ArKind::PostAttnResidualRmsNormF32ToF16Tp2,
                rank,
                cfg,
                stream,
                proj_local,
                peer,
                [DevicePtr(0), DevicePtr(0)],
                post_norm_w,
                resid_in,
                resid_out,
                n,
                eps,
            )
        }
    }

    /// TP=4 sibling of [`Self::postattn_residual_rmsnorm_f32_to_f16_tp2_rank`].
    /// # Safety
    /// Same contract; 3 peer F32 partials instead of 1.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn postattn_residual_rmsnorm_f32_to_f16_tp4_rank(
        &self,
        rank: usize,
        proj_local: DevicePtr,
        peers: [DevicePtr; 3],
        post_norm_w: DevicePtr,
        resid_in: DevicePtr,
        resid_out: DevicePtr,
        n_rows: u32,
        n: u32,
        eps: f32,
        stream: &HipStream,
    ) -> DeviceResult<()> {
        self.expect_ranks(4)?;
        if n > 8192 {
            return Err(DeviceError::Backend {
                backend: "hip",
                code: -1,
                message: format!(
                    "postattn_residual_rmsnorm_f32_to_f16_tp4_rank: n={n} > 8192"
                ),
            });
        }
        let cfg = LaunchCfg::one_d(n_rows, BLOCK_THREADS);
        // SAFETY: forwarded from public-method contract.
        unsafe {
            self.launch_postattn_norm_f32_to_f16(
                ArKind::PostAttnResidualRmsNormF32ToF16Tp4,
                rank,
                cfg,
                stream,
                proj_local,
                peers[0],
                [peers[1], peers[2]],
                post_norm_w,
                resid_in,
                resid_out,
                n,
                eps,
            )
        }
    }

    fn expect_ranks(&self, expected: usize) -> DeviceResult<()> {
        if self.cluster.ranks() == expected {
            Ok(())
        } else {
            Err(DeviceError::Backend {
                backend: "hip",
                code: -1,
                message: format!(
                    "BarP2pAllReduce::*_tp{}: cluster has {} ranks, expected {}",
                    expected,
                    self.cluster.ranks(),
                    expected
                ),
            })
        }
    }

    /// SAFETY: caller's contract on the public methods.
    unsafe fn launch_one(
        &self,
        kind: ArKind,
        rank: usize,
        cfg: LaunchCfg,
        stream: &HipStream,
        args: ArArgs,
        elem_count: u32,
    ) -> DeviceResult<()> {
        self.cluster.device(rank).bind()?;
        let kern: HipKernel<'_> = self.modules[rank].kernel(kind.fn_name())?;
        let n = elem_count;
        // Pack pointer args as u64 (matches `void*` ABI on 64-bit).
        match args {
            ArArgs::Residual {
                hidden,
                partial_local,
                peers,
            } => {
                let h = hidden.as_usize() as u64;
                let pl = partial_local.as_usize() as u64;
                let p0 = peers[0].as_usize() as u64;
                let p1 = peers[1].as_usize() as u64;
                let p2 = peers[2].as_usize() as u64;
                let mut k_args = KernelArgs::new();
                k_args.push(&h);
                k_args.push(&pl);
                k_args.push(&p0);
                if matches!(kind, ArKind::ResidualTp4) {
                    k_args.push(&p1);
                    k_args.push(&p2);
                }
                k_args.push(&n);
                // SAFETY: forwarded from public-method contracts. `kern`
                // and `stream` live for the duration of the launch; arg
                // locals live until end of scope (after the synchronous
                // return of `hipModuleLaunchKernel`).
                unsafe { kern.launch(stream, cfg, k_args)? };
            }
            ArArgs::Sum {
                partial_local,
                peers,
            } => {
                let pl = partial_local.as_usize() as u64;
                let p0 = peers[0].as_usize() as u64;
                let p1 = peers[1].as_usize() as u64;
                let p2 = peers[2].as_usize() as u64;
                let mut k_args = KernelArgs::new();
                k_args.push(&pl);
                k_args.push(&p0);
                if matches!(kind, ArKind::SumTp4 | ArKind::SumTp4F32) {
                    k_args.push(&p1);
                    k_args.push(&p2);
                }
                k_args.push(&n);
                // SAFETY: same as Residual arm above.
                unsafe { kern.launch(stream, cfg, k_args)? };
            }
        }
        Ok(())
    }

    /// launch the fused AR+residual+RMSNorm kernel on rank
    /// `rank`. Resolved from the rank's `modules_fused_norm[rank]`.
    /// # Safety
    /// Forwarded from `residual_rmsnorm_tp{2,4}` public-method contracts.
    #[expect(
        clippy::too_many_arguments,
        reason = "matches the kernel's flat ABI; collapsing into a struct \
                  would just rename the same locals"
    )]
    unsafe fn launch_fused_rmsnorm(
        &self,
        kind: ArKind,
        rank: usize,
        cfg: LaunchCfg,
        stream: &HipStream,
        hidden: DevicePtr,
        partial_local: DevicePtr,
        peers: [DevicePtr; 3],
        rms_weight: DevicePtr,
        out_norm: DevicePtr,
        n: u32,
        eps: f32,
    ) -> DeviceResult<()> {
        self.cluster.device(rank).bind()?;
        let kern: HipKernel<'_> = self.modules_fused_norm[rank].kernel(kind.fn_name())?;
        let h = hidden.as_usize() as u64;
        let pl = partial_local.as_usize() as u64;
        let p0 = peers[0].as_usize() as u64;
        let p1 = peers[1].as_usize() as u64;
        let p2 = peers[2].as_usize() as u64;
        let w = rms_weight.as_usize() as u64;
        let o = out_norm.as_usize() as u64;
        let mut k_args = KernelArgs::new();
        k_args.push(&h);
        k_args.push(&pl);
        k_args.push(&p0);
        if matches!(kind, ArKind::ResidualRmsNormTp4) {
            k_args.push(&p1);
            k_args.push(&p2);
        }
        k_args.push(&w);
        k_args.push(&o);
        k_args.push(&n);
        k_args.push(&eps);
        // SAFETY: forwarded from the public-method contract — every
        // pointer is a live device alloc on `rank`'s device with the
        // documented sizing; producer streams synced against `stream`
        // via the caller-side ordering contract.
        unsafe { kern.launch(stream, cfg, k_args)? };
        Ok(())
    }

    /// launch the gemma4 post-attn / post-ffn fused kernel.
    /// Kernel ABI (TP=2):
    ///   (proj_local, peer0, post_norm_w, resid_in, resid_out, n, eps)
    /// TP=4 inserts `peer1, peer2` after `peer0`.
    /// # Safety
    /// Forwarded from `postattn_residual_rmsnorm_f32_to_f16_tp{2,4}_rank`.
    #[expect(clippy::too_many_arguments, reason = "matches the kernel's flat ABI")]
    unsafe fn launch_postattn_norm_f32_to_f16(
        &self,
        kind: ArKind,
        rank: usize,
        cfg: LaunchCfg,
        stream: &HipStream,
        proj_local: DevicePtr,
        peer0: DevicePtr,
        peers_extra: [DevicePtr; 2],
        post_norm_w: DevicePtr,
        resid_in: DevicePtr,
        resid_out: DevicePtr,
        n: u32,
        eps: f32,
    ) -> DeviceResult<()> {
        self.cluster.device(rank).bind()?;
        let kern: HipKernel<'_> = self.modules_fused_norm[rank].kernel(kind.fn_name())?;
        let pl = proj_local.as_usize() as u64;
        let p0 = peer0.as_usize() as u64;
        let p1 = peers_extra[0].as_usize() as u64;
        let p2 = peers_extra[1].as_usize() as u64;
        let w = post_norm_w.as_usize() as u64;
        let ri = resid_in.as_usize() as u64;
        let ro = resid_out.as_usize() as u64;
        let mut k_args = KernelArgs::new();
        k_args.push(&pl);
        k_args.push(&p0);
        if matches!(kind, ArKind::PostAttnResidualRmsNormF32ToF16Tp4) {
            k_args.push(&p1);
            k_args.push(&p2);
        }
        k_args.push(&w);
        k_args.push(&ri);
        k_args.push(&ro);
        k_args.push(&n);
        k_args.push(&eps);
        // SAFETY: forwarded from the public-method contract.
        unsafe { kern.launch(stream, cfg, k_args)? };
        Ok(())
    }

    /// launch the 4-op fused AR+residual+RMSNorm+Q8_1
    /// kernel. Same arg layout as `launch_fused_rmsnorm` except the
    /// final pointer is a `flambeau_block_q8_1*` output instead of a
    /// `__half*` normed output. Kernel ABI matches.
    /// # Safety
    /// Forwarded from `residual_rmsnorm_q8_1_tp{2,4}` public-method
    /// contracts.
    #[expect(clippy::too_many_arguments, reason = "matches the kernel's flat ABI")]
    unsafe fn launch_fused_rmsnorm_q8_1(
        &self,
        kind: ArKind,
        rank: usize,
        cfg: LaunchCfg,
        stream: &HipStream,
        hidden: DevicePtr,
        partial_local: DevicePtr,
        peers: [DevicePtr; 3],
        rms_weight: DevicePtr,
        out_q8_1: DevicePtr,
        n: u32,
        eps: f32,
    ) -> DeviceResult<()> {
        self.cluster.device(rank).bind()?;
        let kern: HipKernel<'_> = self.modules_fused_norm_q8_1[rank].kernel(kind.fn_name())?;
        let h = hidden.as_usize() as u64;
        let pl = partial_local.as_usize() as u64;
        let p0 = peers[0].as_usize() as u64;
        let p1 = peers[1].as_usize() as u64;
        let p2 = peers[2].as_usize() as u64;
        let w = rms_weight.as_usize() as u64;
        let o = out_q8_1.as_usize() as u64;
        let mut k_args = KernelArgs::new();
        k_args.push(&h);
        k_args.push(&pl);
        k_args.push(&p0);
        if matches!(kind, ArKind::ResidualRmsNormQ8_1Tp4) {
            k_args.push(&p1);
            k_args.push(&p2);
        }
        k_args.push(&w);
        k_args.push(&o);
        k_args.push(&n);
        k_args.push(&eps);
        // SAFETY: forwarded from the public-method contract.
        unsafe { kern.launch(stream, cfg, k_args)? };
        Ok(())
    }
}

enum ArArgs {
    Residual {
        hidden: DevicePtr,
        partial_local: DevicePtr,
        /// peers[0..usable] where usable is 1 (tp2) or 3 (tp4).
        peers: [DevicePtr; 3],
    },
    Sum {
        partial_local: DevicePtr,
        peers: [DevicePtr; 3],
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_cfg_blocks_round_up() {
        // Exact multiples: 1024 / 512 = 2 blocks.
        assert_eq!(launch_cfg(1024).grid.0, 2);
        // 5120 / 512 = 10 blocks.
        assert_eq!(launch_cfg(5120).grid.0, 10);
        // 8193 needs ceil(8193/512) = 17 blocks.
        assert_eq!(launch_cfg(8193).grid.0, 17);
        // Block size always 256 threads.
        assert_eq!(launch_cfg(1).block.0, 256);
    }

    #[test]
    fn ar_kind_fn_names_match_kernel_exports() {
        // String-level guard — if the kernel file gets renamed, the cert
        // harness in `tests/p2p_allreduce_smoke.rs` will catch it at
        // runtime, but we want a compile-locality check too.
        assert_eq!(
            ArKind::ResidualTp4.fn_name(),
            "flambeau_p2p_allreduce_residual_tp4"
        );
        assert_eq!(
            ArKind::ResidualTp2.fn_name(),
            "flambeau_p2p_allreduce_residual_tp2"
        );
        assert_eq!(ArKind::SumTp4.fn_name(), "flambeau_p2p_allreduce_sum_tp4");
        assert_eq!(ArKind::SumTp2.fn_name(), "flambeau_p2p_allreduce_sum_tp2");
        assert_eq!(
            ArKind::ResidualRmsNormTp4.fn_name(),
            "flambeau_p2p_allreduce_residual_rmsnorm_tp4"
        );
        assert_eq!(
            ArKind::ResidualRmsNormTp2.fn_name(),
            "flambeau_p2p_allreduce_residual_rmsnorm_tp2"
        );
        assert_eq!(
            ArKind::ResidualRmsNormQ8_1Tp4.fn_name(),
            "flambeau_p2p_allreduce_residual_rmsnorm_q8_1_tp4"
        );
        assert_eq!(
            ArKind::ResidualRmsNormQ8_1Tp2.fn_name(),
            "flambeau_p2p_allreduce_residual_rmsnorm_q8_1_tp2"
        );
    }
}
