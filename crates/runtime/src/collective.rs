//! Collective op traits and a CPU host-bounce reference implementation.
//! The reference impl runs on a shared-memory `Mesh<N>` — each "rank" is a
//! thread that uses a barrier + shared buffers to fan data through the
//! collective. It has two jobs:
//! 1. Exercise the `Mesh<N>` trait surface for end gate without
//!    requiring physical GPUs.
//! 2. Serve as the correctness oracle against which RCCL / NCCL impls are
//!    certed in `bench sweep`.

use std::sync::{Arc, Barrier, Mutex};

use flambeau_core::device::Device;
use flambeau_core::DevicePtr;

use crate::mesh::{CollectiveCfg, CollectiveDType, Mesh, RankId, ReduceOp};

/// Error surface for the collective layer.
#[derive(Debug, thiserror::Error)]
pub enum CollectiveError {
    #[error("rank {rank} out of range for mesh of size {size}")]
    RankOutOfRange { rank: u32, size: u32 },

    #[error("buffer length mismatch: expected {expected_elems} {dtype:?}, got {got} bytes")]
    BufferLen {
        expected_elems: usize,
        dtype: CollectiveDType,
        got: usize,
    },

    #[error("collective backend not built ({feature:?})")]
    BackendNotBuilt { feature: &'static str },

    #[error("{backend} collective error ({ctx}): code={code} {message}")]
    Backend {
        backend: &'static str,
        ctx: &'static str,
        code: i32,
        message: String,
    },

    #[error("{backend} device op failed ({ctx}): {message}")]
    Device {
        backend: &'static str,
        ctx: &'static str,
        message: String,
    },
}

pub type CollectiveResult<T> = std::result::Result<T, CollectiveError>;

fn check_buf(buf: &[u8], cfg: &CollectiveCfg) -> CollectiveResult<()> {
    let expected = cfg.buffer_bytes();
    if buf.len() != expected {
        return Err(CollectiveError::BufferLen {
            expected_elems: cfg.elem_count,
            dtype: cfg.dtype,
            got: buf.len(),
        });
    }
    Ok(())
}

// ---- reference mesh --------------------------------------------------------

/// CPU host-bounce mesh: N "ranks" run in N threads and cooperate on
/// shared-memory buffers guarded by a `Barrier` + a `Mutex<Vec<...>>`.
/// Each rank holds a `RefRankHandle` that issues collectives against this
/// shared state. Tests spawn N threads, hand each its handle, and assert the
/// post-collective buffer matches the math.
#[derive(Debug)]
pub struct RefMesh {
    size: u32,
    staging: Arc<RefStaging>,
}

#[derive(Debug)]
struct RefStaging {
    /// Per-rank byte buffers for the current collective phase. Initial contents
    /// come from the rank's call; post-barrier the reducer fills them in-place.
    slots: Mutex<Vec<Vec<u8>>>,
    barrier: Barrier,
}

impl RefMesh {
    pub fn new(size: u32) -> Arc<Self> {
        assert!(size >= 1);
        Arc::new(Self {
            size,
            staging: Arc::new(RefStaging {
                slots: Mutex::new(vec![Vec::new(); size as usize]),
                barrier: Barrier::new(size as usize),
            }),
        })
    }

    /// Hand a `RefRankHandle` to rank `r`. Each handle clones the `Arc` so
    /// dropping the mesh after handles have been distributed is fine.
    pub fn rank_handle(self: &Arc<Self>, r: RankId) -> RefRankHandle {
        assert!(
            r.0 < self.size,
            "rank id {} exceeds mesh size {}",
            r.0,
            self.size
        );
        RefRankHandle {
            rank: r,
            size: self.size,
            staging: Arc::clone(&self.staging),
        }
    }
}

impl Mesh for RefMesh {
    fn rank_count(&self) -> u32 {
        self.size
    }
    fn backend(&self) -> &'static str {
        "cpu-ref"
    }
}

/// Per-rank handle to the reference mesh. Clonable; held by driver threads.
#[derive(Clone, Debug)]
pub struct RefRankHandle {
    pub rank: RankId,
    size: u32,
    staging: Arc<RefStaging>,
}

impl RefRankHandle {
    pub fn rank_count(&self) -> u32 {
        self.size
    }

    fn deposit(&self, bytes: &[u8]) {
        let mut slots = self.staging.slots.lock().unwrap();
        slots[self.rank.0 as usize] = bytes.to_vec();
    }

    /// Release the lock before barrier to avoid deadlock, then re-acquire.
    fn barrier(&self) {
        self.staging.barrier.wait();
    }
}

// ---- op traits + ref impls -------------------------------------------------

/// In-place sum-reduce across all ranks.
/// Contract: every rank calls `all_reduce` with a same-size, same-dtype buffer;
/// on return each rank's buffer holds the element-wise reduction.
pub trait AllReduce {
    fn all_reduce(&self, buf: &mut [u8], cfg: &CollectiveCfg) -> CollectiveResult<()>;
}

/// Concatenate each rank's buffer into a (rank_count × elem_count) output.
/// Input `send` is this rank's shard; `recv` is the full gathered buffer.
pub trait AllGather {
    fn all_gather(&self, send: &[u8], recv: &mut [u8], cfg: &CollectiveCfg)
        -> CollectiveResult<()>;
}

/// Element at rank `r` is received from rank `r`'s `send[r * shard..]`. The
/// per-shard size is `elem_count` (same across all ranks).
pub trait AllToAll {
    fn all_to_all(&self, send: &[u8], recv: &mut [u8], cfg: &CollectiveCfg)
        -> CollectiveResult<()>;
}

/// Fan rank `root`'s buffer out to every rank.
pub trait Broadcast {
    fn broadcast(&self, buf: &mut [u8], root: RankId, cfg: &CollectiveCfg) -> CollectiveResult<()>;
}

/// On-device AllReduce-sum: the device-pointer analog of [`AllReduce`].
/// Reduces `n_elems` `dtype` elements in place at `buf` across all ranks
/// without a host bounce. Implemented on a per-rank handle (HIP BAR1 P2P,
/// later NCCL on CUDA) carrying the rank's identity, mirroring how
/// [`RefRankHandle`] carries its rank for the byte-buffer path. `op` is
/// always `Sum` — the only reduction the forward AR path issues.
pub trait DeviceAllReduce {
    /// Per-rank device type (`HipDevice`, `CudaDevice`).
    type Device: Device;

    fn all_reduce_sum(
        &self,
        buf: DevicePtr,
        n_elems: usize,
        dtype: CollectiveDType,
        device: &Self::Device,
        stream: &<Self::Device as Device>::Stream,
    ) -> CollectiveResult<()>;
}

/// Buffer set for [`FusedAllReduce::ar_residual_rmsnorm_f16`].
#[derive(Copy, Clone, Debug)]
pub struct ArResidualRmsNormHookBuffers {
    pub residual_inout: DevicePtr,
    pub partial_f16: DevicePtr,
    pub rms_weight: DevicePtr,
    pub out_norm: DevicePtr,
}

/// Buffer set for [`FusedAllReduce::ar_postattn_residual_rmsnorm_f32_to_f16`].
#[derive(Copy, Clone, Debug)]
pub struct ArPostAttnRmsNormHookBuffers {
    pub proj_local_f32: DevicePtr,
    pub post_norm_w_f16: DevicePtr,
    pub resid_in_f16: DevicePtr,
    pub resid_out_f16: DevicePtr,
}

/// BAR1-class fused collectives: AllReduce composed with the residual-add and
/// RMSNorm epilogues into single kernels, plus an F16-payload AR-sum. A
/// capability layered on [`DeviceAllReduce`] that only a fully-peer-connected
/// device backend provides — HIP BAR1 P2P here; on CUDA, NCCL-allreduce plus a
/// separate epilogue kernel. The host-bounce path does NOT implement it;
/// callers gate on the `supports_*` predicates and fall back to the unfused
/// `ar_sum_f32 + cast + add/rmsnorm` sequence. `supports_*` is rank-based
/// (TP=2 for the residual fusions, TP ∈ {2,4} for the F16 sum / post-attn).
pub trait FusedAllReduce: DeviceAllReduce {
    fn supports_ar_sum_f16(&self) -> bool;

    /// F16-payload AR-sum: `buf = Σ peer buf[rank]` in F16, halving BAR1
    /// traffic vs the F32 `all_reduce_sum`.
    fn ar_sum_f16(
        &self,
        buf: DevicePtr,
        n_elems: usize,
        device: &Self::Device,
        stream: &<Self::Device as Device>::Stream,
    ) -> CollectiveResult<()>;

    fn supports_ar_residual_f16(&self) -> bool;

    /// In-place fused AR + residual-add: `residual_inout += Σ partial_f16`.
    fn ar_residual_f16(
        &self,
        residual_inout: DevicePtr,
        partial_f16: DevicePtr,
        n_elems: usize,
        device: &Self::Device,
        stream: &<Self::Device as Device>::Stream,
    ) -> CollectiveResult<()>;

    fn supports_ar_residual_rmsnorm_f16(&self) -> bool;

    /// Fused AR + residual-add + RMSNorm, F16 throughout.
    fn ar_residual_rmsnorm_f16(
        &self,
        bufs: ArResidualRmsNormHookBuffers,
        n_elems: usize,
        eps: f32,
        device: &Self::Device,
        stream: &<Self::Device as Device>::Stream,
    ) -> CollectiveResult<()>;

    fn supports_ar_postattn_residual_rmsnorm_f32_to_f16(&self) -> bool;

    /// Fused post-attn / post-ffn path:
    /// `resid_out = resid_in + rmsnorm(Σ proj_partial_f32, w, eps)`.
    /// `resid_out` must not alias `resid_in`.
    fn ar_postattn_residual_rmsnorm_f32_to_f16(
        &self,
        bufs: ArPostAttnRmsNormHookBuffers,
        n_rows: usize,
        n: usize,
        eps: f32,
        device: &Self::Device,
        stream: &<Self::Device as Device>::Stream,
    ) -> CollectiveResult<()>;
}

// Generic element-wise reduction over a dtype, in place on rank 0 then
// broadcast back out. This is the host-bounce pattern.
fn reduce_slots(slots: &mut [Vec<u8>], cfg: &CollectiveCfg) {
    match cfg.dtype {
        CollectiveDType::F32 => reduce_slots_typed::<f32>(slots, cfg),
        CollectiveDType::F16 => reduce_slots_typed_f16(slots, cfg),
    }
}

fn reduce_slots_typed<T: Copy + bytemuck::Pod + ReduceField>(
    slots: &mut [Vec<u8>],
    cfg: &CollectiveCfg,
) {
    let n = cfg.elem_count;
    // Copy rank 0's values into an accumulator, then fold in each other rank.
    let mut acc: Vec<T> = bytemuck::cast_slice::<u8, T>(&slots[0]).to_vec();
    for rank_buf in &slots[1..] {
        let xs: &[T] = bytemuck::cast_slice(rank_buf);
        for i in 0..n {
            acc[i] = match cfg.op {
                ReduceOp::Sum => acc[i].add(xs[i]),
                ReduceOp::Max => acc[i].max_(xs[i]),
                ReduceOp::Min => acc[i].min_(xs[i]),
            };
        }
    }
    // Write reduced vector back to every rank's slot.
    for rank_buf in slots.iter_mut() {
        rank_buf.copy_from_slice(bytemuck::cast_slice(&acc));
    }
}

fn reduce_slots_typed_f16(slots: &mut [Vec<u8>], cfg: &CollectiveCfg) {
    // Accumulate in f32 — matches RCCL's numerical behaviour well enough for
    // the correctness gate (1e-3 tol). For bit-exact parity we would switch
    // to f16 arithmetic, but that's a V2 concern.
    let n = cfg.elem_count;
    let mut acc: Vec<f32> = {
        let xs: &[half::f16] = bytemuck::cast_slice(&slots[0]);
        xs.iter().map(|x| x.to_f32()).collect()
    };
    for rank_buf in &slots[1..] {
        let xs: &[half::f16] = bytemuck::cast_slice(rank_buf);
        for i in 0..n {
            let v = xs[i].to_f32();
            acc[i] = match cfg.op {
                ReduceOp::Sum => acc[i] + v,
                ReduceOp::Max => acc[i].max(v),
                ReduceOp::Min => acc[i].min(v),
            };
        }
    }
    let out: Vec<half::f16> = acc.iter().map(|v| half::f16::from_f32(*v)).collect();
    for rank_buf in slots.iter_mut() {
        rank_buf.copy_from_slice(bytemuck::cast_slice(&out));
    }
}

trait ReduceField: Copy {
    fn add(self, other: Self) -> Self;
    fn max_(self, other: Self) -> Self;
    fn min_(self, other: Self) -> Self;
}

impl ReduceField for f32 {
    fn add(self, o: Self) -> Self {
        self + o
    }
    fn max_(self, o: Self) -> Self {
        self.max(o)
    }
    fn min_(self, o: Self) -> Self {
        self.min(o)
    }
}

impl AllReduce for RefRankHandle {
    fn all_reduce(&self, buf: &mut [u8], cfg: &CollectiveCfg) -> CollectiveResult<()> {
        check_buf(buf, cfg)?;
        self.deposit(buf);
        self.barrier(); // every rank has deposited
        if self.rank.0 == 0 {
            let mut slots = self.staging.slots.lock().unwrap();
            reduce_slots(&mut slots, cfg);
        }
        self.barrier(); // rank 0 done writing back
        let slots = self.staging.slots.lock().unwrap();
        buf.copy_from_slice(&slots[self.rank.0 as usize]);
        Ok(())
    }
}

impl AllGather for RefRankHandle {
    fn all_gather(
        &self,
        send: &[u8],
        recv: &mut [u8],
        cfg: &CollectiveCfg,
    ) -> CollectiveResult<()> {
        check_buf(send, cfg)?;
        let expected_recv = cfg.buffer_bytes() * self.size as usize;
        if recv.len() != expected_recv {
            return Err(CollectiveError::BufferLen {
                expected_elems: cfg.elem_count * self.size as usize,
                dtype: cfg.dtype,
                got: recv.len(),
            });
        }
        self.deposit(send);
        self.barrier();
        let slots = self.staging.slots.lock().unwrap();
        let shard = cfg.buffer_bytes();
        for r in 0..self.size as usize {
            recv[r * shard..(r + 1) * shard].copy_from_slice(&slots[r]);
        }
        drop(slots);
        self.barrier(); // no rank frees slots before others have read
        Ok(())
    }
}

impl AllToAll for RefRankHandle {
    fn all_to_all(
        &self,
        send: &[u8],
        recv: &mut [u8],
        cfg: &CollectiveCfg,
    ) -> CollectiveResult<()> {
        let shard_bytes = cfg.buffer_bytes();
        let total_bytes = shard_bytes * self.size as usize;
        if send.len() != total_bytes {
            return Err(CollectiveError::BufferLen {
                expected_elems: cfg.elem_count * self.size as usize,
                dtype: cfg.dtype,
                got: send.len(),
            });
        }
        if recv.len() != total_bytes {
            return Err(CollectiveError::BufferLen {
                expected_elems: cfg.elem_count * self.size as usize,
                dtype: cfg.dtype,
                got: recv.len(),
            });
        }
        self.deposit(send);
        self.barrier();
        // rank r receives from every rank s the shard `s.send[r * shard..(r+1)*shard]`.
        {
            let slots = self.staging.slots.lock().unwrap();
            let my_r = self.rank.0 as usize;
            for s in 0..self.size as usize {
                let src = &slots[s][my_r * shard_bytes..(my_r + 1) * shard_bytes];
                recv[s * shard_bytes..(s + 1) * shard_bytes].copy_from_slice(src);
            }
        }
        self.barrier();
        Ok(())
    }
}

impl Broadcast for RefRankHandle {
    fn broadcast(&self, buf: &mut [u8], root: RankId, cfg: &CollectiveCfg) -> CollectiveResult<()> {
        if root.0 >= self.size {
            return Err(CollectiveError::RankOutOfRange {
                rank: root.0,
                size: self.size,
            });
        }
        check_buf(buf, cfg)?;
        if self.rank == root {
            self.deposit(buf);
        }
        self.barrier();
        let slots = self.staging.slots.lock().unwrap();
        buf.copy_from_slice(&slots[root.0 as usize]);
        drop(slots);
        self.barrier();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn spawn_ranks<F>(size: u32, f: F)
    where
        F: Fn(RefRankHandle) + Send + Sync + 'static + Clone,
    {
        let mesh = RefMesh::new(size);
        let mut handles = Vec::with_capacity(size as usize);
        for r in 0..size {
            let h = mesh.rank_handle(RankId(r));
            let f = f.clone();
            handles.push(thread::spawn(move || f(h)));
        }
        for h in handles {
            h.join().unwrap();
        }
    }

    fn cfg_f32(n: usize) -> CollectiveCfg {
        CollectiveCfg::new(n, CollectiveDType::F32, ReduceOp::Sum)
    }

    fn f32_buf_bytes(vals: &[f32]) -> Vec<u8> {
        bytemuck::cast_slice(vals).to_vec()
    }

    fn bytes_as_f32(bytes: &[u8]) -> Vec<f32> {
        bytemuck::cast_slice::<u8, f32>(bytes).to_vec()
    }

    #[test]
    fn all_reduce_sum_mesh_4() {
        // Each rank contributes [r+1, r+2, r+3, r+4] → sum across 4 ranks:
        // [1+2+3+4, 2+3+4+5, 3+4+5+6, 4+5+6+7] = [10, 14, 18, 22].
        spawn_ranks(4, |h| {
            let r = h.rank.0 as f32;
            let input: Vec<f32> = (0..4).map(|i| r + 1.0 + i as f32).collect();
            let mut buf = f32_buf_bytes(&input);
            h.all_reduce(&mut buf, &cfg_f32(4)).unwrap();
            assert_eq!(bytes_as_f32(&buf), vec![10.0, 14.0, 18.0, 22.0]);
        });
    }

    #[test]
    fn all_reduce_degenerate_mesh_1() {
        spawn_ranks(1, |h| {
            let mut buf = f32_buf_bytes(&[1.0, 2.0, 3.0]);
            h.all_reduce(&mut buf, &cfg_f32(3)).unwrap();
            assert_eq!(bytes_as_f32(&buf), vec![1.0, 2.0, 3.0]);
        });
    }

    #[test]
    fn all_reduce_mesh_2() {
        spawn_ranks(2, |h| {
            let input: Vec<f32> = vec![h.rank.0 as f32 + 1.0; 8];
            let mut buf = f32_buf_bytes(&input);
            h.all_reduce(&mut buf, &cfg_f32(8)).unwrap();
            assert_eq!(bytes_as_f32(&buf), vec![3.0; 8]);
        });
    }

    #[test]
    fn all_gather_mesh_4_in_rank_order() {
        spawn_ranks(4, |h| {
            let send = f32_buf_bytes(&[h.rank.0 as f32 + 10.0, h.rank.0 as f32 + 20.0]);
            let mut recv = vec![0u8; 2 * 4 * 4];
            let cfg = CollectiveCfg::new(2, CollectiveDType::F32, ReduceOp::Sum);
            h.all_gather(&send, &mut recv, &cfg).unwrap();
            assert_eq!(
                bytes_as_f32(&recv),
                vec![10.0, 20.0, 11.0, 21.0, 12.0, 22.0, 13.0, 23.0],
            );
        });
    }

    #[test]
    fn all_to_all_mesh_4_permutes_as_expected() {
        // Each rank s sends shard r the value (s*10 + r). Rank r receives
        // from every s the value (s*10 + r), so recv_r = [0..10..20..30] + r.
        spawn_ranks(4, |h| {
            let s = h.rank.0 as usize;
            let send_vals: Vec<f32> = (0..4).map(|r| (s * 10 + r) as f32).collect();
            let send = f32_buf_bytes(&send_vals);
            let mut recv = vec![0u8; 4 * 4]; // 1 elem per shard × 4 ranks × 4 bytes
            let cfg = CollectiveCfg::new(1, CollectiveDType::F32, ReduceOp::Sum);
            h.all_to_all(&send, &mut recv, &cfg).unwrap();
            let got = bytes_as_f32(&recv);
            let my_r = h.rank.0 as usize;
            let expected: Vec<f32> = (0..4).map(|s| (s * 10 + my_r) as f32).collect();
            assert_eq!(got, expected);
        });
    }

    #[test]
    fn broadcast_mesh_4_from_rank_2() {
        spawn_ranks(4, |h| {
            let mut buf = if h.rank.0 == 2 {
                f32_buf_bytes(&[7.0, 8.0, 9.0])
            } else {
                f32_buf_bytes(&[-1.0, -1.0, -1.0])
            };
            let cfg = CollectiveCfg::new(3, CollectiveDType::F32, ReduceOp::Sum);
            h.broadcast(&mut buf, RankId(2), &cfg).unwrap();
            assert_eq!(bytes_as_f32(&buf), vec![7.0, 8.0, 9.0]);
        });
    }

    #[test]
    fn all_reduce_f16_mesh_2() {
        spawn_ranks(2, |h| {
            let v = if h.rank.0 == 0 { 1.5 } else { 2.25 };
            let input: Vec<half::f16> = vec![half::f16::from_f32(v); 4];
            let mut buf = bytemuck::cast_slice::<half::f16, u8>(&input).to_vec();
            let cfg = CollectiveCfg::new(4, CollectiveDType::F16, ReduceOp::Sum);
            h.all_reduce(&mut buf, &cfg).unwrap();
            let out: &[half::f16] = bytemuck::cast_slice(&buf);
            for x in out {
                assert!((x.to_f32() - 3.75).abs() < 1e-3);
            }
        });
    }

    #[test]
    fn buffer_len_mismatch_is_reported() {
        spawn_ranks(2, |h| {
            let mut buf = vec![0u8; 13]; // not a multiple of f32
            let cfg = cfg_f32(4);
            let err = h.all_reduce(&mut buf, &cfg).unwrap_err();
            assert!(format!("{err}").contains("buffer length"));
        });
    }
}
