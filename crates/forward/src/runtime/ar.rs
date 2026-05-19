//! AllReduce coordinators. Two backends:
//! * [`ArCoordinator`] / [`ar_sum_f32`] — host-bounce (DtoH → CPU sum
//!   → HtoD). Used as the universal fallback.
//! * [`BarArCoordinator`] / [`bar_ar_sum_f32`] — BAR1 P2P. Each rank
//!   publishes its partial pointer to a shared slab, all ranks
//!   synchronize, then each rank launches its own
//!   `sum_tp{2,4}_f32_rank` kernel on its own stream. ~100× less
//!   per-call overhead than the host bounce on TP2 because there
//!   are no DtoH/HtoD bytes — just BAR1 reads inside the kernel.

use anyhow::Result;
use flambeau_backend_hip::{BarP2pAllReduce, HipDevice, HipStream};
use flambeau_core::{CopyDirection, Device, DevicePtr};
use std::sync::{Arc, Barrier, Mutex};

/// One coordinator per stage / per TP cluster. Workers in the same
/// stage share an `Arc<ArCoordinator>` and call `ar_sum` through the
/// TpHooks / HybridHooks callback.
pub struct ArCoordinator {
    pub n_ranks: usize,
    partials: Mutex<Vec<Option<Vec<f32>>>>,
    barrier: Barrier,
}

impl ArCoordinator {
    pub fn new(n_ranks: usize) -> Self {
        Self {
            n_ranks,
            partials: Mutex::new(vec![None; n_ranks]),
            barrier: Barrier::new(n_ranks),
        }
    }
}

/// DtoH this rank's partial → barrier → host sum → barrier → reset
/// (rank 0) → barrier → HtoD the summed result back.
pub fn ar_sum_f32(
    coord: &ArCoordinator,
    rank: usize,
    buf: DevicePtr,
    n_elems: usize,
    device: &HipDevice,
    stream: &HipStream,
) -> Result<()> {
    let bytes = n_elems * 4;
    let mut host = vec![0.0_f32; n_elems];
    // SAFETY: host has n_elems*4 bytes; buf owns the same.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            DevicePtr(host.as_mut_ptr() as usize),
            buf,
            bytes,
        )?;
    }
    flambeau_core::Stream::synchronize(stream)?;
    {
        let mut p = coord.partials.lock().unwrap();
        p[rank] = Some(host);
    }
    coord.barrier.wait();

    let summed: Vec<f32> = {
        let p = coord.partials.lock().unwrap();
        let mut s = p[0].as_ref().unwrap().clone();
        for r in 1..coord.n_ranks {
            let other = p[r].as_ref().unwrap();
            for i in 0..n_elems {
                s[i] += other[i];
            }
        }
        s
    };
    coord.barrier.wait();
    if rank == 0 {
        let mut p = coord.partials.lock().unwrap();
        for r in 0..coord.n_ranks {
            p[r] = None;
        }
    }
    coord.barrier.wait();

    // SAFETY: buf owns bytes; summed is hidden host F32.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            buf,
            DevicePtr(summed.as_ptr() as usize),
            bytes,
        )?;
    }
    flambeau_core::Stream::synchronize(stream)?;
    Ok(())
}

/// Build the boxed callback that TpHooks / HybridHooks expect.
pub fn make_ar_callback(
    coord: Arc<ArCoordinator>,
    rank: usize,
) -> Box<
    dyn FnMut(usize, usize, DevicePtr, usize, &HipDevice, &HipStream) -> Result<()> + Send,
> {
    Box::new(move |_r, _nr, buf, n_elems, dev, st| {
        ar_sum_f32(&coord, rank, buf, n_elems, dev, st)
    })
}

/// BAR1 P2P AllReduce coordinator. Each rank's worker thread calls
/// [`bar_ar_sum_f32`] from inside its `TopologyHooks::ar_sum_f32`
/// callback; the call exchanges per-rank `partial` device pointers,
/// drains the producer stream, then launches `sum_tp{2,4}_f32_rank` on
/// the calling rank's stream. The kernel reads peer pointers via BAR1
/// — no DtoH/HtoD bytes leave the device.
pub struct BarArCoordinator {
    pub bar: Arc<BarP2pAllReduce>,
    partials: Mutex<Vec<Option<DevicePtr>>>,
    barrier: Barrier,
}

impl BarArCoordinator {
    pub fn new(bar: Arc<BarP2pAllReduce>) -> Self {
        let n = bar.ranks();
        Self {
            bar,
            partials: Mutex::new(vec![None; n]),
            barrier: Barrier::new(n),
        }
    }

    pub fn ranks(&self) -> usize {
        self.bar.ranks()
    }
}

/// BAR1 P2P AR-sum. Steps: producer-stream sync → publish own buf →
/// barrier → snap peer pointer(s) → launch this rank's kernel →
/// barrier → reset slab (rank 0).
pub fn bar_ar_sum_f32(
    coord: &BarArCoordinator,
    rank: usize,
    buf: DevicePtr,
    n_elems: usize,
    _device: &HipDevice,
    stream: &HipStream,
) -> Result<()> {
    let n_ranks = coord.ranks();
    if n_ranks == 1 {
        return Ok(());
    }
    // Drain own producer-stream work so the partial is committed
    // before any peer reads it via BAR1.
    flambeau_core::Stream::synchronize(stream)?;

    {
        let mut p = coord.partials.lock().unwrap();
        p[rank] = Some(buf);
    }
    coord.barrier.wait();

    let peers_snapshot: Vec<DevicePtr> = {
        let p = coord.partials.lock().unwrap();
        (0..n_ranks)
            .map(|r| p[r].expect("BarArCoordinator: peer pointer unpublished"))
            .collect()
    };

    // SAFETY: partials are pool-owned DevicePtrs alive for the request;
    // producer ordering held by step-1 sync; each rank launches on its
    // own stream; BarP2pAllReduce validates per-rank cluster device.
    unsafe {
        match n_ranks {
            2 => coord
                .bar
                .sum_tp2_f32_rank(rank, peers_snapshot[rank], peers_snapshot[1 - rank], n_elems as u32, stream)?,
            4 => {
                let peers = [
                    peers_snapshot[(rank + 1) % 4],
                    peers_snapshot[(rank + 2) % 4],
                    peers_snapshot[(rank + 3) % 4],
                ];
                coord
                    .bar
                    .sum_tp4_f32_rank(rank, peers_snapshot[rank], peers, n_elems as u32, stream)?
            }
            other => anyhow::bail!("BarArCoordinator: unsupported n_ranks={other}"),
        }
    }

    coord.barrier.wait();
    if rank == 0 {
        let mut p = coord.partials.lock().unwrap();
        for r in 0..n_ranks {
            p[r] = None;
        }
    }
    coord.barrier.wait();
    Ok(())
}

/// BAR1 P2P fused AR + residual-add. `partial_f16` is the
/// rank-local F16 partial (caller has cast from F32 if needed).
/// `residual_inout_f16` is the rank-local residual; updated in place
/// to `residual + Σ peer-partials`. TP=2 only.
pub fn bar_ar_residual_f16(
    coord: &BarArCoordinator,
    rank: usize,
    residual_inout_f16: DevicePtr,
    partial_f16: DevicePtr,
    n_elems: usize,
    _device: &HipDevice,
    stream: &HipStream,
) -> Result<()> {
    let n_ranks = coord.ranks();
    if n_ranks == 1 {
        // SD path: caller must do residual_add separately. We don't
        // touch resid here so the no-fuse fallback semantics hold.
        return Ok(());
    }
    if n_ranks != 2 {
        anyhow::bail!("bar_ar_residual_f16: only TP=2 supported (got {n_ranks})");
    }
    flambeau_core::Stream::synchronize(stream)?;
    {
        let mut p = coord.partials.lock().unwrap();
        p[rank] = Some(partial_f16);
    }
    coord.barrier.wait();
    let peers_snapshot: Vec<DevicePtr> = {
        let p = coord.partials.lock().unwrap();
        (0..n_ranks)
            .map(|r| p[r].expect("BarArCoordinator: peer pointer unpublished"))
            .collect()
    };
    // SAFETY: partials are pool-owned F16 alive for the request;
    // producer-stream sync drained own writes before publish.
    unsafe {
        coord.bar.residual_tp2_rank(
            rank,
            residual_inout_f16,
            peers_snapshot[0],
            peers_snapshot[1],
            n_elems as u32,
            stream,
        )?;
    }
    coord.barrier.wait();
    if rank == 0 {
        let mut p = coord.partials.lock().unwrap();
        for r in 0..n_ranks {
            p[r] = None;
        }
    }
    coord.barrier.wait();
    Ok(())
}

/// BAR1 P2P fused AR + residual-add + RMSNorm. `partial_f16` is the
/// rank-local F16 partial; `residual_inout_f16` is the per-rank
/// residual (updated in-place to `residual + Σ peers`); `out_norm`
/// receives the rmsnormed result. TP=2 only.
#[allow(clippy::too_many_arguments)]
pub fn bar_ar_residual_rmsnorm_f16(
    coord: &BarArCoordinator,
    rank: usize,
    residual_inout_f16: DevicePtr,
    partial_f16: DevicePtr,
    rms_weight: DevicePtr,
    out_norm: DevicePtr,
    n_elems: usize,
    eps: f32,
    _device: &HipDevice,
    stream: &HipStream,
) -> Result<()> {
    let n_ranks = coord.ranks();
    if n_ranks != 2 {
        anyhow::bail!("bar_ar_residual_rmsnorm_f16: only TP=2 supported (got {n_ranks})");
    }
    flambeau_core::Stream::synchronize(stream)?;
    {
        let mut p = coord.partials.lock().unwrap();
        p[rank] = Some(partial_f16);
    }
    coord.barrier.wait();
    let peers_snapshot: Vec<DevicePtr> = {
        let p = coord.partials.lock().unwrap();
        (0..n_ranks)
            .map(|r| p[r].expect("BarArCoordinator: peer pointer unpublished"))
            .collect()
    };
    // SAFETY: partials are pool-owned F16 alive for the request;
    // producer-stream sync drained own writes before publish.
    unsafe {
        coord.bar.residual_rmsnorm_tp2_rank(
            rank,
            residual_inout_f16,
            peers_snapshot[0],
            peers_snapshot[1],
            rms_weight,
            out_norm,
            n_elems as u32,
            eps,
            stream,
        )?;
    }
    coord.barrier.wait();
    if rank == 0 {
        let mut p = coord.partials.lock().unwrap();
        for r in 0..n_ranks {
            p[r] = None;
        }
    }
    coord.barrier.wait();
    Ok(())
}

/// Build the boxed callback that TpHooks / HybridHooks expect, with
/// the BAR1 P2P backend.
pub fn make_bar_ar_callback(
    coord: Arc<BarArCoordinator>,
    rank: usize,
) -> Box<
    dyn FnMut(usize, usize, DevicePtr, usize, &HipDevice, &HipStream) -> Result<()> + Send,
> {
    Box::new(move |_r, _nr, buf, n_elems, dev, st| {
        bar_ar_sum_f32(&coord, rank, buf, n_elems, dev, st)
    })
}

/// Shared host F16 staging slot between consecutive PP / hybrid stages.
/// Length `hidden`. Built once, threaded into both the sender and
/// receiver rank's contexts.
pub type PeerBuffer = Arc<Mutex<Vec<half::f16>>>;

pub fn new_peer_buffer(hidden: usize) -> PeerBuffer {
    Arc::new(Mutex::new(vec![half::f16::ZERO; hidden]))
}
