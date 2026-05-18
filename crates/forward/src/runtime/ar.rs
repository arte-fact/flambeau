//! Host-roundtrip AR coordinator + peer-buffer helpers. Extracted
//! verbatim from the *_parity.rs tests.

use anyhow::Result;
use flambeau_backend_hip::{HipDevice, HipStream};
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

/// Shared host F16 staging slot between consecutive PP / hybrid stages.
/// Length `hidden`. Built once, threaded into both the sender and
/// receiver rank's contexts.
pub type PeerBuffer = Arc<Mutex<Vec<half::f16>>>;

pub fn new_peer_buffer(hidden: usize) -> PeerBuffer {
    Arc::new(Mutex::new(vec![half::f16::ZERO; hidden]))
}
