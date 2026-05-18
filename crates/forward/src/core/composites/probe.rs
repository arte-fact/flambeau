//! Env-gated per-layer hidden-state probe. Set `FLAMBEAU_PROBE_LAYER=<N>`
//! to dump the first 16 F16 values of the residual after the `N`-th
//! call here (counted by the static counter, reset on `FLAMBEAU_PROBE_RESET`).

use anyhow::Result;
use flambeau_backend_hip::{HipDevice, HipStream};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use std::sync::atomic::{AtomicUsize, Ordering};

static CALL_COUNTER: AtomicUsize = AtomicUsize::new(0);

pub fn dump_if_set(
    tag: &str,
    ptr: DevicePtr,
    hidden: usize,
    device: &HipDevice,
    stream: &HipStream,
) -> Result<()> {
    let Some(target) = std::env::var("FLAMBEAU_PROBE_LAYER")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    else {
        return Ok(());
    };
    let idx = CALL_COUNTER.fetch_add(1, Ordering::Relaxed);
    if idx != target {
        return Ok(());
    }
    let n = hidden.min(16);
    let mut host = vec![0u16; n];
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            DevicePtr(host.as_mut_ptr() as usize),
            ptr,
            n * 2,
        )?;
    }
    Stream::synchronize(stream)?;
    let head: Vec<f32> = host
        .iter()
        .map(|&b| half::f16::from_bits(b).to_f32())
        .collect();
    eprintln!("[{tag} probe call={idx}] hidden[0..{n}] = {head:?}");
    Ok(())
}
