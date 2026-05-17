//! Topology customisation surface. Defaults match single-device
//! semantics; PP / TP / Hybrid override what they need. Don't push
//! whole composite bodies through here — hooks are for the discrete
//! points where topologies actually differ.

use anyhow::Result;
use flambeau_backend_hip::{HipDevice, HipStream};
use flambeau_core::DevicePtr;

pub trait TopologyHooks {
    /// In-place reduce-sum across ranks. Called after row-parallel
    /// matmuls (attn output_proj, ffn down).
    fn ar_sum_f32(
        &mut self,
        buf: DevicePtr,
        n_elems: usize,
        device: &HipDevice,
        stream: &HipStream,
    ) -> Result<()> {
        let _ = (buf, n_elems, device, stream);
        Ok(())
    }
}

pub struct NoopHooks;

impl TopologyHooks for NoopHooks {}
