//! Topology-customisation hooks.
//!
//! `TopologyHooks` is the small surface each topology impl provides on
//! top of the shared composite engine. Every method has a sensible
//! default that matches **single-device** semantics (no AR, no
//! peer-copy, every rank owns everything). PP / TP / Hybrid override
//! the methods they need.
//!
//! The trait stays small on purpose: we resist the urge to push entire
//! composite bodies through hooks. Hooks are for the discrete points
//! where topologies actually differ — AllReduce after a row-parallel
//! matmul, peer-copy at a PP stage boundary, rank guards for embed /
//! output_head. If you find yourself adding a hook that one topology
//! returns `None` from, the abstraction is wrong — push back.

use anyhow::Result;
use flambeau_backend_hip::{HipDevice, HipStream};
use flambeau_core::DevicePtr;

/// Per-topology hook surface. Default impls match SingleDevice semantics
/// (no AR, no peer-copy, every rank owns everything).
pub trait TopologyHooks {
    /// Reduce-sum `n_elems` F32 values in-place across ranks on the
    /// given device buffer. Called after row-parallel matmuls
    /// (output_proj in attention, down_proj in dense FFN) where each
    /// rank produces an `[hidden]` F32 partial that must be summed
    /// before being cast back to F16 delta.
    ///
    /// Default: no-op (SingleDevice / PP).
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

/// All-default hook bundle used by SingleDevice (and by tests that want
/// to exercise composites with no topology behaviour).
pub struct NoopHooks;

impl TopologyHooks for NoopHooks {}
