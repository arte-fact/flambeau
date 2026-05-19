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

    /// Whether [`Self::ar_residual_f16`] is available — i.e., BAR1 P2P
    /// is engaged on a fully-connected peer-access matrix. SD/PP and
    /// host-bounce TP return `false`; BAR1 TP=2 returns `true`.
    fn supports_ar_residual_f16(&self) -> bool {
        false
    }

    /// In-place fused AR + residual-add:
    /// `residual_inout = residual_inout + Σ peer-partials`. F16
    /// throughout. Bit-equivalent to:
    /// 1. `ar_sum_f16(partial)` across ranks
    /// 2. `add_f16(residual_inout, partial, residual_inout)`
    ///
    /// Default impl is a hard bail — callers MUST gate with
    /// [`Self::supports_ar_residual_f16`] and fall back to the
    /// `ar_sum_f32 + cast + add_f16` path when it's false.
    fn ar_residual_f16(
        &mut self,
        residual_inout: DevicePtr,
        partial_f16: DevicePtr,
        n_elems: usize,
        device: &HipDevice,
        stream: &HipStream,
    ) -> Result<()> {
        let _ = (residual_inout, partial_f16, n_elems, device, stream);
        anyhow::bail!(
            "TopologyHooks::ar_residual_f16: unsupported on this topology — \
             check supports_ar_residual_f16() before calling"
        )
    }
}

pub struct NoopHooks;

impl TopologyHooks for NoopHooks {}
