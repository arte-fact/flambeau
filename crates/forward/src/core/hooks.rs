//! Topology customisation surface. Defaults match single-device
//! semantics; PP / TP / Hybrid override what they need. Don't push
//! whole composite bodies through here — hooks are for the discrete
//! points where topologies actually differ.

use anyhow::Result;
use flambeau_backend_hip::{HipDevice, HipStream};
use flambeau_core::DevicePtr;

/// Buffer set for [`TopologyHooks::ar_residual_rmsnorm_f16`].
#[derive(Copy, Clone, Debug)]
pub struct ArResidualRmsNormHookBuffers {
    pub residual_inout: DevicePtr,
    pub partial_f16: DevicePtr,
    pub rms_weight: DevicePtr,
    pub out_norm: DevicePtr,
}

/// Buffer set for [`TopologyHooks::ar_postattn_residual_rmsnorm_f32_to_f16`].
#[derive(Copy, Clone, Debug)]
pub struct ArPostAttnRmsNormHookBuffers {
    pub proj_local_f32: DevicePtr,
    pub post_norm_w_f16: DevicePtr,
    pub resid_in_f16: DevicePtr,
    pub resid_out_f16: DevicePtr,
}

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

    fn supports_ar_residual_rmsnorm_f16(&self) -> bool {
        false
    }

    /// Whether [`Self::ar_sum_f16`] is available. SD/PP return false;
    /// BAR1 TP=2 / TP=4 return true. Gemma4's attn/FFN post-norm
    /// paths use this when the safety predicate allows the F16
    /// payload (caller checks `output_proj_safe_for_f16_ar`).
    fn supports_ar_sum_f16(&self) -> bool {
        false
    }

    /// F16-payload AR-sum: `buf = Σ peer buf[rank]` in F16. Halves
    /// BAR1 traffic vs [`Self::ar_sum_f32`]. Default impl bails;
    /// callers MUST gate with [`Self::supports_ar_sum_f16`].
    fn ar_sum_f16(
        &mut self,
        buf: DevicePtr,
        n_elems: usize,
        device: &HipDevice,
        stream: &HipStream,
    ) -> Result<()> {
        let _ = (buf, n_elems, device, stream);
        anyhow::bail!(
            "TopologyHooks::ar_sum_f16: unsupported on this topology — \
             check supports_ar_sum_f16() before calling"
        )
    }

    fn ar_residual_rmsnorm_f16(
        &mut self,
        bufs: ArResidualRmsNormHookBuffers,
        n_elems: usize,
        eps: f32,
        device: &HipDevice,
        stream: &HipStream,
    ) -> Result<()> {
        let _ = (bufs, n_elems, eps, device, stream);
        anyhow::bail!(
            "TopologyHooks::ar_residual_rmsnorm_f16: unsupported — gate with \
             supports_ar_residual_rmsnorm_f16() first"
        )
    }

    /// Whether [`Self::ar_postattn_residual_rmsnorm_f32_to_f16`] is
    /// available. SD/PP return false; BAR1 TP=2 / TP=4 return true
    /// when per-row hidden ≤ 8192 (kernel's per-thread register cap).
    fn supports_ar_postattn_residual_rmsnorm_f32_to_f16(&self) -> bool {
        false
    }

    /// Fused gemma4 post-attn / post-ffn path:
    /// `resid_out = resid_in + rmsnorm(Σ proj_partial_f32, post_norm_w, eps)`.
    /// Collapses the `ar_sum_f32` + `rmsnorm_f32_to_f16_add_residual`
    /// 2-launch sequence into one. `resid_out` must NOT alias
    /// `resid_in`. Default bails; gate with the `supports_…` flag.
    fn ar_postattn_residual_rmsnorm_f32_to_f16(
        &mut self,
        bufs: ArPostAttnRmsNormHookBuffers,
        n_rows: usize,
        n: usize,
        eps: f32,
        device: &HipDevice,
        stream: &HipStream,
    ) -> Result<()> {
        let _ = (bufs, n_rows, n, eps, device, stream);
        anyhow::bail!(
            "TopologyHooks::ar_postattn_residual_rmsnorm_f32_to_f16: unsupported — gate \
             with supports_ar_postattn_residual_rmsnorm_f32_to_f16() first"
        )
    }
}

pub struct NoopHooks;

impl TopologyHooks for NoopHooks {}
