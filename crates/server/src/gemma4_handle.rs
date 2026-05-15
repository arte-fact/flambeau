//! Gemma4 server binding.
//!
//! Mirrors the qwen3-moe `model_handle` shape: a `Gemma4HipModel`
//! marker plus a `Gemma4HipSession` that wraps `Box<dyn ModelDriver>`.
//!
//! Phase 12.9 MVP: gemma4 drivers (`Gemma4PpDriver` / `Gemma4TpDriver`
//! / `Gemma4HybridDriver`) bundle weights + KV cache state inside one
//! struct. That means *one* driver instance backs *one* request slot;
//! multi-slot batched decode requires splitting weights from session
//! state and ships in a follow-up. For now, `FLAMBEAU_INFLIGHT_SLOTS=1`
//! is enforced at boot for gemma4 models.

#![cfg(feature = "hip")]

use anyhow::{bail, Result};
use flambeau_backend_hip::HipCluster;
use flambeau_gemma4::Gemma4Config;
use flambeau_qwen3_moe::forward::ShardedForwardPrefillScratchTp;
use flambeau_runtime::ModelDriver;

use crate::model::{BoundaryCallback, HybridHipSession, PpHipSession, TpHipSession};
use crate::model_handle::{HipModel, HipSession};

/// Marker model handle for gemma4. The actual driver lives on the
/// per-request `Gemma4HipSession` (gemma4 weights + KV are bundled
/// inside the driver — splitting them is a follow-up).
pub struct Gemma4HipModel {
    pub cfg: Gemma4Config,
    pub topology: &'static str,
}

impl HipModel for Gemma4HipModel {
    fn topology(&self) -> &'static str {
        self.topology
    }
    fn is_gemma4(&self) -> bool {
        true
    }
}

/// Per-request gemma4 session. Owns the entire driver instance —
/// because gemma4's weights and KV-state are bundled in one struct,
/// each inflight slot has its own driver. Use
/// `FLAMBEAU_INFLIGHT_SLOTS=1` until weights/session split lands.
pub struct Gemma4HipSession {
    pub driver: Box<dyn ModelDriver>,
}

impl HipSession for Gemma4HipSession {
    fn prefill_logits(
        &mut self,
        _cluster: &HipCluster,
        prompt_ids: &[u32],
        start_position: usize,
        logits_out: &mut Vec<f32>,
        _tp_pool_prefill: Option<&mut ShardedForwardPrefillScratchTp>,
        _on_boundary: Option<BoundaryCallback<'_>>,
        _prefill_ubatch: usize,
    ) -> Result<()> {
        // Gemma4 drivers own their own cluster + stream and don't yet
        // chunk-prefill via `prefill_ubatch`; we pass the whole prompt
        // through `forward_prefill_logits`. The prefix-cache
        // `on_boundary` callback never fires for gemma4 (prefix cache
        // gated off via `as_pp/as_tp/as_hybrid` returning None — see
        // routes.rs `prefix_cache_try_restore` early-out).
        self.driver
            .forward_prefill_logits(prompt_ids, start_position, logits_out)
    }

    fn reset_for_next_request(&mut self, _cluster: &HipCluster) -> Result<()> {
        // V1: gemma4 drivers don't yet expose a KV-reset hook on the
        // ModelDriver trait. The server pre-allocates one driver per
        // slot and runs one request through it; subsequent requests on
        // the same slot need explicit KV reset support (follow-up).
        bail!(
            "Gemma4HipSession::reset_for_next_request: KV-reset on gemma4 drivers \
             not yet wired through ModelDriver. Restart the server for a fresh KV."
        )
    }

    fn dispose(self: Box<Self>, _cluster: &HipCluster) -> Result<()> {
        let mut driver = self.driver;
        driver.dispose()
    }

    fn as_pp(&self) -> Option<&PpHipSession> {
        None
    }
    fn as_pp_mut(&mut self) -> Option<&mut PpHipSession> {
        None
    }
    fn as_tp(&self) -> Option<&TpHipSession> {
        None
    }
    fn as_tp_mut(&mut self) -> Option<&mut TpHipSession> {
        None
    }
    fn as_hybrid(&self) -> Option<&HybridHipSession> {
        None
    }
    fn as_hybrid_mut(&mut self) -> Option<&mut HybridHipSession> {
        None
    }

    fn as_gemma4_driver_mut(&mut self) -> Option<&mut dyn ModelDriver> {
        Some(self.driver.as_mut())
    }
}

/// Build a `LoadedModel` (`Arc<dyn HipModel>`) for gemma4 from a
/// `Gemma4Config` + topology string. Mirror of qwen3-moe's
/// `LoadedModel` construction at server boot.
pub fn build_gemma4_loaded_model(
    cfg: Gemma4Config,
    topology: &'static str,
) -> crate::model::LoadedModel {
    std::sync::Arc::new(Gemma4HipModel { cfg, topology })
}

/// Wrap a constructed `Gemma4*Driver` (as a `Box<dyn ModelDriver>`)
/// into a HipSession trait object for the inflight pool.
pub fn wrap_gemma4_driver(driver: Box<dyn ModelDriver>) -> Box<dyn HipSession> {
    Box::new(Gemma4HipSession { driver })
}

/// Best-effort topology-from-arch hint. Used by `is_gemma4_arch` style
/// gates at the boot path.
pub fn arch_matches(arch: &str) -> bool {
    matches!(
        arch,
        "gemma4" | "gemma4-26b-a4b" | "gemma4-31b" | "gemma4-9b" | "gemma4-2b"
    )
}
