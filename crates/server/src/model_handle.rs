//! Server-side `Model` + `Session` trait surface.
//!
//! - `Model` — load-time entry. Owns weights + per-topology auxiliary
//!   state. Reports topology + arch capability hints.
//! - `Session` — per-request handle. Owns KV caches + scratches and a
//!   back-reference to its parent `Model`. `prefill_logits` advances
//!   generation; `dispose` frees device buffers.
//!
//! Concrete impls (`Qwen3MoeOwnedSession`, `Gemma4Session`) live in the
//! arch-specific glue modules.

#![cfg(feature = "hip")]

use anyhow::Result;
use flambeau_backend_hip::HipCluster;
use flambeau_qwen3_moe::forward::ShardedForwardPrefillScratchTp;
use flambeau_qwen3_moe::session::KvLayout;

use crate::model::{
    BoundaryCallback, HybridHipModel, HybridHipSession, Inflight, LoadedModel, PpHipModel,
    PpHipSession, TpHipModel, TpHipSession,
};

pub trait Model: Send + Sync + 'static {
    /// Topology label for handler metrics: `"pp"`, `"tp"`, `"pp+tp"`.
    fn topology(&self) -> &'static str;

    /// Concrete-type accessors. Each topology overrides exactly one of
    /// these to return `Some(self)`; the others stay at the default
    /// `None`. Server call sites use these in place of pattern-matching
    /// on a closed enum, so adding a new model topology in the future
    /// only requires implementing `Model` (no enum-variant churn).
    fn as_pp(&self) -> Option<&PpHipModel> {
        None
    }
    fn as_tp(&self) -> Option<&TpHipModel> {
        None
    }
    fn as_hybrid(&self) -> Option<&HybridHipModel> {
        None
    }

    /// True when the model arch supports the batched-decode scheduler.
    /// Qwen3-moe (PP/TP/Hybrid) overrides to true; gemma4 + future N=1
    /// archs leave it false so the legacy single-stream decode handler
    /// is selected. JSON / logprobs paths bypass the scheduler
    /// regardless via the orthogonal gates in `scheduler_can_engage`.
    fn supports_scheduler_batching(&self) -> bool {
        false
    }

    /// True when the model arch requires the TP/Hybrid prefill
    /// serialiser lock held across a `prefill_logits` call (caps peak
    /// scratch alloc to one chunk's worth across N concurrent
    /// requests). Qwen3-moe TP + Hybrid override to true.
    fn requires_prefill_serialiser(&self) -> bool {
        false
    }

    /// True when the model arch consumes a pre-allocated TP prefill
    /// scratch (`ShardedForwardPrefillScratchTp`) handed in via
    /// `Session::prefill_logits`'s `tp_pool_prefill` parameter. Only
    /// qwen3-moe TP returns true.
    fn requires_tp_prefill_scratch(&self) -> bool {
        false
    }

    /// Arch-specific byte-level chat-template fragments that should
    /// stop generation when present in the decoded text. Mirrored from
    /// `Session::chat_stop_markers`; lives here too so the decode
    /// loop and `finalise` can read it via `&state.model` without
    /// holding the inflight mutex. Default `&[]`.
    fn chat_stop_markers(&self) -> &'static [&'static str] {
        &[]
    }

    /// Batched-decode entry point. Default handles N=1 only by
    /// delegating to [`Session::decode_one_logits`] — multi-slot archs
    /// (qwen3-moe PP/TP/Hybrid) override this method.
    fn forward_decode_batched(
        &self,
        _state: &crate::routes::ServerState,
        inflights: &mut [&mut dyn Session],
        slots: &[flambeau_qwen3_moe::forward::BatchSlot],
        logits_refs: &mut [&mut Vec<f32>],
    ) -> Result<()> {
        if slots.len() != 1 {
            anyhow::bail!(
                "Model::forward_decode_batched: N={} not supported by this arch \
                 (default impl is N=1 only). Run with `FLAMBEAU_INFLIGHT_SLOTS=1` \
                 or override the trait method.",
                slots.len(),
            );
        }
        let slot = &slots[0];
        let out: &mut Vec<f32> = logits_refs[0];
        out.clear();
        inflights[0].decode_one_logits(slot.token_id, slot.position, out)
    }
}

pub trait Session: Send {
    /// Ingest a (chunked-as-needed) prompt and write the last
    /// position's logits into `logits_out`. `start_position` is the
    /// absolute position of `prompt_ids[0]` inside the original full
    /// prompt — `0` for a fresh request, `> 0` after a prefix-cache
    /// restore covering `[0..start_position)`. `tp_pool_prefill` (TP
    /// only) lets the caller share a pre-allocated scratch when it
    /// already holds the prefill serialiser. `on_boundary` (PP / TP
    /// only) fires after every non-final chunk for prefix-cache
    /// snapshotting.
    #[allow(clippy::too_many_arguments)]
    fn prefill_logits(
        &mut self,
        prompt_ids: &[u32],
        start_position: usize,
        logits_out: &mut Vec<f32>,
        tp_pool_prefill: Option<&mut ShardedForwardPrefillScratchTp>,
        on_boundary: Option<BoundaryCallback<'_>>,
        prefill_ubatch: usize,
    ) -> Result<()>;

    fn reset_for_next_request(&mut self) -> Result<()>;

    fn dispose(self: Box<Self>) -> Result<()>;

    /// Single-token decode. Default bails — only archs whose
    /// `Model::forward_decode_batched` falls through to the trait
    /// default need a real impl (gemma4 today). qwen3-moe sessions
    /// take the `as_pp/_tp/_hybrid` branch in
    /// `Model::forward_decode_batched` and never reach here.
    fn decode_one_logits(
        &mut self,
        _token: u32,
        _position: usize,
        _logits_out: &mut Vec<f32>,
    ) -> Result<()> {
        anyhow::bail!(
            "Session::decode_one_logits: no impl on this session type. \
             Either override or route through `Model::forward_decode_batched`."
        )
    }

    /// Concrete-session accessors, mirror of `Model::as_pp`. Server
    /// uses these in spec-decode init / GPU-sampler scratch
    /// resolution / batched-dispatch field access without matching on
    /// the now-retired `Inflight` enum.
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

    /// Phase 12.9 — gemma4 driver accessor. `Gemma4Session` returns
    /// `Some(&mut dyn ModelDriver)`; qwen3-moe sessions return `None`.
    /// Routes.rs uses this to dispatch decode through the gemma4
    /// `forward_one_token_logits` path (N=1 only until weights/session
    /// split lands).
    fn as_gemma4_driver_mut(&mut self) -> Option<&mut dyn flambeau_runtime::ModelDriver> {
        None
    }

    /// Gemma4 mandates BOS prepended to every prompt; the server-side
    /// `state.tokenizer.encode` does not add specials, so the prefill
    /// path consults this accessor and prepends when present. Returns
    /// `None` for non-gemma4 sessions (or gemma4 sessions constructed
    /// without a BOS id).
    fn gemma4_bos_id(&self) -> Option<u32> {
        None
    }

    /// Arch-specific byte-level chat-template fragments that should
    /// stop generation when they appear in the decoded text. Used by
    /// `routes.rs::run_completion_blocking_ids` and its sibling
    /// scheduler-aware variant for mid-flight string-stop detection,
    /// and by `finalise` to truncate any leak that slipped past the
    /// in-loop check. Default `&[]` (no extra markers); gemma4
    /// implementations return their template's turn / channel / EOS
    /// fragments. Lets routes.rs stay arch-agnostic.
    fn chat_stop_markers(&self) -> &'static [&'static str] {
        &[]
    }
}

/// Self-sufficient qwen3-moe session: bundles an `Inflight`
/// (KV state + scratches) with a back-reference to its parent
/// `LoadedModel` and an `Arc<HipCluster>` clone so the trait methods
/// don't need a cluster passed in. Constructed via
/// [`create_qwen3moe_session`]; the server holds it as
/// `Box<dyn Session>` so call sites stay arch-agnostic.
pub struct Qwen3MoeOwnedSession {
    pub model: LoadedModel,
    pub inflight: Inflight,
    /// Owned cluster Arc, shared with `ServerState.cluster` and
    /// `Inflight`'s per-rank scratches. Lets `Session` trait methods
    /// run without a `cluster: &HipCluster` parameter.
    pub cluster: std::sync::Arc<HipCluster>,
}

/// Build a per-request session bound to `model`. `prefill_ubatch` sizes
/// the PP prefill scratch (ignored for TP/Hybrid); `kv_layout` selects
/// between F16 / Q8 / turbo-quant KV. `cluster` is cloned-Arc'd onto
/// the returned session so trait methods can run without a cluster
/// parameter.
pub fn create_qwen3moe_session(
    model: LoadedModel,
    cluster: std::sync::Arc<HipCluster>,
    prefill_ubatch: usize,
    kv_layout: KvLayout,
) -> Result<Box<dyn Session>> {
    let inflight = Inflight::new(&model, &cluster, prefill_ubatch, kv_layout)?;
    Ok(Box::new(Qwen3MoeOwnedSession {
        model,
        inflight,
        cluster,
    }))
}

impl Model for PpHipModel {
    fn topology(&self) -> &'static str {
        "pp"
    }
    fn as_pp(&self) -> Option<&PpHipModel> {
        Some(self)
    }
    fn supports_scheduler_batching(&self) -> bool {
        true
    }
    fn forward_decode_batched(
        &self,
        state: &crate::routes::ServerState,
        inflights: &mut [&mut dyn Session],
        slots: &[flambeau_qwen3_moe::forward::BatchSlot],
        logits_refs: &mut [&mut Vec<f32>],
    ) -> Result<()> {
        crate::model::qwen3moe_forward_decode_batched(state, inflights, slots, logits_refs)
    }
}

impl Model for TpHipModel {
    fn topology(&self) -> &'static str {
        "tp"
    }
    fn as_tp(&self) -> Option<&TpHipModel> {
        Some(self)
    }
    fn supports_scheduler_batching(&self) -> bool {
        true
    }
    fn requires_prefill_serialiser(&self) -> bool {
        true
    }
    fn requires_tp_prefill_scratch(&self) -> bool {
        true
    }
    fn forward_decode_batched(
        &self,
        state: &crate::routes::ServerState,
        inflights: &mut [&mut dyn Session],
        slots: &[flambeau_qwen3_moe::forward::BatchSlot],
        logits_refs: &mut [&mut Vec<f32>],
    ) -> Result<()> {
        crate::model::qwen3moe_forward_decode_batched(state, inflights, slots, logits_refs)
    }
}

impl Model for HybridHipModel {
    fn topology(&self) -> &'static str {
        "pp+tp"
    }
    fn as_hybrid(&self) -> Option<&HybridHipModel> {
        Some(self)
    }
    fn supports_scheduler_batching(&self) -> bool {
        true
    }
    fn requires_prefill_serialiser(&self) -> bool {
        true
    }
    fn forward_decode_batched(
        &self,
        state: &crate::routes::ServerState,
        inflights: &mut [&mut dyn Session],
        slots: &[flambeau_qwen3_moe::forward::BatchSlot],
        logits_refs: &mut [&mut Vec<f32>],
    ) -> Result<()> {
        crate::model::qwen3moe_forward_decode_batched(state, inflights, slots, logits_refs)
    }
}

impl Session for Qwen3MoeOwnedSession {
    fn prefill_logits(
        &mut self,
        prompt_ids: &[u32],
        start_position: usize,
        logits_out: &mut Vec<f32>,
        tp_pool_prefill: Option<&mut ShardedForwardPrefillScratchTp>,
        on_boundary: Option<BoundaryCallback<'_>>,
        prefill_ubatch: usize,
    ) -> Result<()> {
        // Clone the model + cluster Arcs out before reborrowing `self`
        // as `&mut dyn Session`, so the free function gets disjoint
        // model + cluster + inflight refs.
        let model = self.model.clone();
        let cluster = self.cluster.clone();
        crate::model::prefill_logits(
            &model,
            &cluster,
            self,
            prompt_ids,
            start_position,
            logits_out,
            tp_pool_prefill,
            on_boundary,
            prefill_ubatch,
        )
    }

    fn reset_for_next_request(&mut self) -> Result<()> {
        let cluster = self.cluster.clone();
        self.inflight.reset_for_next_request(&cluster, &self.model)
    }

    fn dispose(self: Box<Self>) -> Result<()> {
        let Qwen3MoeOwnedSession {
            model,
            inflight,
            cluster,
        } = *self;
        inflight.dispose(&cluster, &model)
    }

    fn as_pp(&self) -> Option<&PpHipSession> {
        self.inflight.as_pp()
    }
    fn as_pp_mut(&mut self) -> Option<&mut PpHipSession> {
        self.inflight.as_pp_mut()
    }
    fn as_tp(&self) -> Option<&TpHipSession> {
        self.inflight.as_tp()
    }
    fn as_tp_mut(&mut self) -> Option<&mut TpHipSession> {
        self.inflight.as_tp_mut()
    }
    fn as_hybrid(&self) -> Option<&HybridHipSession> {
        self.inflight.as_hybrid()
    }
    fn as_hybrid_mut(&mut self) -> Option<&mut HybridHipSession> {
        self.inflight.as_hybrid_mut()
    }
}
