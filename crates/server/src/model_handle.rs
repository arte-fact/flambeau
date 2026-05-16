//! Server-side `Model` + `Session` trait surface.
//!
//! Designed to be backend-neutral: the trait defs live here without
//! referencing qwen3-moe-typed concrete types, so the file can be lifted
//! into a `flambeau-server-core` crate that future arch crates (gemma4,
//! cuda) can implement against without depending on flambeau-server or
//! flambeau-qwen3-moe.
//!
//! Concrete-type access for qwen3-moe call sites lives behind extension
//! traits (`Qwen3MoeModelExt`, `Qwen3MoeSessionExt`) in `crate::model`
//! — those wrap `as_any().downcast_ref::<…>()` so the trait file stays
//! arch-clean.

#![cfg(feature = "hip")]

use std::any::Any;

use anyhow::Result;
use flambeau_backend_hip::HipCluster;
use flambeau_qwen3_moe::session::KvLayout;

use crate::model::{
    BoundaryCallback, HybridHipModel, Inflight, LoadedModel, PpHipModel, TpHipModel,
};

/// One queued slot in a batched decode dispatch. Arch-neutral mirror
/// of qwen3-moe's `flambeau_qwen3_moe::forward::BatchSlot`; qwen3-moe
/// dispatchers convert at the boundary so the generic trait surface
/// stays arch-clean.
#[derive(Debug, Clone, Copy)]
pub struct BatchSlot {
    /// Index into the caller's `inflights` / `logits_refs` parallel array.
    pub idx: usize,
    /// Token to decode this step.
    pub token_id: u32,
    /// Cache position to decode at.
    pub position: usize,
}

pub trait Model: Send + Sync + 'static {
    /// Topology label for handler metrics: `"pp"`, `"tp"`, `"pp+tp"`,
    /// `"gemma4_pp"`, …
    fn topology(&self) -> &'static str;

    /// Downcast hatch for arch-specific code paths. Implementors return
    /// `self`. Use the per-arch extension traits
    /// (e.g. `Qwen3MoeModelExt::as_pp`) for typed access.
    fn as_any(&self) -> &dyn Any;

    /// True when the model arch supports the batched-decode scheduler.
    /// Qwen3-moe (PP/TP/Hybrid) overrides to true; gemma4 + future N=1
    /// archs leave it false so the legacy single-stream decode handler
    /// is selected. JSON / logprobs paths bypass the scheduler
    /// regardless via the orthogonal gates in `scheduler_can_engage`.
    fn supports_scheduler_batching(&self) -> bool {
        false
    }

    /// True when the model arch requires the TP/Hybrid prefill
    /// serialiser lock held across a `prefill_logits` call. Qwen3-moe
    /// TP + Hybrid override to true.
    fn requires_prefill_serialiser(&self) -> bool {
        false
    }

    /// True when the model arch consumes a pre-allocated TP prefill
    /// scratch handed in via `Session::prefill_logits`'s `tp_pool_prefill`
    /// parameter. Only qwen3-moe TP returns true.
    fn requires_tp_prefill_scratch(&self) -> bool {
        false
    }

    /// Arch-specific byte-level chat-template fragments that should
    /// stop generation when present in the decoded text. Default `&[]`.
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
        slots: &[BatchSlot],
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

pub trait Session: Send + 'static {
    /// Downcast hatch for arch-specific code paths. Implementors return
    /// `self`. Use the per-arch extension traits (e.g.
    /// `Qwen3MoeSessionExt::as_pp_mut`) for typed access.
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// Ingest a (chunked-as-needed) prompt and write the last
    /// position's logits into `logits_out`. `start_position` is the
    /// absolute position of `prompt_ids[0]` inside the original full
    /// prompt — `0` for a fresh request, `> 0` after a prefix-cache
    /// restore covering `[0..start_position)`. `tp_pool_prefill` (TP
    /// only) lets the caller share a pre-allocated scratch. `on_boundary`
    /// (PP / TP only) fires after every non-final chunk for prefix-cache
    /// snapshotting.
    #[allow(clippy::too_many_arguments)]
    fn prefill_logits(
        &mut self,
        prompt_ids: &[u32],
        start_position: usize,
        logits_out: &mut Vec<f32>,
        tp_pool_prefill: Option<&mut dyn Any>,
        on_boundary: Option<BoundaryCallback<'_>>,
        prefill_ubatch: usize,
    ) -> Result<()>;

    fn reset_for_next_request(&mut self) -> Result<()>;

    fn dispose(self: Box<Self>) -> Result<()>;

    /// Single-token decode. Default bails — only archs that fall
    /// through `Model::forward_decode_batched`'s N=1 default need a real
    /// impl (gemma4 today). qwen3-moe sessions route through
    /// `qwen3moe_forward_decode_batched` and never reach here.
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

    /// Arch-cross `ModelDriver` accessor. Gemma4 returns its bundled
    /// driver; archs that don't bundle a `ModelDriver` (qwen3-moe today)
    /// return `None`. Used for the N=1 decode path that calls
    /// `forward_one_token_logits` directly.
    fn as_model_driver_mut(&mut self) -> Option<&mut dyn flambeau_runtime::ModelDriver> {
        None
    }

    /// BOS token to prepend when starting a fresh prefill, for arches
    /// that mandate it (Gemma4). `None` for arches where the chat
    /// template / tokenizer handles BOS itself (qwen3-moe).
    fn bos_id(&self) -> Option<u32> {
        None
    }

    /// Arch-specific byte-level chat-template fragments that should
    /// stop generation when they appear in the decoded text. Default
    /// `&[]`; gemma4 returns its template's turn / channel / EOS
    /// fragments.
    fn chat_stop_markers(&self) -> &'static [&'static str] {
        &[]
    }
}

/// Self-sufficient qwen3-moe session: bundles an `Inflight`
/// (KV state + scratches) with a back-reference to its parent
/// `LoadedModel` and an `Arc<HipCluster>` clone so the trait methods
/// don't need a cluster passed in.
pub struct Qwen3MoeOwnedSession {
    pub model: LoadedModel,
    pub inflight: Inflight,
    pub cluster: std::sync::Arc<HipCluster>,
}

/// Build a per-request session bound to `model`.
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
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn supports_scheduler_batching(&self) -> bool {
        true
    }
    fn forward_decode_batched(
        &self,
        state: &crate::routes::ServerState,
        inflights: &mut [&mut dyn Session],
        slots: &[BatchSlot],
        logits_refs: &mut [&mut Vec<f32>],
    ) -> Result<()> {
        crate::model::qwen3moe_forward_decode_batched(state, inflights, slots, logits_refs)
    }
}

impl Model for TpHipModel {
    fn topology(&self) -> &'static str {
        "tp"
    }
    fn as_any(&self) -> &dyn Any {
        self
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
        slots: &[BatchSlot],
        logits_refs: &mut [&mut Vec<f32>],
    ) -> Result<()> {
        crate::model::qwen3moe_forward_decode_batched(state, inflights, slots, logits_refs)
    }
}

impl Model for HybridHipModel {
    fn topology(&self) -> &'static str {
        "pp+tp"
    }
    fn as_any(&self) -> &dyn Any {
        self
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
        slots: &[BatchSlot],
        logits_refs: &mut [&mut Vec<f32>],
    ) -> Result<()> {
        crate::model::qwen3moe_forward_decode_batched(state, inflights, slots, logits_refs)
    }
}

impl Session for Qwen3MoeOwnedSession {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn prefill_logits(
        &mut self,
        prompt_ids: &[u32],
        start_position: usize,
        logits_out: &mut Vec<f32>,
        tp_pool_prefill: Option<&mut dyn Any>,
        on_boundary: Option<BoundaryCallback<'_>>,
        prefill_ubatch: usize,
    ) -> Result<()> {
        let model = self.model.clone();
        let cluster = self.cluster.clone();
        let typed = tp_pool_prefill.and_then(|p| {
            p.downcast_mut::<flambeau_qwen3_moe::forward::ShardedForwardPrefillScratchTp>()
        });
        crate::model::prefill_logits(
            &model,
            &cluster,
            self,
            prompt_ids,
            start_position,
            logits_out,
            typed,
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
}
