//! Core trait surface: `Model`, `Session`, `SessionContext`, `BatchSlot`.
//! `LogitsSink` lives in `flambeau-model-ops` so model crates can use it
//! without taking a server-core dep; re-exported here for convenience.

use std::any::Any;

use anyhow::Result;
use flambeau_backend_hip::HipCluster;

/// One queued slot in a batched decode dispatch. Arch-neutral mirror of
/// `flambeau_qwen3_moe::forward::BatchSlot`; qwen3-moe dispatchers
/// convert at the boundary so the trait surface stays arch-clean.
#[derive(Debug, Clone, Copy)]
pub struct BatchSlot {
    /// Index into the caller's `inflights` / `logits_refs` parallel array.
    pub idx: usize,
    /// Token to decode this step.
    pub token_id: u32,
    /// Cache position to decode at.
    pub position: usize,
}

/// Backend-neutral context handed to per-arch `Model::forward_decode_batched`
/// impls. Exposes the shared infrastructure those impls actually consume
/// from `ServerState` (cluster, inflight-pool size, arch-specific extras
/// via `Any` downcast). Lets the trait file live without a back-reference
/// to flambeau-server's concrete `ServerState`.
pub trait SessionContext {
    fn cluster(&self) -> &HipCluster;
    fn max_inflight_slots(&self) -> usize;
    /// Arch-specific shared extras (e.g. qwen3-moe's `Qwen3MoeServerExtras`).
    /// Returns `None` for arches that don't carry shared state beyond the
    /// cluster.
    fn extras(&self) -> Option<&dyn Any>;
}

pub trait Model: Send + Sync + 'static {
    /// Topology label for handler metrics: `"pp"`, `"tp"`, `"pp+tp"`,
    /// `"gemma4_pp"`, …
    fn topology(&self) -> &'static str;

    /// Downcast hatch for arch-specific code paths. Implementors return
    /// `self`. Use per-arch extension traits (e.g. `Qwen3MoeModelExt::as_pp`
    /// in flambeau-server's `model.rs`) for typed access.
    fn as_any(&self) -> &dyn Any;

    /// True when the model arch supports the batched-decode scheduler.
    /// Qwen3-moe (PP/TP/Hybrid) overrides to true; gemma4 + future N=1
    /// archs leave it false so the legacy single-stream decode handler
    /// is selected.
    fn supports_scheduler_batching(&self) -> bool {
        false
    }

    /// True when the model arch requires the TP/Hybrid prefill
    /// serialiser lock held across a prefill call. Qwen3-moe TP + Hybrid
    /// override to true.
    fn requires_prefill_serialiser(&self) -> bool {
        false
    }

    /// True when the model arch consumes a pre-allocated TP prefill
    /// scratch. Only qwen3-moe TP returns true.
    fn requires_tp_prefill_scratch(&self) -> bool {
        false
    }

    /// Arch-specific byte-level chat-template fragments that should stop
    /// generation when present in the decoded text. Default `&[]`.
    fn chat_stop_markers(&self) -> &'static [&'static str] {
        &[]
    }

    /// Batched-decode entry point. Default handles N=1 only by
    /// delegating to [`Session::decode_one_logits`] — multi-slot archs
    /// (qwen3-moe PP/TP/Hybrid) override this method.
    fn forward_decode_batched(
        &self,
        _ctx: &dyn SessionContext,
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
    /// `self`. Use per-arch extension traits (e.g. `Qwen3MoeSessionExt::as_pp_mut`)
    /// for typed access.
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;

    fn reset_for_next_request(&mut self) -> Result<()>;

    fn dispose(self: Box<Self>) -> Result<()>;

    /// Single-token decode. Default bails — only archs that fall
    /// through `Model::forward_decode_batched`'s N=1 default need a real
    /// impl (gemma4 today). Qwen3-moe sessions route through
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
    /// `&[]`.
    fn chat_stop_markers(&self) -> &'static [&'static str] {
        &[]
    }
}
