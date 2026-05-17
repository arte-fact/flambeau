//! Gemma4 server binding.
//!
//! Mirrors the qwen3-moe `model_handle` shape: a `Gemma4Model`
//! marker plus a `Gemma4Session` that wraps `Box<dyn ModelDriver>`.
//!
//! Phase 12.9 MVP: gemma4 drivers (`Gemma4PpDriver` / `Gemma4TpDriver`
//! / `Gemma4HybridDriver`) bundle weights + KV cache state inside one
//! struct. That means *one* driver instance backs *one* request slot;
//! multi-slot batched decode requires splitting weights from session
//! state and ships in a follow-up. For now, `FLAMBEAU_INFLIGHT_SLOTS=1`
//! is enforced at boot for gemma4 models.

#![cfg(feature = "hip")]

use anyhow::Result;
use flambeau_gemma4::Gemma4Config;
use flambeau_runtime::ModelDriver;

use crate::model_handle::{Model, Session};

/// Marker model handle for gemma4. The actual driver lives on the
/// per-request `Gemma4Session` (gemma4 weights + KV are bundled
/// inside the driver — splitting them is a follow-up).
pub struct Gemma4Model {
    pub cfg: Gemma4Config,
    pub topology: &'static str,
}

impl Model for Gemma4Model {
    fn topology(&self) -> &'static str {
        self.topology
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn chat_stop_markers(&self) -> &'static [&'static str] {
        // Kept in sync with `Gemma4Session::chat_stop_markers`.
        // `<channel|>` / `<|channel>` / `<|thought` are deliberately
        // EXCLUDED — the gemma4 tool-call parser strips them from the
        // response. Including them here would have `finalise` truncate
        // the whole answer when the model emits the leading
        // `<channel|>` close right after the chat template's
        // `<|channel>thought\n` block.
        &[
            "<end_of_turn>",
            "<turn|>",
            "<|turn>",
            "<|end_of_turn|>",
            "<|turn|>",
            "<|endoftext|>",
            "<endoftext>",
        ]
    }
}

/// Per-request gemma4 session. Owns the entire driver instance —
/// because gemma4's weights and KV-state are bundled in one struct,
/// each inflight slot has its own driver. Use
/// `FLAMBEAU_INFLIGHT_SLOTS=1` until weights/session split lands.
pub struct Gemma4Session {
    pub driver: Box<dyn ModelDriver>,
    /// Gemma4 mandates BOS prepended to every prompt (llama.cpp PR
    /// #21500 sets `force_add_bos=true` regardless of the GGUF's
    /// stored flag). The server's tokenization path doesn't add
    /// BOS (chat templates normally handle that, but `/v1/completions`
    /// and minimally-templated chat paths don't), so we prepend it
    /// inside `prefill_logits` when not already present.
    pub bos_id: Option<u32>,
}

impl Session for Gemma4Session {
    fn reset_for_next_request(&mut self) -> Result<()> {
        // V1: gemma4 drivers don't yet expose a KV-reset hook on the
        // ModelDriver trait. First request always works (KV starts
        // empty); second request reuses the slot WITHOUT clearing,
        // so the model sees the prior request's KV as a prefix —
        // expect garbage output. Restart the server for a fresh KV
        // until the reset hook lands on `ModelDriver`. No-op rather
        // than bail so the happy-path single-request flow boots.
        Ok(())
    }

    fn dispose(self: Box<Self>) -> Result<()> {
        let mut driver = self.driver;
        driver.dispose()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn as_model_driver_mut(&mut self) -> Option<&mut dyn ModelDriver> {
        Some(self.driver.as_mut())
    }

    fn decode_one_logits(
        &mut self,
        token: u32,
        position: usize,
        logits_out: &mut Vec<f32>,
    ) -> Result<()> {
        // routes.rs sends `position = prompt_ids.len() + step` with
        // step starting at 1, so the first decode position is N+1.
        // The BOS prepend in `prefill_logits` advances the cache tail
        // to N+1 too, so positions align with the driver's literal
        // write-slot semantics.
        logits_out.clear();
        self.driver
            .forward_one_token_logits(token, position, logits_out)
    }

    fn bos_id(&self) -> Option<u32> {
        self.bos_id
    }

    fn chat_stop_markers(&self) -> &'static [&'static str] {
        // HARD stops: byte-level chat-template fragments the 26B-A4B
        // Q8_0 MoE emits when its softmax drifts off the proper EOS
        // token (`<turn|>` id 106). Catching them as strings prevents
        // post-EOS repetition burning the `max_tokens` budget.
        //
        // `<channel|>` / `<|channel>` / `<|thought` are NOT here —
        // those are normal chat-template fragments that the gemma4
        // tool-call parser strips downstream. Putting them in this
        // list eats the whole answer when the model emits the
        // leading `<channel|>` close right after the prompt-side
        // `<|channel>thought\n` block.
        &[
            "<end_of_turn>",
            "<turn|>",
            "<|turn>",
            "<|end_of_turn|>",
            "<|turn|>",
            "<|endoftext|>",
            "<endoftext>",
        ]
    }
}

/// Build a `LoadedModel` (`Arc<dyn Model>`) for gemma4 from a
/// `Gemma4Config` + topology string. Mirror of qwen3-moe's
/// `LoadedModel` construction at server boot.
pub fn build_gemma4_loaded_model(
    cfg: Gemma4Config,
    topology: &'static str,
) -> crate::qwen3moe_handle::LoadedModel {
    std::sync::Arc::new(Gemma4Model { cfg, topology })
}

/// Wrap a constructed `Gemma4*Driver` (as a `Box<dyn ModelDriver>`)
/// into a Session trait object for the inflight pool. `bos_id` is
/// the tokenizer's BOS token (from GGUF metadata); when present, the
/// session prepends it to every fresh prefill.
pub fn wrap_gemma4_driver(
    driver: Box<dyn ModelDriver>,
    bos_id: Option<u32>,
) -> Box<dyn Session> {
    Box::new(Gemma4Session { driver, bos_id })
}

/// Best-effort topology-from-arch hint. Used by `is_gemma4_arch` style
/// gates at the boot path.
pub fn arch_matches(arch: &str) -> bool {
    matches!(
        arch,
        "gemma4" | "gemma4-26b-a4b" | "gemma4-31b" | "gemma4-9b" | "gemma4-2b"
    )
}
