//! Server binding for the v2 forward stack. `V2Model` + `V2Session`
//! wrap any `Box<dyn ModelDriver>` produced from a `Session<A>` — the
//! arch dispatch lives in `serve::create_v2_driver`. Mirrors the
//! `gemma4_handle` shape; once the legacy gemma4 path retires, this
//! is the only adapter.

#![cfg(feature = "hip")]

use anyhow::Result;
use flambeau_runtime::ModelDriver;

use crate::model_handle::{Model, Session};

/// Topology-tag-bearing model marker for every v2 arch. The actual
/// driver lives on each `V2Session` (Session<A> bundles weights + KV
/// per-worker, so each inflight slot has its own Session<A>).
pub struct V2Model {
    pub gguf_arch: &'static str,
    pub topology: &'static str,
    pub chat_stops: &'static [&'static str],
}

impl Model for V2Model {
    fn topology(&self) -> &'static str {
        self.topology
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    /// v2 has no batched-decode kernel yet. Fall through to N=1 via
    /// `Session::decode_one_logits` for each pending slot.
    /// Tracked as [P9-OUT batched-decode].
    fn supports_scheduler_batching(&self) -> bool {
        false
    }
    fn forward_decode_batched(
        &self,
        _ctx: &dyn crate::model_handle::SessionContext,
        inflights: &mut [&mut dyn crate::model_handle::Session],
        slots: &[crate::model_handle::BatchSlot],
        logits_refs: &mut [&mut Vec<f32>],
    ) -> Result<()> {
        if slots.len() != inflights.len() || slots.len() != logits_refs.len() {
            anyhow::bail!(
                "V2Model::forward_decode_batched: mismatched slice lengths \
                 (slots={}, inflights={}, logits_refs={})",
                slots.len(),
                inflights.len(),
                logits_refs.len(),
            );
        }
        for (i, slot) in slots.iter().enumerate() {
            let out: &mut Vec<f32> = logits_refs[i];
            out.clear();
            inflights[i].decode_one_logits(slot.token_id, slot.position, out)?;
        }
        Ok(())
    }
    fn chat_stop_markers(&self) -> &'static [&'static str] {
        self.chat_stops
    }
}

pub struct V2Session {
    pub driver: Box<dyn ModelDriver>,
    pub bos_id: Option<u32>,
    pub chat_stops: &'static [&'static str],
}

impl Session for V2Session {
    fn reset_for_next_request(&mut self) -> Result<()> {
        self.driver.reset_kv()
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
        logits_out.clear();
        self.driver
            .forward_one_token_logits(token, position, logits_out)
    }

    fn bos_id(&self) -> Option<u32> {
        self.bos_id
    }

    fn chat_stop_markers(&self) -> &'static [&'static str] {
        self.chat_stops
    }
}

/// Chat-template stop markers per supported v2 arch. Empty slice for
/// archs whose tokenizers' EOS handling is sufficient.
pub fn chat_stops_for(gguf_arch: &str) -> &'static [&'static str] {
    match gguf_arch {
        "gemma3" | "gemma4" | "gemma4-26b-a4b" | "gemma4-31b" | "gemma4-9b" | "gemma4-2b" => {
            // Mirror of `gemma4_handle::Gemma4Session::chat_stop_markers`.
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
        _ => &[],
    }
}

/// Whether this arch wants BOS prepended to every fresh prompt.
/// Gemma4 mandates it (llama.cpp #21500 forces add_bos regardless of
/// GGUF). Qwen3 / qwen35moe set `tokenizer.ggml.add_bos_token = false`
/// — the chat template doesn't include BOS and prepending one shifts
/// position embeddings and confuses some checkpoints (Qwen3.6-27B
/// emits EOS immediately).
pub fn wants_bos_prepend(gguf_arch: &str) -> bool {
    matches!(
        gguf_arch,
        "gemma3" | "gemma4" | "gemma4-26b-a4b" | "gemma4-31b" | "gemma4-9b" | "gemma4-2b"
    )
}
