//! v2 server binding. One shared `Session<A>` carries N inflight
//! conversation slots; each `V2Conv` is a per-conversation handle
//! (slot_id + Arc to the shared session) that the server's inflight
//! pool stores as `Box<dyn crate::Session>`.
//!
//! `V2BatchableSession` is an `A`-erasing trait so the type-erased
//! pool can dispatch batched decodes without the server crate growing
//! a generic parameter. `Session<A>` for every arch satisfies it via
//! the blanket impl below.

#![cfg(feature = "hip")]

use std::sync::Arc;

use anyhow::Result;
use flambeau_forward::runtime::{Arch, Session};
use flambeau_runtime::ModelDriver;
use tokio::sync::Mutex;

use crate::model_handle::{Model, Session as ServerSession};

/// Arch-erased trait around `Session<A>` so the server's v2 pool can
/// dispatch slot-aware forward calls without knowing which `A` is
/// underneath. Every method routes to the matching `Session<A>` API.
pub trait V2BatchableSession: Send {
    fn forward_one_token_into_slot(
        &mut self,
        token: u32,
        position: usize,
        slot_id: usize,
        out: &mut Vec<f32>,
    ) -> Result<()>;

    fn forward_prefill_into_slot(
        &mut self,
        tokens: &[u32],
        start_position: usize,
        slot_id: usize,
        out: &mut Vec<f32>,
    ) -> Result<()>;

    /// Drive N concurrent decodes through one Session forward. After
    /// return, logits rows are reachable via [`Self::logits_row`] with
    /// the model's vocab. Slot ids must be distinct.
    fn forward_decode_batched(
        &mut self,
        slots: &[(u32, usize, usize)],
    ) -> Result<()>;

    fn logits_row(&self, i: usize, vocab: usize) -> &[f32];

    fn reset_kv_slot(&mut self, slot_id: usize) -> Result<()>;

    fn dispose_in_place(&mut self) -> Result<()>;
}

impl<A: Arch> V2BatchableSession for Session<A> {
    fn forward_one_token_into_slot(
        &mut self,
        token: u32,
        position: usize,
        slot_id: usize,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        Session::forward_one_token_logits_slot(self, token, position, slot_id, out)
    }

    fn forward_prefill_into_slot(
        &mut self,
        tokens: &[u32],
        start_position: usize,
        slot_id: usize,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        Session::forward_prefill_logits_slot(self, tokens, start_position, slot_id, out)
    }

    fn forward_decode_batched(
        &mut self,
        slots: &[(u32, usize, usize)],
    ) -> Result<()> {
        Session::forward_decode_batched(self, slots)
    }

    fn logits_row(&self, i: usize, vocab: usize) -> &[f32] {
        Session::logits_row(self, i, vocab)
    }

    fn reset_kv_slot(&mut self, slot_id: usize) -> Result<()> {
        Session::reset_kv_slot(self, slot_id)
    }

    fn dispose_in_place(&mut self) -> Result<()> {
        Session::dispose_in_place(self)
    }
}

/// Handle to the single shared v2 session. Wrapped in `tokio::sync::Mutex`
/// so handlers can `blocking_lock` from sync code; the lock is held for
/// the duration of one forward call.
pub type SharedV2Session = Arc<Mutex<Box<dyn V2BatchableSession>>>;

/// Topology-tag-bearing model marker for every v2 arch. Holds the
/// shared session so `Model::forward_decode_batched` can dispatch N
/// concurrent decodes through one Session forward.
pub struct V2Model {
    pub gguf_arch: &'static str,
    pub topology: &'static str,
    pub chat_stops: &'static [&'static str],
    pub shared: SharedV2Session,
    /// Model vocab — set at boot from GGUF metadata so the
    /// `forward_decode_batched` impl can slice the `[N, vocab]` logits
    /// buffer without re-reading.
    pub vocab: usize,
}

impl Model for V2Model {
    fn topology(&self) -> &'static str {
        self.topology
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn supports_scheduler_batching(&self) -> bool {
        true
    }
    fn forward_decode_batched(
        &self,
        _ctx: &dyn crate::model_handle::SessionContext,
        _inflights: &mut [&mut dyn ServerSession],
        slots: &[crate::model_handle::BatchSlot],
        logits_refs: &mut [&mut Vec<f32>],
    ) -> Result<()> {
        if slots.len() != logits_refs.len() {
            anyhow::bail!(
                "V2Model::forward_decode_batched: slots={} logits_refs={}",
                slots.len(),
                logits_refs.len(),
            );
        }
        if slots.is_empty() {
            return Ok(());
        }
        let n = slots.len();
        let tuples: Vec<(u32, usize, usize)> = slots
            .iter()
            .map(|s| (s.token_id, s.position, s.idx))
            .collect();
        let single_slot = n == 1;
        let vocab = self.vocab;
        let mut shared = self.shared.blocking_lock();
        if single_slot {
            // N=1: Session::forward_decode_batched would refuse (its
            // sanity check forces distinct slot_ids for N>1); route
            // through forward_one_token_into_slot directly.
            let (token, position, slot_id) = tuples[0];
            let out: &mut Vec<f32> = logits_refs[0];
            return shared.forward_one_token_into_slot(token, position, slot_id, out);
        }
        shared.forward_decode_batched(&tuples)?;
        for i in 0..n {
            let row = shared.logits_row(i, vocab);
            let out: &mut Vec<f32> = logits_refs[i];
            out.clear();
            out.extend_from_slice(row);
        }
        Ok(())
    }
    fn chat_stop_markers(&self) -> &'static [&'static str] {
        self.chat_stops
    }
}

/// Per-conversation handle in the v2 inflight pool. Owns nothing
/// device-side; the shared session is held collectively by V2Model +
/// every V2Conv via Arc.
pub struct V2Conv {
    pub shared: SharedV2Session,
    pub slot_id: usize,
    pub bos_id: Option<u32>,
    pub chat_stops: &'static [&'static str],
}

impl ServerSession for V2Conv {
    fn reset_for_next_request(&mut self) -> Result<()> {
        let mut shared = self.shared.blocking_lock();
        shared.reset_kv_slot(self.slot_id)
    }

    fn dispose(self: Box<Self>) -> Result<()> {
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn as_model_driver_mut(&mut self) -> Option<&mut dyn ModelDriver> {
        Some(self as &mut dyn ModelDriver)
    }

    fn decode_one_logits(
        &mut self,
        token: u32,
        position: usize,
        logits_out: &mut Vec<f32>,
    ) -> Result<()> {
        logits_out.clear();
        let mut shared = self.shared.blocking_lock();
        shared.forward_one_token_into_slot(token, position, self.slot_id, logits_out)
    }

    fn bos_id(&self) -> Option<u32> {
        self.bos_id
    }

    fn chat_stop_markers(&self) -> &'static [&'static str] {
        self.chat_stops
    }
}

fn argmax(logits: &[f32]) -> u32 {
    let (mut best_i, mut best_v) = (0_u32, f32::NEG_INFINITY);
    for (i, &l) in logits.iter().enumerate() {
        if l > best_v {
            best_v = l;
            best_i = i as u32;
        }
    }
    best_i
}

impl ModelDriver for V2Conv {
    fn forward_prefill(&mut self, tokens: &[u32], start_position: usize) -> Result<u32> {
        let mut buf = Vec::new();
        self.forward_prefill_logits(tokens, start_position, &mut buf)?;
        Ok(argmax(&buf))
    }

    fn forward_one_token(&mut self, token_id: u32, position: usize) -> Result<u32> {
        let mut buf = Vec::new();
        self.forward_one_token_logits(token_id, position, &mut buf)?;
        Ok(argmax(&buf))
    }

    fn forward_prefill_logits(
        &mut self,
        tokens: &[u32],
        start_position: usize,
        logits_out: &mut Vec<f32>,
    ) -> Result<()> {
        let mut shared = self.shared.blocking_lock();
        shared.forward_prefill_into_slot(tokens, start_position, self.slot_id, logits_out)
    }

    fn forward_one_token_logits(
        &mut self,
        token_id: u32,
        position: usize,
        logits_out: &mut Vec<f32>,
    ) -> Result<()> {
        let mut shared = self.shared.blocking_lock();
        shared.forward_one_token_into_slot(token_id, position, self.slot_id, logits_out)
    }

    fn vocab_size(&self) -> usize {
        0
    }

    fn reset_kv(&mut self) -> Result<()> {
        let mut shared = self.shared.blocking_lock();
        shared.reset_kv_slot(self.slot_id)
    }

    fn dispose(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Chat-template stop markers per supported v2 arch. Empty slice for
/// archs whose tokenizers' EOS handling is sufficient.
pub fn chat_stops_for(gguf_arch: &str) -> &'static [&'static str] {
    match gguf_arch {
        "gemma3" | "gemma4" | "gemma4-26b-a4b" | "gemma4-31b" | "gemma4-9b" | "gemma4-2b" => &[
            "<end_of_turn>",
            "<turn|>",
            "<|turn>",
            "<|end_of_turn|>",
            "<|turn|>",
            "<|endoftext|>",
            "<endoftext>",
        ],
        _ => &[],
    }
}

/// Whether this arch wants BOS prepended to every fresh prompt.
pub fn wants_bos_prepend(gguf_arch: &str) -> bool {
    matches!(
        gguf_arch,
        "gemma3" | "gemma4" | "gemma4-26b-a4b" | "gemma4-31b" | "gemma4-9b" | "gemma4-2b"
    )
}
