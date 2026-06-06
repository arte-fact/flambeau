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
    fn forward_decode_batched(&mut self, slots: &[(u32, usize, usize)]) -> Result<()>;

    /// Sarathi-Serve mixed-batch forward (Phase K4c). Combines one
    /// prefill chunk (`K` tokens starting at `prefill_start_position`
    /// on `prefill_slot_id`) with N decode tokens (`decodes`) into a
    /// single Session::forward_mixed call. Returns Ok(()) iff the arch
    /// supports mixed (qwen35-v2 / qwen35moe-v2 today); other archs
    /// bail. After return, the (N + 1) logit rows are reachable via
    /// [`Self::logits_row`] in row-major `[(N + 1), vocab]` order —
    /// row 0 = prefill slot's next-token logit; rows 1..=N = decode
    /// slot logits.
    fn forward_mixed(
        &mut self,
        prefill_tokens: &[u32],
        prefill_start_position: usize,
        prefill_slot_id: usize,
        decodes: &[(u32, usize, usize)],
    ) -> Result<()>;

    fn logits_row(&self, i: usize, vocab: usize) -> &[f32];

    fn reset_kv_slot(&mut self, slot_id: usize) -> Result<()>;

    fn release_paged_slot(&mut self, slot_id: usize) -> Result<()>;

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

    fn forward_decode_batched(&mut self, slots: &[(u32, usize, usize)]) -> Result<()> {
        Session::forward_decode_batched(self, slots)
    }

    fn forward_mixed(
        &mut self,
        prefill_tokens: &[u32],
        prefill_start_position: usize,
        prefill_slot_id: usize,
        decodes: &[(u32, usize, usize)],
    ) -> Result<()> {
        let k = prefill_tokens.len();
        if k == 0 {
            anyhow::bail!("V2BatchableSession::forward_mixed: empty prefill_tokens");
        }
        if decodes.is_empty() {
            anyhow::bail!("V2BatchableSession::forward_mixed: empty decodes");
        }
        let n = k + decodes.len();
        let mut tokens: Vec<u32> = Vec::with_capacity(n);
        let mut positions: Vec<usize> = Vec::with_capacity(n);
        let mut slot_ids: Vec<usize> = Vec::with_capacity(n);
        tokens.extend_from_slice(prefill_tokens);
        for i in 0..k {
            positions.push(prefill_start_position + i);
            slot_ids.push(prefill_slot_id);
        }
        for &(tok, pos, slot) in decodes {
            tokens.push(tok);
            positions.push(pos);
            slot_ids.push(slot);
        }
        Session::forward_mixed(self, &tokens, &positions, &slot_ids, k)
    }

    fn logits_row(&self, i: usize, vocab: usize) -> &[f32] {
        Session::logits_row(self, i, vocab)
    }

    fn reset_kv_slot(&mut self, slot_id: usize) -> Result<()> {
        Session::reset_kv_slot(self, slot_id)
    }

    fn release_paged_slot(&mut self, slot_id: usize) -> Result<()> {
        Session::release_paged_slot(self, slot_id)
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
        inflights: &mut [&mut dyn ServerSession],
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
        if inflights.len() != slots.len() {
            anyhow::bail!(
                "V2Model::forward_decode_batched: inflights={} slots={}",
                inflights.len(),
                slots.len(),
            );
        }
        if slots.is_empty() {
            return Ok(());
        }
        let n = slots.len();
        // `BatchSlot.idx` is the parallel-array position (= `i` in this
        // loop) — used only for routing logits_refs[i] back to the
        // right pending sender. The actual KV/GDN slot for the forward
        // is the per-V2Conv `slot_id` set at inflight construction;
        // read it from `inflights[i]`. Using `s.idx` directly was a
        // long-standing bug: under N>1 concurrency the pending queue
        // can arrive in any order, so `s.idx` (queue position) and the
        // per-conv slot_id diverge, routing decode reads/writes to the
        // wrong slot's KV slab and producing inter-stream topic mixing.
        let tuples: Vec<(u32, usize, usize)> = slots
            .iter()
            .zip(inflights.iter())
            .map(|(s, inflight)| {
                let conv = inflight
                    .as_any()
                    .downcast_ref::<V2Conv>()
                    .expect("V2Model::forward_decode_batched expects V2Conv inflight");
                (s.token_id, s.position, conv.slot_id)
            })
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
        for (i, out) in logits_refs.iter_mut().enumerate().take(n) {
            let row = shared.logits_row(i, vocab);
            out.clear();
            out.extend_from_slice(row);
        }
        Ok(())
    }
    fn supports_mixed_batch(&self) -> bool {
        matches!(
            self.gguf_arch,
            "qwen35"
                | "qwen35moe"
                | "gemma3"
                | "gemma4"
                | "gemma4-26b-a4b"
                | "gemma4-31b"
                | "gemma4-9b"
                | "gemma4-2b"
        )
    }
    fn forward_mixed_decode(
        &self,
        _ctx: &dyn crate::model_handle::SessionContext,
        prefill: flambeau_server_core::MixedBatchPrefill<'_>,
        decodes: flambeau_server_core::MixedBatchDecodes<'_, '_>,
    ) -> Result<()> {
        let flambeau_server_core::MixedBatchPrefill {
            inflight: prefill_inflight,
            tokens: prefill_tokens,
            start_position: prefill_start_position,
            logits_out: prefill_logits_out,
        } = prefill;
        let flambeau_server_core::MixedBatchDecodes {
            inflights: decode_inflights,
            slots: decode_slots,
            logits_refs: decode_logits_refs,
        } = decodes;
        if decode_slots.len() != decode_logits_refs.len()
            || decode_slots.len() != decode_inflights.len()
        {
            anyhow::bail!(
                "V2Model::forward_mixed_decode: decode_slots={} decode_inflights={} \
                 decode_logits_refs={}",
                decode_slots.len(),
                decode_inflights.len(),
                decode_logits_refs.len(),
            );
        }
        if prefill_tokens.is_empty() {
            anyhow::bail!("V2Model::forward_mixed_decode: empty prefill_tokens");
        }
        if decode_slots.is_empty() {
            anyhow::bail!("V2Model::forward_mixed_decode: empty decode_slots");
        }
        let prefill_slot_id = prefill_inflight
            .as_any()
            .downcast_ref::<V2Conv>()
            .expect("V2Model::forward_mixed_decode expects V2Conv prefill inflight")
            .slot_id;
        let decode_tuples: Vec<(u32, usize, usize)> = decode_slots
            .iter()
            .zip(decode_inflights.iter())
            .map(|(s, inflight)| {
                let conv = inflight
                    .as_any()
                    .downcast_ref::<V2Conv>()
                    .expect("V2Model::forward_mixed_decode expects V2Conv decode inflight");
                (s.token_id, s.position, conv.slot_id)
            })
            .collect();
        let vocab = self.vocab;
        let mut shared = self.shared.blocking_lock();
        shared.forward_mixed(
            prefill_tokens,
            prefill_start_position,
            prefill_slot_id,
            &decode_tuples,
        )?;
        // Row 0 = prefill slot's next-token logit (the (K-1)-th token).
        let row = shared.logits_row(0, vocab);
        prefill_logits_out.clear();
        prefill_logits_out.extend_from_slice(row);
        // Rows 1..=N = decode logits, one per decode slot in caller's order.
        for (i, out) in decode_logits_refs.iter_mut().enumerate().take(decode_slots.len()) {
            let row = shared.logits_row(i + 1, vocab);
            out.clear();
            out.extend_from_slice(row);
        }
        Ok(())
    }
    fn chat_stop_markers(&self) -> &'static [&'static str] {
        self.chat_stops
    }
    fn release_paged_slot(&self, slot: usize) {
        let mut shared = self.shared.blocking_lock();
        if let Err(e) = shared.release_paged_slot(slot) {
            tracing::warn!("V2Model::release_paged_slot(slot={slot}) failed: {e:#}");
        }
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
    is_gemma_family(gguf_arch)
}

fn is_gemma_family(gguf_arch: &str) -> bool {
    matches!(
        gguf_arch,
        "gemma3" | "gemma4" | "gemma4-26b-a4b" | "gemma4-31b" | "gemma4-9b" | "gemma4-2b"
    )
}

/// Minimum decode steps before a stop token is honored, guarding against
/// immediate-EOS empty replies (Qwen3.6 multi-turn). gemma4 emits its
/// turn-end as soon as a short answer is complete and has no immediate-EOS
/// failure mode, so forcing continuation only drives it into repetition /
/// off-topic drift — trust its stop (the first-token stop mask still applies).
pub fn min_response_tokens_for(gguf_arch: &str) -> usize {
    if is_gemma_family(gguf_arch) {
        2
    } else {
        24
    }
}

/// Nats subtracted from every stop-token logit beyond the min-response
/// window to discourage premature stops. Zero for gemma4 (see
/// [`min_response_tokens_for`]).
pub fn stop_bias_for(gguf_arch: &str) -> f32 {
    if is_gemma_family(gguf_arch) {
        0.0
    } else {
        3.0
    }
}
