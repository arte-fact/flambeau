//! **P2.9b-i2-A** — batched decode driver: API surface for running N
//! concurrent slots through one forward pass.
//!
//! ## Status
//!
//! This is the **v0 scaffold**. The public API
//! ([`BatchSlot`] + [`forward_decode_batched_pp`]) is the contract that
//! the server scheduler (P2.9b-i2-B) calls into. The current
//! implementation runs N sequential single-slot decodes via
//! [`super::pp::forward_one_token_pp_logits`] — i.e. **no batched-kernel
//! win yet**. Each slot still pays its own per-layer kernel-launch cost
//! and the FFN/MoE GEMM is not amortised across slots.
//!
//! Follow-ups that swap real batching behind this surface:
//! - **i2-A1**: batched Q/K/V projection ([N, hidden] in, [N, *] out).
//! - **i2-A2**: batched FFN/MoE (the biggest lever — reuses the
//!   prefill MoE kernels at `n_tokens=N`).
//! - **i2-A3**: per-slot attention + KV append wired into the batched
//!   layer body so attn stays correct against each slot's KV history.
//! - **i2-E**: optional batched-attention kernel (single launch handles
//!   N (Q, KV) pairs) if i2-F's 3× gate misses without it.
//!
//! ## Why this scaffold ships
//!
//! P2.9b-i1 already gave us request-level concurrency through the slot
//! pool (`Vec<Mutex<Inflight>>`). Two clients running `/v1/chat` in
//! parallel see ~1.3× throughput vs serial. To go further we need real
//! kernel batching, which is multi-session work. The scaffold pins the
//! interface so:
//!   - P2.9b-i2-B (scheduler) can be designed against a stable signature.
//!   - Each future kernel-batching subtask (i2-A1/A2/A3) is a localised
//!     swap inside this module rather than a server-wide refactor.

#![cfg(feature = "hip")]

use anyhow::{bail, Result};

/// One queued slot in a batched decode dispatch.
///
/// Each slot owns its own `(token_id, position)` pair plus an index into
/// the caller's session/scratch arrays — the server scheduler (i2-B)
/// stages slots whose i1-pool guards it currently holds, so the session
/// references live for the call.
#[derive(Debug, Clone, Copy)]
pub struct BatchSlot {
    /// Index into the caller's `sessions` / `scratches` parallel arrays.
    pub idx: usize,
    /// Token to decode this step (output of slot's previous sample).
    pub token_id: u32,
    /// Cache position to decode at (typically `current_tokens` of the
    /// slot's KV cache, before this token is appended).
    pub position: usize,
}

/// **P2.9b-i2-A scaffold** — drive `slots.len()` concurrent decode steps
/// through the PP topology, returning per-slot `[vocab]` F32 logits.
///
/// `sessions[s.idx]` and `scratches[s.idx]` must be the per-slot session
/// and scratch belonging to `BatchSlot` `s`. Caller is responsible for
/// holding the slot pool guards (P2.9b-i1) for the lifetime of this call.
///
/// `logits_out[s.idx]` receives the slot's logits row; the vec is resized
/// to `vocab_size` and existing contents are overwritten.
///
/// ## Current implementation (v0)
///
/// Loops over `slots` and calls
/// [`super::pp::forward_one_token_pp_logits`] sequentially per slot.
/// Identical wall to N independent decode calls; the win is purely API
/// shape — i2-B's scheduler can target this signature today, and i2-A1
/// / A2 / A3 will replace the loop body with batched-kernel calls without
/// changing this surface.
pub fn forward_decode_batched_pp(
    model: &crate::sharded::Qwen3MoEShardedModel,
    sessions: &mut [&mut crate::sharded::Qwen3MoEShardedSession],
    cluster: &flambeau_backend_hip::HipCluster,
    scratches: &mut [&mut super::pp::ShardedForwardOneTokenScratch],
    slots: &[BatchSlot],
    logits_out: &mut [&mut Vec<f32>],
) -> Result<()> {
    if slots.is_empty() {
        bail!("forward_decode_batched_pp: empty slot list");
    }
    if sessions.len() != scratches.len() || sessions.len() != logits_out.len() {
        bail!(
            "forward_decode_batched_pp: parallel arrays disagree (sessions={}, scratches={}, logits={})",
            sessions.len(),
            scratches.len(),
            logits_out.len(),
        );
    }
    for s in slots {
        if s.idx >= sessions.len() {
            bail!(
                "forward_decode_batched_pp: BatchSlot.idx {} out of bounds (n={})",
                s.idx,
                sessions.len()
            );
        }
    }

    // i2-A scaffold: per-slot serial dispatch. i2-A1/A2/A3 replace this
    // body with a batched per-layer driver that gathers [N, hidden]
    // residuals and runs FFN/MoE once per layer instead of N times.
    for slot in slots {
        super::pp::forward_one_token_pp_logits(
            model,
            sessions[slot.idx],
            cluster,
            scratches[slot.idx],
            slot.token_id,
            slot.position,
            logits_out[slot.idx],
        )?;
    }
    Ok(())
}
