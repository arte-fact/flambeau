//! `LogitsSink` — common reducer hatch for per-arch single-token forward.
//!
//! Replaces per-arch sibling sinks (qwen3-moe's `PpLogitsSink` enum,
//! gemma4's split fn-per-sink). Arch-specific `forward_one_token_*`
//! call sites dispatch on the sink variant after the LM-head matmul
//! produces the F32 `[vocab]` row:
//! * `Host(vec)` — DtoH-download into `vec`, resized to `vocab_size`.
//!   Used by the chat sampler.
//! * `Argmax` — host-side argmax of the F32 row; the topology forward
//!   returns the chosen token id as `u32` instead of `()`.
//! * `KeepOnDevice` — leave the F32 row in the head rank's
//!   `output_head.logits_f32` device buffer. Caller (GPU sampler)
//!   consumes the device pointer BEFORE the next forward clobbers it.

#![cfg(feature = "hip")]

/// Where a `forward_one_token_*` call should write its F32 logits row.
pub enum LogitsSink<'a> {
    Host(&'a mut Vec<f32>),
    Argmax,
    KeepOnDevice,
}

impl<'a> LogitsSink<'a> {
    pub fn is_host(&self) -> bool {
        matches!(self, LogitsSink::Host(_))
    }
    pub fn is_argmax(&self) -> bool {
        matches!(self, LogitsSink::Argmax)
    }
    pub fn is_keep_on_device(&self) -> bool {
        matches!(self, LogitsSink::KeepOnDevice)
    }
}
