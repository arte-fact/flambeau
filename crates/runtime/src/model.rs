//! Backend-portable `Model` trait.
//!
//! V1 surface is intentionally minimal: each `Model` implementor names
//! the GGUF `general.architecture` strings it handles plus a
//! description for diagnostics. Forward methods land in R5.2+ as the
//! topology drivers hoist out of model crates.
//!
//! A `Model` impl is a thin pointer to "which crate handles this
//! arch". Concrete model state (weights, sessions, scratches) is
//! allocated by the model crate on demand; the trait only holds
//! identity + capability metadata.

/// One model architecture's runtime entry. Implementors live in their
/// own crate (`flambeau-qwen3-moe`, future `flambeau-gemma4`, etc.)
/// and declare which GGUF arch keys they handle.
pub trait Model: Send + Sync + 'static {
    /// GGUF `general.architecture` strings this model accepts.
    /// `Registry::find` matches the GGUF's arch field against this
    /// list. A single `Model` may cover multiple GGUF aliases (e.g.
    /// Qwen3-MoE handles `qwen35moe`, `qwen3moe`, and `qwen3next`).
    fn supported_archs(&self) -> &[&'static str];

    /// Brief human-readable description for boot logs and registry
    /// diagnostics. Not parsed.
    fn description(&self) -> &'static str;
}

/// Backend-agnostic, single-stream forward-pass surface.
///
/// Each model crate's topology driver (e.g. `Gemma4PpDriver`,
/// `Gemma4TpDriver`, future `Qwen3MoeDriver`) implements this so the
/// CLI / server can drive prefill + decode without knowing the
/// concrete model type. Argmax of the final-token logits is returned
/// (greedy sampling); samplers that need raw logits should land on a
/// follow-up trait method.
///
/// Construction is model-specific (each model crate exposes its own
/// `build_*` constructor that takes the GGUF + cluster). This trait
/// only covers the run-time forward calls.
pub trait ModelDriver: Send + 'static {
    /// Run a multi-token prefill, returning the argmax of the last
    /// token's logits. `start_position` is the cache tail length
    /// before this chunk's K/V are appended. KV-cache state is
    /// mutated in place.
    fn forward_prefill(&mut self, tokens: &[u32], start_position: usize) -> anyhow::Result<u32>;

    /// Run a single-token decode, returning the argmax of the new
    /// token's logits. `position` is the cache tail length before
    /// `token_id`'s K/V are appended.
    fn forward_one_token(&mut self, token_id: u32, position: usize) -> anyhow::Result<u32>;

    /// Prefill variant that copies the final-token F32 logits into
    /// `logits_out` (caller-owned, resized to vocab) instead of
    /// argmax-ing on host. Lets samplers (top-k / top-p / temperature)
    /// consume raw logits.
    ///
    /// Default impl bails so existing greedy-only drivers compile
    /// without churn. Impls that support sampling should override.
    fn forward_prefill_logits(
        &mut self,
        _tokens: &[u32],
        _start_position: usize,
        _logits_out: &mut Vec<f32>,
    ) -> anyhow::Result<()> {
        anyhow::bail!("ModelDriver: forward_prefill_logits not implemented by this driver")
    }

    /// Decode variant that copies the new-token F32 logits into
    /// `logits_out` instead of argmax-ing on host.
    ///
    /// Default impl bails — see [`Self::forward_prefill_logits`].
    fn forward_one_token_logits(
        &mut self,
        _token_id: u32,
        _position: usize,
        _logits_out: &mut Vec<f32>,
    ) -> anyhow::Result<()> {
        anyhow::bail!("ModelDriver: forward_one_token_logits not implemented by this driver")
    }

    /// Copy the slot's resident attention state for positions
    /// `[0..n_tokens)` (KV rows + recurrent state) to host buffers, one
    /// per rank. Restorable via [`Self::restore_slot`] on an
    /// identically-configured driver.
    ///
    /// Default impl bails — see [`Self::forward_prefill_logits`].
    fn snapshot_slot(&mut self, _n_tokens: usize) -> anyhow::Result<Vec<Vec<u8>>> {
        anyhow::bail!("ModelDriver: snapshot_slot not implemented by this driver")
    }

    /// Inverse of [`Self::snapshot_slot`]. After a successful restore the
    /// slot's state is exactly the post-position-`n_tokens` state; the
    /// caller must continue from position `n_tokens` — recurrent (GDN)
    /// layers cannot re-run earlier tokens. The snapshot is shared via
    /// `Arc` (multi-hundred-MB buffers; no caller-side copy).
    ///
    /// Default impl bails — see [`Self::forward_prefill_logits`].
    fn restore_slot(
        &mut self,
        _n_tokens: usize,
        _snaps: std::sync::Arc<Vec<Vec<u8>>>,
    ) -> anyhow::Result<()> {
        anyhow::bail!("ModelDriver: restore_slot not implemented by this driver")
    }

    /// Vocab size — needed by callers to pre-size the `logits_out`
    /// buffer. Default impl bails so existing impls can opt in.
    fn vocab_size(&self) -> usize {
        0
    }

    /// Reset per-request state (KV cache write tails, position
    /// counters) so the next request starts fresh. Weights are
    /// untouched. Default impl bails so the server's
    /// `reset_for_next_request` can detect arches that haven't opted
    /// in to slot reuse.
    fn reset_kv(&mut self) -> anyhow::Result<()> {
        anyhow::bail!("ModelDriver::reset_kv: not implemented for this arch")
    }

    /// Free all device allocations. Idempotent.
    fn dispose(&mut self) -> anyhow::Result<()>;
}
