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

    /// Free all device allocations. Idempotent.
    fn dispose(&mut self) -> anyhow::Result<()>;
}
