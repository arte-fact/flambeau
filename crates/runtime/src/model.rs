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
