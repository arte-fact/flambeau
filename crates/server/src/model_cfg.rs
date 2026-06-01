//! Model-agnostic config slice held on [`crate::ServerState`].
//!
//! The server's routes / handlers only need a handful of fields from
//! the loaded model: `arch`, `vocab_size`, `context_length`, and
//! `num_layers`. The arch-specific config types live in their model
//! crates; v2 boot paths populate this struct directly from GGUF.

#[derive(Debug, Clone)]
pub struct ServerModelCfg {
    pub arch: String,
    pub vocab_size: usize,
    pub context_length: usize,
    pub num_layers: usize,
    /// PP-split boundary forced by cross-rank KV sharing constraints
    /// (gemma4 4n / E2B / E4B). When `Some(b)`, the last PP rank must
    /// own `[b..num_layers]` so every `kv_share_src` reference resolves
    /// within the rank's local KV pool. `None` for archs without
    /// shared KV layers — the default even split applies.
    pub kv_share_pp_boundary: Option<usize>,
}
