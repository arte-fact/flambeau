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
}
