//! flambeau-qwen3-moe — Qwen3.x MoE family composition.
//!
//! V1.7 target: `Qwen3MoEModel` with `forward_one_token` and `forward_prefill`,
//! `Mesh<N>`-generic, weight-name map parsed from GGUF metadata. No new kernels —
//! if this crate needs one that isn't in `flambeau-ops`, fix `flambeau-ops` instead.
