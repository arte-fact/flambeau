//! Backend-generic op-launch bodies.
//!
//! Every model-visible op lands here as a stateless free function generic over
//! the [`core::KernelBackend`](flambeau_core::KernelBackend) launch seam:
//! `fn op<B: KernelBackend>(reg: &Registry<B>, stream: &<B::Device as Device>::Stream, …)`.
//! One body serves HIP and CUDA — the only backend-specific things are the
//! module / kernel / args / launch *types*, projected off `B`; the dispatch
//! arithmetic (shape → variant, grid / block) is backend-identical.
//!
//! Variant selection resolves through the committed `KernelDescriptor` tables
//! via `B::dispatch_qmatmul` (architectural rule 1: dispatch lives in
//! `dispatch/<backend>/<arch>.toml`, mirrored by the Rust table) — never an env
//! predicate. Each op call borrows the registry and a stream; all kernel args
//! are locals so they live until the launch returns.

pub mod attention;
pub mod cast;
pub mod conv;
pub mod mlp;
pub mod moe;
pub mod norm;
pub mod pe;
pub mod qmatmul;
pub mod recurrent;
pub mod router;
pub mod sampling;
pub mod softcap;
pub mod softmax;
