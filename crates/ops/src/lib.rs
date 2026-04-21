//! flambeau-ops — typed fused building blocks used by model crates.
//!
//! Everything here is `Mesh<N>`-generic: `Mesh<1>` is a degenerate single-GPU
//! instance routed through the same trait. Model crates import only from
//! here, never from `backend-*` or `kernels-*`.
//!
//! ## V1.7.1 surface
//!
//! - [`OpsRegistry`] — one-shot loader for every HIP kernel module the model
//!   touches. Model code holds `&OpsRegistry` for the life of the session.
//! - `qmatmul` — quantised weight × activation matmul. Runtime-dispatched to
//!   MMVQ or MMQ by `(dtype, m)`; all K-quants routed through the existing
//!   `dispatch/hip/gfx906.toml` rows.
//! - `norm` — RMSNorm (F16 in, F16 out) + fused RMSNorm+Quantize<Q8_1>.
//! - `pe` — RoPE in-place (F16, interleaved-pair).
//! - `mlp` — SwiGLU pointwise.
//! - `softmax` — masked-and-scaled softmax.
//! - `attention` — decode (F16 KV, Q8 KV) + prefill (F16).
//! - `moe` — TopK router, IndexedMoE MMVQ + MMQ + fused gate+up, combine.
//!
//! Each op module exposes stateless free functions that accept `&OpsRegistry`
//! + device pointers + shape. The registry owns the `HipModule`s; each launch
//! looks up the kernel by name (cheap) and fires. No per-call hsaco parsing.
//!
//! Mesh-genericity: the HIP-specialised surface below takes a single
//! `HipDevice` / `HipStream`. V1.7.5 wraps this behind a `MeshOps<M: Mesh>`
//! trait that lives alongside the collective ops in `flambeau-runtime`.

#[cfg(feature = "hip")]
pub mod hip;

#[cfg(feature = "hip")]
pub use hip::{OpsRegistry, OpsRegistryError};

#[cfg(feature = "hip")]
pub use hip::{attention, conv, mlp, moe, norm, pe, qmatmul, softmax};
