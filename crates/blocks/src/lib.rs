//! flambeau-blocks — reusable model-building blocks.
//!
//! Each block is generic over a backend `Ops` trait from `flambeau-ops`.
//! Blocks compose Ops calls into the patterns model crates need:
//! `StandardAttention`, `MoeExperts`, `DenseMlp`, `DeltaNetLayer`,
//! `Rope`, etc. Adding a new model becomes "wire layers from blocks";
//! adding a new backend becomes "implement `Ops`".
//!
//! Blocks today take a HIP-flavored `(device, stream)` pair alongside
//! `&O: &impl Ops` because non-kernel work — host→device memcpy of the
//! position buffer, KV-cache append — is not yet on the Ops trait.
//! When CUDA arrives we will introduce a `Backend` abstraction at this
//! layer; until then, the structural value is "kernel dispatch flows
//! through `Ops`, the rest is a thin per-backend glue layer".

#![cfg(feature = "hip")]

pub mod attention;
pub mod dense_mlp;

pub use attention::{
    StandardAttention, StandardAttentionDecodeScratch, StandardAttentionPrefillScratch,
    WeightHandle, MAX_SPLITK_CHUNKS,
};
pub use dense_mlp::{DenseMlp, DenseMlpDecodeScratch, DenseMlpPrefillScratch};
