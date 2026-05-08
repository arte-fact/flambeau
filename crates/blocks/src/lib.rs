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
pub mod delta_net;
pub mod dense_mlp;
pub mod layer;
pub mod moe_experts;
pub mod shared_expert;
pub mod topology;

pub use attention::{
    AttnDecodeSlots, AttnPrefillSlots, StandardAttention, StandardAttentionDecodeScratch,
    StandardAttentionPrefillScratch, WeightHandle, MAX_SPLITK_CHUNKS,
};
pub use delta_net::{DeltaNetLayer, DeltaNetLayerDecodeScratch};
pub use dense_mlp::{DenseMlp, DenseMlpDecodeScratch, DenseMlpPrefillScratch};
pub use layer::{
    AttnBlock, AttnDecodeScratch, AttnPrefillScratch, AttnState, FfnBlock, FfnDecodeScratch,
    FfnPrefillScratch, LayerKind,
};
pub use moe_experts::{MoeExperts, MoeExpertsDecodeScratch};
pub use shared_expert::{SharedExpert, SharedExpertDecodeScratch};
pub use topology::{
    forward_one_token_hybrid, forward_one_token_pp, forward_one_token_tp,
    forward_prefill_hybrid, forward_prefill_pp, forward_prefill_pp_chunk, forward_prefill_tp,
    HybridDecodeDriver, HybridPrefillDriver, PpDecodeDriver, PpPrefillDriver, TpDecodeDriver,
    TpPrefillDriver,
};
