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
pub mod driver_base;
pub mod driver_utils;
pub mod layer;
pub mod moe_experts;
pub mod shared_expert;
pub mod sharding;
pub mod topology;

pub use attention::{
    AttentionScratchDims, AttnDecodeSlots, AttnPrefillSlots,
    OwnedStandardAttentionBatchedDecodeScratch, OwnedStandardAttentionDecodeScratch,
    OwnedStandardAttentionPrefillScratch, StandardAttention,
    StandardAttentionBatchedDecodeScratch, StandardAttentionDecodeScratch,
    StandardAttentionPrefillScratch, WeightHandle, MAX_SPLITK_CHUNKS,
};
pub use delta_net::{
    DeltaNetLayer, DeltaNetLayerDecodeScratch, DeltaNetLayerPrefillScratch, DeltaNetScratchDims,
    OwnedDeltaNetLayerDecodeScratch, OwnedDeltaNetLayerPrefillScratch,
};
pub use driver_base::{HybridCluster, HybridStageCluster, TpCluster};
pub use driver_utils::{
    alloc_zeroed, embed_token_host, ggml_to_qdtype, row_bytes_for_dtype, upload_f16_ones,
    RawAllocTracker,
};
pub use dense_mlp::{
    DenseMlp, DenseMlpDecodeScratch, DenseMlpPrefillScratch, DenseMlpScratchDims, DenseMlpTp,
    OwnedDenseMlpDecodeScratch, OwnedDenseMlpPrefillScratch,
};
pub use layer::{
    post_norm_residual_f16, tp_allreduce_sum_into, AttnBlock, AttnDecodeScratch,
    AttnPrefillScratch, AttnState, FfnBlock, FfnDecodeScratch, FfnPrefillScratch, LayerKind,
};
pub use moe_experts::{
    Activation, MoeExperts, MoeExpertsDecodeScratch, MoeExpertsPrefillScratch,
    MoeExpertsScratchDims, OwnedMoeExpertsDecodeScratch, OwnedMoeExpertsPrefillScratch,
    RouterInput, RouterNormalize, RouterPolicy,
};
pub use shared_expert::{
    OwnedSharedExpertDecodeScratch, OwnedSharedExpertPrefillScratch, SharedExpert,
    SharedExpertDecodeScratch, SharedExpertPrefillScratch, SharedExpertScratchDims,
};
pub use sharding::{
    upload_replicated_norm_f32_to_f16, upload_replicated_tensor, upload_sharded_tensor,
    UploadedTensor,
};
pub use topology::{
    forward_one_token_hybrid, forward_one_token_pp, forward_one_token_tp,
    forward_prefill_hybrid, forward_prefill_pp, forward_prefill_pp_chunk, forward_prefill_tp,
    HybridDecodeDriver, HybridPrefillDriver, PpDecodeDriver, PpPrefillDriver, TpDecodeDriver,
    TpPrefillDriver,
};
