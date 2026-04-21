//! flambeau-qwen3-moe — Qwen3.x MoE family composition.
//!
//! V1.7 target: `Qwen3MoEModel` with `forward_one_token` and `forward_prefill`,
//! `Mesh<N>`-generic, weight-name map parsed from GGUF metadata. No new
//! kernels — if this crate needs one that isn't in `flambeau-ops`, fix
//! `flambeau-ops` instead.
//!
//! **V1.7.2 (this commit):** config + weight-name map + layer descriptor.
//! No forward pass yet — that lands in V1.7.3 with the ops wiring.

pub mod config;
pub mod names;
pub mod layout;

#[cfg(feature = "hip")]
pub mod forward;

#[cfg(feature = "hip")]
pub mod model;

#[cfg(feature = "hip")]
pub mod session;

#[cfg(feature = "hip")]
pub mod sharded;

#[cfg(feature = "hip")]
pub mod weights;

#[cfg(feature = "hip")]
pub use model::Qwen3MoEModel;

#[cfg(feature = "hip")]
pub use sharded::{
    Qwen3MoERankSession, Qwen3MoERankShard, Qwen3MoEShardedModel, Qwen3MoEShardedSession,
};

#[cfg(feature = "hip")]
pub use session::{GdnLayerState, LayerCache, Qwen3MoESession};

#[cfg(feature = "hip")]
pub use weights::{
    AttnWeights, DenseAttnWeights, DeviceTensor, FfnWeights, FullAttnWeights, GdnWeights,
    LayerWeights, ModelWeights, SharedExpertWeights,
};

pub use config::{
    AttentionFamily, GdnDims, Qwen3MoEConfig, Qwen3MoEConfigError, RopeSpec, SUPPORTED_ARCHS,
};
pub use layout::{
    DenseAttnTensors, FullAttnTensors, GdnTensors, LayerAttnBlock, LayerDescriptor,
    ModelLayout, MoeFfnTensors, ResolvedTensor, SharedExpertTensors,
};
pub use names::{
    layer_names, CommonNames, DenseAttnNames, FullAttnNames, GdnNames, GlobalNames,
    MoeFfnNames, TensorNames,
};
