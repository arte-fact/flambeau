//! flambeau-gemma4 — Gemma 4 family composition.
//!
//! **S4 (this commit):** config parser + tensor-name resolver +
//! per-layer descriptor table + non-device weight presence/shape
//! checks. No device upload, no forward pass — those land in S5.

pub mod config;
pub mod layout;
pub mod model_arch;
pub mod names;
pub mod weights;

#[cfg(feature = "hip")]
pub mod layer;
#[cfg(feature = "hip")]
pub mod moe;
#[cfg(feature = "hip")]
pub mod output_head;
#[cfg(feature = "hip")]
pub mod per_layer_embd;
#[cfg(feature = "hip")]
pub mod hybrid;
#[cfg(feature = "hip")]
pub mod pp;
#[cfg(feature = "hip")]
pub mod tp;
#[cfg(feature = "hip")]
pub mod scratch;
#[cfg(feature = "hip")]
pub mod session;
#[cfg(feature = "hip")]
pub mod single_device;
#[cfg(feature = "hip")]
pub mod softcap;
#[cfg(feature = "hip")]
pub mod weights_hip;

pub use config::{
    Gemma4Config, Gemma4ConfigError, Gemma4Variant, MoeDims, PerLayerEmbed, SUPPORTED_ARCHS,
};
pub use layout::{FfnKind, LayerSpec, ModelLayout};
pub use model_arch::Gemma4ModelArch;
pub use names::{
    AttnNames, DenseFfnNames, GlobalNames, MoeFfnNames, PerLayerEmbedNames,
};
pub use weights::{
    resolve_weights, validate_shapes, AttnTensors, DenseFfnTensors, Gemma4WeightsError,
    GlobalTensors, LayerTensors, MoeFfnTensors, PerLayerEmbedTensors, ResolvedWeights,
};

#[cfg(feature = "hip")]
pub use layer::{forward_layer_decode, forward_layer_prefill, Gemma4LayerWeights};
#[cfg(feature = "hip")]
pub use moe::{forward_ffn_moe, Gemma4MoeFfnWeights, Gemma4MoeScratch};
#[cfg(feature = "hip")]
pub use output_head::{forward_output_head, OutputHeadScratch};
#[cfg(feature = "hip")]
pub use per_layer_embd::{
    build_inp_per_layer_table, forward_per_layer_post_block, per_layer_token_embd_row_bytes,
    table_slice_ptr, upload_inp_per_layer_table, PerLayerEmbedGlobals,
    PerLayerEmbedLayerWeights,
};
#[cfg(feature = "hip")]
pub use scratch::{LayerDecodeScratch, LayerPrefillScratch};
#[cfg(feature = "hip")]
pub use session::Gemma4Session;
#[cfg(feature = "hip")]
pub use single_device::{forward_one_token, forward_one_token_logits};
#[cfg(feature = "hip")]
pub use softcap::apply_logit_softcap;
#[cfg(feature = "hip")]
pub use weights_hip::{DeviceTensor, Gemma4DeviceWeights};
#[cfg(feature = "hip")]
pub use pp::{partition_layers, Gemma4PpDriver, Gemma4PpStage};
#[cfg(feature = "hip")]
pub use tp::{Gemma4TpDriver, Gemma4TpStage};
#[cfg(feature = "hip")]
pub use hybrid::{
    partition_layers_pp, Gemma4HybridDriver, Gemma4HybridStage, HybridRankState,
};
