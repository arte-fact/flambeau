//! Arch-agnostic GGUF → device helpers. Every helper appends
//! `(DevicePtr, bytes)` to a caller-supplied `&mut Vec` so one
//! disposer walks every alloc.
//!
//! Two layers of API:
//! - byte/tensor primitives (`upload_bytes`, `wrap_quant`,
//!   `upload_*_sharded_quant`) at the bottom — what the existing TP
//!   sharded paths use directly.
//! - per-layer-kind composers (`load_embedding`, `load_lm_head`,
//!   `load_dense_attn_layer`, `load_dense_ffn_layer`,
//!   `load_gdn_layer`) on top — what arch-specific model loaders
//!   call. Each composer takes a `ShardMode` and a spec struct; the
//!   model loader is just per-arch tensor-name + per-layer-dim glue.

mod dense_attn;
mod dense_ffn;
mod gdn_layer;
mod gdn_shard;
mod globals;
mod primitives;
mod shard;

pub use dense_attn::{load_dense_attn_layer, DenseAttnLayerSpec};
pub use dense_ffn::{load_dense_ffn_layer, DenseFfnLayerSpec};
pub use gdn_layer::{load_gdn_layer, GdnLayerSpec, GdnTpMode};
pub use gdn_shard::{
    upload_f32_array_sharded, upload_gdn_fused_qkv_f32, upload_gdn_fused_qkv_quant,
};
pub use globals::{load_embedding, load_lm_head, EmbeddingSpec, LmHeadSpec};
pub use primitives::{
    dtype_qmatmul_native, ggml_to_qdtype, upload_bytes, upload_dequant_to_f16, upload_f16_from_f32,
    upload_f32_tensor, upload_raw, wrap_quant,
};
pub use shard::{upload_col_sharded_quant, upload_quant_weight, upload_row_sharded_quant};

/// How a matmul weight gets uploaded.
#[derive(Clone, Copy, Debug)]
pub enum ShardMode {
    Replicated,
    Tp { rank: usize, n_ranks: usize },
}

impl ShardMode {
    pub fn n_ranks(self) -> usize {
        match self {
            ShardMode::Replicated => 1,
            ShardMode::Tp { n_ranks, .. } => n_ranks,
        }
    }
}
