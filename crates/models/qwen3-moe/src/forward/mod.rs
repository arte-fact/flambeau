//! Forward-pass composition for a Qwen3.x model.
//!
//! Module map:
//! - `common` — private cross-cutting helpers (`upload_position`,
//!   `qdtype_of`, `mat_shape`, `row_bytes_for_dtype`,
//!   `run_mmvq_from_tensor`, `run_qmatmul_from_tensor`).
//! - `attn` — full-attention decode + prefill (RMSNorm + fused QKV/gate +
//!   RoPE + softmax attention + output projection).
//! - `gdn` — gated-delta-net decode + prefill (hybrid SSM layer).
//! - `dense_ffn` — dense gate/up/down FFN (arch=qwen35).
//! - `moe` — routed experts + shared expert + router (arch=qwen36 MoE).
//! - `io` — token embedding gather + output head + argmax.
//! - `layer` — per-layer composition dispatcher
//!   (`forward_layer_{decode,prefill}` pick attn or gdn and the ffn flavour).
//! - `single_device` — Mesh&lt;1&gt; entry points
//!   (`forward_one_token`, `forward_prefill`).
//! - `pp` — Mesh&lt;N&gt; pipeline-parallel entry points
//!   (`forward_one_token_pp`, `forward_prefill_pp`).
//!
//! Shape conventions for one decode step (single token, Qwen3.6-35B):
//! - hidden `H = 2048`, `n_heads = 16`, `n_kv_heads = 2`, `head_dim = 256`
//! - fused Q|gate projection width: `2 * n_heads * head_dim = 8192`
//! - K/V projection width: `n_kv_heads * head_dim = 512`
//! - post-attention intermediate: `n_heads * head_dim = 4096`
//!
//! All intermediates are F16 except MMVQ accumulator outputs, which are
//! F32 and get cast back with `ops::cast::cast_f32_to_f16`.

#![cfg(feature = "hip")]

mod common;

pub mod attn;
pub use attn::{
    forward_full_attn_decode, forward_full_attn_layer_decode, forward_full_attn_prefill,
    FullAttnPrefillScratch, FullAttnScratch,
};

pub mod gdn;
pub use gdn::{
    forward_gdn_decode, forward_gdn_layer_decode, forward_gdn_prefill,
    GdnPrefillScratch, GdnScratch,
};

pub mod dense_ffn;
pub use dense_ffn::{
    forward_dense_ffn_decode, forward_dense_ffn_prefill, DenseFfnPrefillScratch, DenseFfnScratch,
};

pub mod moe;
pub use moe::{
    forward_moe_ffn_decode, forward_moe_ffn_prefill, forward_router_decode,
    forward_router_prefill, forward_shared_expert_decode, forward_shared_expert_prefill,
    MoePrefillScratch, MoeScratch, SharedExpertPrefillScratch, SharedExpertScratch,
};

pub mod io;
pub use io::{
    argmax_token_host, download_logits_host, forward_embed_decode_host,
    forward_embed_prefill_batch, forward_output_head_decode, EmbedPrefillHostScratch,
    OutputHeadScratch,
};

pub mod layer;
pub use layer::{
    forward_layer_decode, forward_layer_prefill, LayerForwardScratch, LayerPrefillScratch,
};

pub mod single_device;
pub use single_device::{
    forward_one_token, forward_prefill, ForwardOneTokenScratch, ForwardPrefillScratch,
};

pub mod pp;
pub use pp::{
    forward_one_token_pp, forward_one_token_pp_logits, forward_prefill_pp,
    forward_prefill_pp_async, forward_prefill_pp_logits, RankForwardPrefillScratch,
    RankForwardScratch, ShardedForwardOneTokenScratch, ShardedForwardPrefillScratch,
    UbatchLane,
};
