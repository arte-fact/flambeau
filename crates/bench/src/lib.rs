//! flambeau-bench — sweep / matrix / cert subcommands.
//! `sweep` runs the MMVQ correctness grid on real HIP hardware
//! (`--features hip`) and writes `certs/<backend>/<arch>/<impl_id>.json` per
//! dtype. `cert-check` validates every dispatch row has a matching green
//! cert — the build-time gate from architecture rule 2.
//! Matrix / PMC / dispatch-A/B land in + on the same scaffolding.


pub mod cert;
pub mod dispatch;
pub mod pmc;

#[cfg(any(feature = "hip", feature = "cuda"))]
pub mod backend;

#[cfg(any(feature = "hip", feature = "cuda"))]
pub mod harness;

#[cfg(any(feature = "hip", feature = "cuda"))]
pub mod sweep_mmvq;

#[cfg(feature = "hip")]
pub mod sweep_mmq;

#[cfg(feature = "hip")]
pub mod sweep_quantize_q8_1_mmq;

#[cfg(feature = "hip")]
pub mod pmc_probe;

#[cfg(feature = "hip")]
pub mod sweep_rmsnorm;

#[cfg(feature = "hip")]
pub mod sweep_swiglu;

#[cfg(feature = "hip")]
pub mod sweep_rmsnorm_q8_1;

#[cfg(feature = "hip")]
pub mod sweep_rope;

#[cfg(feature = "hip")]
pub mod sweep_rope_neox;


#[cfg(feature = "hip")]
pub mod sweep_causal_conv1d;

#[cfg(feature = "hip")]
pub mod sweep_gdn_step;

#[cfg(feature = "hip")]
pub mod sweep_gdn_step_alphabeta;


#[cfg(feature = "hip")]
pub mod sweep_f32_pointwise;

#[cfg(any(feature = "hip", feature = "cuda"))]
pub mod sweep_pointwise_cb;

#[cfg(feature = "hip")]
pub mod sweep_peer_copy;

#[cfg(feature = "hip")]
pub mod sweep_shared_expert;


#[cfg(feature = "hip")]
pub mod sweep_softmax;

#[cfg(feature = "hip")]
pub mod sweep_attention;

#[cfg(feature = "hip")]
pub mod sweep_attention_prefill;

#[cfg(feature = "hip")]
pub mod sweep_attention_prefill_q8_kv;

#[cfg(feature = "hip")]
pub mod sweep_attention_q8_kv;

#[cfg(feature = "hip")]
pub mod sweep_attention_q8_kv_splitk;

#[cfg(feature = "hip")]
pub mod sweep_attention_splitk;

#[cfg(any(feature = "hip", feature = "cuda"))]
pub mod sweep_mmvq_f16;

#[cfg(any(feature = "hip", feature = "cuda"))]
pub mod sweep_attention_batched_splitk;

#[cfg(any(feature = "hip", feature = "cuda"))]
pub mod sweep_attention_splitk_dpos;

#[cfg(any(feature = "hip", feature = "cuda"))]
pub mod sweep_kv_append_v_unit_norm_dpos;

#[cfg(any(feature = "hip", feature = "cuda"))]
pub mod sweep_embed_gather_dpos;

#[cfg(any(feature = "hip", feature = "cuda"))]
pub mod sweep_attention_q8_kv_splitk_dpos;

#[cfg(any(feature = "hip", feature = "cuda"))]
pub mod sweep_quantize_f16_q8_0_dpos;
pub mod sweep_kv_append_f16_dpos;

#[cfg(feature = "hip")]
pub mod sweep_q4_0_q5_0;

#[cfg(feature = "hip")]
pub mod sweep_moe;
