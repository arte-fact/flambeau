//! flambeau-bench — sweep / matrix / cert subcommands.
//!
//! V1.3: `sweep` runs the MMVQ correctness grid on real HIP hardware
//! (`--features hip`) and writes `certs/<backend>/<arch>/<impl_id>.json` per
//! dtype. `cert-check` validates every dispatch row has a matching green
//! cert — the build-time gate from architecture rule 2.
//!
//! Matrix / PMC / dispatch-A/B land in V1.4+ on the same scaffolding.

// Kernel sweep harnesses legitimately take many arguments (device + kernel
// handles + shape scalars + seed). Silencing the style lint crate-wide is
// simpler than annotating every `run_*_shape` function.
#![allow(clippy::too_many_arguments)]

pub mod cert;
pub mod dispatch;
pub mod pmc;

#[cfg(feature = "hip")]
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
pub mod sweep_l2_norm;

#[cfg(feature = "hip")]
pub mod sweep_causal_conv1d;

#[cfg(feature = "hip")]
pub mod sweep_gdn_step;

#[cfg(feature = "hip")]
pub mod sweep_cast;

#[cfg(feature = "hip")]
pub mod sweep_f32_pointwise;

#[cfg(feature = "hip")]
pub mod sweep_peer_copy;

#[cfg(feature = "hip")]
pub mod sweep_shared_expert;

#[cfg(feature = "hip")]
pub mod sweep_split_q_gate;

#[cfg(feature = "hip")]
pub mod sweep_softmax;

#[cfg(feature = "hip")]
pub mod sweep_attention;

#[cfg(feature = "hip")]
pub mod sweep_attention_prefill;

#[cfg(feature = "hip")]
pub mod sweep_attention_q8_kv;

#[cfg(feature = "hip")]
pub mod sweep_attention_splitk;

#[cfg(feature = "hip")]
pub mod sweep_mmvq_f16;

#[cfg(feature = "hip")]
pub mod sweep_q4_0_q5_0;

#[cfg(feature = "hip")]
pub mod sweep_moe;
