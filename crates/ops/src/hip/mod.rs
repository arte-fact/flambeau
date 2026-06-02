//! HIP-specialised op surface.
//! Every model-visible op lands here as a stateless free function taking
//! `&OpsRegistry` + `&HipStream` + device pointers + shape. The registry
//! pre-loads one `HipModule` per kernel stem at session init; per-call cost
//! is one symbol lookup (`hipModuleGetFunction`) + `hipModuleLaunchKernel`.
//! Architectural placement:
//! - Op wrappers here **never** parse hsaco, never write dispatch predicates
//! from env flags. They resolve variant selection through the committed
//! `KernelDescriptor` tables in `flambeau-backend-hip::impls` (architectural
//! rule 1: dispatch lives in `dispatch/<backend>/<arch>.toml`, mirrored by
//! the Rust table).
//! - The lifetime story: each op call borrows `&OpsRegistry` and a `&HipStream`;
//! all kernel args are locals in the launch function so they live until
//! `kernel.launch(...)` returns. This matches the pattern already in
//! `flambeau-bench::sweep_*`.

use std::collections::HashMap;

use anyhow::{anyhow, Result};
use flambeau_backend_hip::HipModule;
pub use flambeau_backend_hip::{HipDevice, HipStream};
use flambeau_core::Device;

pub mod attention;
pub mod cast;
pub mod conv;
pub mod mlp;
pub mod moe;
pub mod norm;
mod ops_impl;
pub mod pe;
pub mod qmatmul;
pub mod recurrent;
pub mod router;
pub mod sampling;
pub mod softcap;
pub mod softmax;

pub use ops_impl::HipOps;

/// Kernel stems every model might touch. Loaded once in
/// [`OpsRegistry::new`]; missing entries fail fast so model code never races
/// an unloaded module.
/// Keep this list aligned with `kernels-hip/src/kernels/*.cu`. The build's
/// `hsaco.rs` lists the authoritative set in its `CATALOGUE`.
pub const KERNEL_STEMS: &[&str] = &[
    // Qmatmul — MMVQ + MMQ, all dtypes.
    "mmvq_q8_0",
    "mmvq_q8_0_dp4a",
    "mmvq_q8_0_dp4a_vdr2",
    "mmvq_q8_0_r4_dp4a",
    "mmvq_q8_0_t128",
    "mmvq_q8_0_t128_vdr2",
    "mmvq_q8_0_gate_up_t128_vdr2",
    "mmvq_q8_0_llamacpp_style",
    "mmvq_q8_0_gate_up_dp4a",
    "mmvq_q4_0_gate_up_dp4a",
    "mmvq_q4_0_kv_f16dst_dp4a",
    "mmvq_q4_0_t128",
    "mmvq_q4_0_gate_up_t128_dp4a",
    "mmvq_q4_0_warpcoop64",
    "mmvq_q4_0_batched",
    "mmvq_q4_0_row_tile_batched",
    "mmvq_q8_0_batched",
    "mmvq_q8_0_row_tile_batched",
    "mmvq_q4_k_batched",
    "mmvq_q6_k_batched",
    "mmvq_q4_0_gate_up_batched",
    "mmvq_q4_0_gate_up_row_tile_batched",
    "mmvq_q5_k_r2_batched",
    "mmvq_q5_k_row_tile_batched",
    "mmvq_q2_k",
    "mmvq_q2_k_r2",
    "mmvq_q3_k",
    "mmvq_q3_k_r2",
    "mmvq_q8_k",
    "mmvq_q4_k",
    "mmvq_q4_1",
    "mmvq_q4_1_batched",
    "mmvq_q4_1_r2",
    "mmvq_q4_1_r2_dp4a",
    "mmvq_q4_1_t128",
    "mmvq_q4_1_gate_up_dp4a",
    "indexed_moe_mmvq_q4_k_r2_dp4a",
    "indexed_moe_mmvq_q4_k_r4_dp4a",
    "indexed_moe_mmvq_q4_k_r4_sorted_dp4a",
    "indexed_moe_mmq_q4_k_gate_up_tile8_dp4a",
    "indexed_moe_mmq_q4_0_gate_up_tile8_dp4a",
    "indexed_moe_mmq_q4_0_down_tile8_dp4a",
    "indexed_moe_mmq_q5_0_gate_up_tile8_dp4a",
    "indexed_moe_mmq_q5_0_down_tile8_dp4a",
    "indexed_moe_mmq_q5_1_gate_up_tile8_dp4a",
    "indexed_moe_mmq_q5_1_down_tile8_dp4a",
    "indexed_moe_mmq_q4_1_down_tile8_dp4a",
    "indexed_moe_mmq_q4_1_gate_up_tile8_dp4a",
    "indexed_moe_mmq_q8_0_gate_up_tile8_dp4a",
    "indexed_moe_mmq_q8_0_down_tile8_dp4a",
    "indexed_moe_mmq_q4_k_down_tile8_dp4a",
    "indexed_moe_mmq_q5_k_down_tile8_dp4a",
    "indexed_moe_mmq_q5_k_gate_up_tile8_dp4a",
    "indexed_moe_mmq_q6_k_gate_up_tile8_dp4a",
    "indexed_moe_mmq_q6_k_down_tile8_dp4a",
    "indexed_moe_mmq_q4_k_gate_up_turbo",
    "indexed_moe_mmq_q4_k_down_turbo",
    "indexed_moe_mmvq_q4_k_gate_up_dp4a",
    "indexed_moe_mmvq_q4_k_gate_up_r2_dp4a",
    "indexed_moe_mmvq_q4_k_gate_up_r4_dp4a",
    "indexed_moe_mmvq_q4_k_gate_up_r8_dp4a",
    "indexed_moe_mmvq_q4_k_gate_up_r4_sorted_dp4a",
    "mmvq_q3_k_dp4a",
    "mmvq_q4_k_r2",
    "mmvq_q4_k_r2_dp4a",
    "mmvq_q4_k_r4",
    "mmvq_q5_k",
    "mmvq_q5_k_r2",
    "mmvq_q5_k_dp4a",
    "mmvq_iq4_nl",
    "mmvq_iq4_nl_r2",
    "mmvq_iq4_xs",
    "mmvq_iq4_xs_r2",
    "mmvq_iq3_xxs",
    "mmvq_iq3_xxs_r2",
    "mmvq_iq3_s",
    "mmvq_iq3_s_r2",
    "mmvq_iq2_xxs",
    "mmvq_iq2_xxs_r2",
    "mmvq_iq2_xs",
    "mmvq_iq2_xs_r2",
    "mmvq_iq2_s",
    "mmvq_iq2_s_r2",
    "mmvq_iq1_s",
    "mmvq_iq1_s_r2",
    "mmvq_iq1_m",
    "mmvq_iq1_m_r2",
    "mmq_iq4_xs_wave64",
    "mmq_iq3_s_wave64",
    "mmq_iq4_nl_wave64",
    "mmq_iq3_xxs_wave64",
    "mmq_iq2_xxs_wave64",
    "mmq_iq2_xs_wave64",
    "mmq_iq2_s_wave64",
    "mmq_iq1_s_wave64",
    "mmq_iq1_m_wave64",
    "indexed_moe_mmvq_iq4_xs",
    "indexed_moe_mmvq_iq4_nl",
    "indexed_moe_mmvq_iq3_xxs",
    "indexed_moe_mmvq_iq3_s",
    "indexed_moe_mmvq_iq2_xxs",
    "indexed_moe_mmvq_iq2_xs",
    "indexed_moe_mmvq_iq2_s",
    "indexed_moe_mmvq_iq1_s",
    "indexed_moe_mmvq_iq1_m",
    "indexed_moe_mmq_iq4_xs_gate_up_tile8_dp4a",
    "indexed_moe_mmq_iq4_nl_gate_up_tile8_dp4a",
    "indexed_moe_mmq_iq3_xxs_gate_up_tile8_dp4a",
    "indexed_moe_mmq_iq3_s_gate_up_tile8_dp4a",
    "indexed_moe_mmq_iq2_xxs_gate_up_tile8_dp4a",
    "indexed_moe_mmq_iq2_xs_gate_up_tile8_dp4a",
    "indexed_moe_mmq_iq2_s_gate_up_tile8_dp4a",
    "indexed_moe_mmq_iq1_s_gate_up_tile8_dp4a",
    "indexed_moe_mmq_iq1_m_gate_up_tile8_dp4a",
    "indexed_moe_mmq_iq4_xs_down_tile8_dp4a",
    "indexed_moe_mmq_iq4_nl_down_tile8_dp4a",
    "indexed_moe_mmq_iq3_xxs_down_tile8_dp4a",
    "indexed_moe_mmq_iq3_s_down_tile8_dp4a",
    "indexed_moe_mmq_iq2_xxs_down_tile8_dp4a",
    "indexed_moe_mmq_iq2_xs_down_tile8_dp4a",
    "indexed_moe_mmq_iq2_s_down_tile8_dp4a",
    "indexed_moe_mmq_iq1_s_down_tile8_dp4a",
    "indexed_moe_mmq_iq1_m_down_tile8_dp4a",
    "mmvq_q6_k",
    "mmvq_q6_k_r4",
    "mmvq_q6_k_dp4a",
    "mmq_q8_0_oracle",
    "mmq_q8_0_4warp",
    "mmq_q8_0_wave64",
    "mmq_q8_0_wave64_tile16",
    "mmvq_f16_q8_1",
    "mmq_f16_q8_1",
    "mmq_f16_tile",
    "mmvq_q4_0",
    "mmvq_q5_0",
    "mmvq_q5_1",
    "indexed_moe_mmvq_q4_0",
    "indexed_moe_mmvq_q5_0",
    "indexed_moe_mmvq_q5_1",
    "indexed_moe_mmvq_q4_0_gate_up_dp4a",
    "indexed_moe_mmvq_q4_1",
    "gdn_split_qkv_f32",
    "gdn_assemble_conv_input_f32",
    "gdn_conv_trio_decode_f32_batched_slots",
    "moe_sort_by_expert",
    "mmq_q4_1_4warp_lds",
    "mmq_q4_1_wave64",
    "mmq_q4_0_wave64",
    "mmq_q4_0_4warp_lds",
    "mmq_q5_0_wave64",
    "mmq_q5_1_wave64",
    "mmq_q4_K_4warp",
    "mmq_q4_K_turbo",
    "mmq_q4_K_wave64",
    "mmq_q2_K_wave64",
    "mmq_q3_K_wave64",
    "mmq_q5_K_wave64",
    "mmq_q8_K_wave64",
    "mmq_q6_K_4warp",
    "mmq_q6_K_wave64",
    // Activation quantisation + glue.
    "quantize_q8_1",
    "quantize_q8_1_mmq",
    "quantize_f16_q8_1_mmq",
    "quantize_f16_q8_1",
    "quantize_f16_q8_0",
    "cast_f32_f16",
    "cast_f16_f32",
    // Norm / pointwise.
    "rmsnorm_f16",
    "rmsnorm_f16_add_residual",
    "v_unit_norm_per_head_f16",
    "rmsnorm_f32",
    "rmsnorm_f32_to_f16",
    "rmsnorm_q8_1_fused",
    "l2_norm_f32",
    "causal_conv1d_f32",
    "gdn_alpha_beta_f32",
    "gdn_state_step_f32",
    "gdn_state_step_alphabeta_f32",
    "gdn_state_step_alphabeta_f32_batched_slots",
    "shared_expert_scale_f32",
    "split_q_gate_f16",
    "swiglu_f16",
    "swiglu_f32",
    "swiglu_f32_to_f16",
    "swiglu_f32_to_q8_1",
    "silu_f32",
    "sigmoid_mul_f16",
    "scale_f32",
    "scale_f16",
    "add_f16",
    "add_f32",
    "rope_f16",
    "rope_neox_partial_f16",
    "rmsnorm_rope_neox_partial_f16",
    "kv_append_v_unit_norm_f16",
    "softmax_masked_f16",
    // Attention.
    "attention_decode_f16",
    "attention_decode_f16_batched",
    "attention_decode_f16_paged",
    "kv_append_f16_batched_slots",
    "kv_append_f16_paged_slots",
    "kv_append_f16_paged_prefill",
    "attention_decode_f16_splitk",
    // "attention_decode_f16_splitk_h2",
    "attention_decode_q8_kv",
    "attention_decode_q8_kv_splitk",
    "attention_prefill_f16",
    "attention_prefill_f16_paged",
    "attention_prefill_flash_tile_f16",
    "attention_prefill_q8_kv",
    "attention_prefill_flash_tile_q8_kv",
    // MoE.
    "topk_f32",
    "apply_per_expert_scale_f32",
    "indexed_moe_mmvq_q4_k",
    "indexed_moe_mmvq_q4_k_r2",
    "indexed_moe_mmvq_q4_k_gate_up",
    "indexed_moe_mmvq_q5_k",
    "indexed_moe_mmvq_q2_k",
    "indexed_moe_mmvq_q3_k",
    "indexed_moe_mmq_q2_k_down_tile8_dp4a",
    "indexed_moe_mmq_q2_k_gate_up_tile8_dp4a",
    "indexed_moe_mmq_q3_k_down_tile8_dp4a",
    "indexed_moe_mmq_q3_k_gate_up_tile8_dp4a",
    "indexed_moe_mmvq_q6_k",
    "indexed_moe_mmvq_q8_0",
    "indexed_moe_mmvq_q8_0_gate_up_dp4a",
    "indexed_moe_mmq_q4_k",
    "moe_combine_f16",
    "moe_combine_two_residuals_f16",
    "moe_combine_no_residual_f16",
    "moe_combine_no_residual_f32",
    "dense_gemv_f32_f16",
    "dense_gemv_f32_f16_batched",
    "dense_gemv_f16_f16",
    "dense_gemv_f16_f16_batched",
    // Sampler-D (#211, #212) — GPU-side sampler kernels for the chat hot path.
    "sampler_topk_softmax_f32",
    "sampler_apply_penalties_f32",
    // Gemma4 — final logit softcap.
    "apply_softcap_f32",
    // Gemma4 — GELU-based FFN + per-layer side-channel.
    "gelu_f32_to_f16",
    "gelu_mul_f32",
];

/// Single-session registry of loaded HIP kernel modules. Built once at model
/// init, passed by shared reference into every op call.
pub struct OpsRegistry {
    device_id: i32,
    modules: HashMap<&'static str, HipModule>,
}

/// Sugar so the loader's `impl_id` → `not loaded` mismatch becomes an error
/// at the call site instead of a panic in `unwrap`.
#[derive(Debug, thiserror::Error)]
pub enum OpsRegistryError {
    #[error("kernel stem `{0}` not present in hsaco catalogue")]
    Missing(&'static str),
    #[error("HIP module load for `{stem}` failed: {source}")]
    Load {
        stem: &'static str,
        #[source]
        source: anyhow::Error,
    },
}

impl OpsRegistry {
    /// Load every entry in [`KERNEL_STEMS`] into a fresh `HipModule`. Fails
    /// fast if any stem is missing from the catalogue or HIP refuses the load.
    /// Binds `dev` first so the modules load onto the right device even when
    /// the caller built multiple `OpsRegistry`s back-to-back on a multi-GPU
    /// cluster — `hipModuleLoadData` silently picks the thread's current
    /// device, and the resulting "invalid device ordinal" at launch time is
    /// very hard to trace without this bind.
    pub fn new(dev: &HipDevice) -> Result<Self, OpsRegistryError> {
        dev.bind().map_err(|e| OpsRegistryError::Load {
            stem: "<bind>",
            source: anyhow!("{e:?}"),
        })?;
        let mut modules = HashMap::with_capacity(KERNEL_STEMS.len());
        for &stem in KERNEL_STEMS {
            let bytes = flambeau_kernels_hip::hsaco(stem).ok_or(OpsRegistryError::Missing(stem))?;
            let module = HipModule::load(dev.id(), bytes).map_err(|e| OpsRegistryError::Load {
                stem,
                source: anyhow!("{e:?}"),
            })?;
            modules.insert(stem, module);
        }
        Ok(Self {
            device_id: dev.id(),
            modules,
        })
    }

    /// Look up a previously-loaded module by kernel stem. Returns `None` if
    /// `new` was called without that stem (i.e. `KERNEL_STEMS` drift).
    pub fn module(&self, stem: &str) -> Option<&HipModule> {
        self.modules.get(stem)
    }

    /// Lookup helper that errors with a useful message. Most call sites want
    /// this.
    pub(crate) fn expect_module(&self, stem: &'static str) -> Result<&HipModule> {
        self.modules
            .get(stem)
            .ok_or_else(|| anyhow!("kernel stem `{stem}` not loaded in OpsRegistry"))
    }

    /// HIP device id the registry was bound to; ops sanity-check the stream's
    /// device matches.
    pub fn device_id(&self) -> i32 {
        self.device_id
    }
}

// HipDevice / HipStream are re-exported at the top of this module so model
// crates don't have to depend on `flambeau-backend-hip` directly.
