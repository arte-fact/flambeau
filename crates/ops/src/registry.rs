//! Backend-generic kernel-module registry. `Registry<B>` pre-loads one
//! `B::Module` per kernel stem at session init (over the `core::KernelBackend`
//! launch seam), so every op call is one cached symbol lookup + launch. HIP and
//! CUDA share this; `hip::OpsRegistry` is the `Registry<HipKernels>` alias.

use std::collections::HashMap;

use flambeau_core::{Device, KernelBackend};

/// Kernel stems every model might touch — the same `extern "C"` stem names in
/// both the `.hsaco` (HIP) and `.cubin` (CUDA) catalogues. Loaded once in
/// [`Registry::new`]; a missing entry fails fast so model code never races an
/// unloaded module. Keep aligned with `kernels-{hip,cuda}/src/kernels/*`.
pub const KERNEL_STEMS: &[&str] = &[
    // Qmatmul — MMVQ + MMQ, all dtypes.
    "mmvq_q8_0",
    "mmvq_q8_0_dp4a",
    "mmvq_q8_0_dp4a_vdr2",
    "mmvq_q8_0_t128",
    "mmvq_q8_0_t128_vdr2",
    "mmvq_q8_0_gate_up_t128_vdr2",
    "mmvq_q8_0_llamacpp_style",
    "mmvq_q8_0_gate_up_dp4a",
    "mmvq_q4_0_gate_up_dp4a",
    "mmvq_q4_0_kv_f16dst_dp4a",
    "mmvq_q4_0_t128",
    "mmvq_q4_0_t128_dp4a",
    "mmvq_q4_0_t128_vdr2",
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
    "indexed_moe_mmq_q4_k_gate_up_lds_tiled",
    "indexed_moe_mmq_q4_k_down_lds_tiled",
    "indexed_moe_mmvq_q4_k_gate_up_r4_dp4a",
    "indexed_moe_mmvq_q4_k_gate_up_r4_sorted_dp4a",
    "mmvq_q2_k_dp4a",
    "mmvq_q2_k_r2_dp4a",
    "mmvq_q3_k_dp4a",
    "mmvq_q3_k_r2_dp4a",
    "mmvq_q4_k_r2",
    "mmvq_q4_k_r2_dp4a",
    "mmvq_q4_k_r4",
    "mmvq_q5_k",
    "mmvq_q5_k_r2",
    "mmvq_q5_k_dp4a",
    "mmvq_q5_k_r2_dp4a",
    "mmvq_q5_k_gate_up_dp4a",
    "mmvq_iq4_nl",
    "mmvq_iq4_nl_dp4a",
    "mmvq_iq4_nl_r2",
    "mmvq_iq4_xs",
    "mmvq_iq4_xs_dp4a",
    "mmvq_iq4_xs_r2",
    "mmvq_iq3_xxs",
    "mmvq_iq3_xxs_dp4a",
    "mmvq_iq3_xxs_r2",
    "mmvq_iq3_s",
    "mmvq_iq3_s_dp4a",
    "mmvq_iq3_s_r2",
    "mmvq_iq2_xxs",
    "mmvq_iq2_xxs_dp4a",
    "mmvq_iq2_xxs_r2",
    "mmvq_iq2_xs",
    "mmvq_iq2_xs_dp4a",
    "mmvq_iq2_xs_r2",
    "mmvq_iq2_s",
    "mmvq_iq2_s_dp4a",
    "mmvq_iq2_s_r2",
    "mmvq_iq1_s",
    "mmvq_iq1_s_dp4a",
    "mmvq_iq1_s_r2",
    "mmvq_iq1_m",
    "mmvq_iq1_m_dp4a",
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
    "indexed_moe_mmvq_iq4_xs_r2_dp4a",
    "indexed_moe_mmvq_iq4_nl_r2_dp4a",
    "indexed_moe_mmvq_iq3_xxs_r2_dp4a",
    "indexed_moe_mmvq_iq3_s_r2_dp4a",
    "indexed_moe_mmvq_iq2_xxs_r2_dp4a",
    "indexed_moe_mmvq_iq2_xs_r2_dp4a",
    "indexed_moe_mmvq_iq2_s_r2_dp4a",
    "indexed_moe_mmvq_iq1_s_r2_dp4a",
    "indexed_moe_mmvq_iq1_m_r2_dp4a",
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
    "mmq_q4_0_mma",
    "mmq_q5_0_wave64",
    "mmq_q5_1_wave64",
    "mmq_q4_K_4warp",
    "mmq_q4_K_lds_tiled",
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
    "gelu_mul_f32_to_q8_1",
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
    // Device-position decode-graph kernels (CUDA-only).
    "kv_append_v_unit_norm_f16_dpos",
    "embed_gather_f16_dpos",
    "softmax_masked_f16",
    // Attention.
    "attention_decode_f16",
    "attention_decode_f16_batched",
    "attention_decode_f16_paged",
    "kv_append_f16_batched_slots",
    "kv_append_f16_paged_slots",
    "kv_append_f16_paged_prefill",
    "attention_decode_f16_splitk",
    "attention_decode_f16_splitk_dpos",
    "attention_decode_batched_splitk",
    // "attention_decode_f16_splitk_h2",
    "attention_decode_q8_kv",
    "attention_decode_q8_kv_splitk",
    "attention_prefill_f16",
    "attention_prefill_f16_paged",
    "attention_prefill_flash_tile_f16",
    "attention_prefill_mma",
    "attention_prefill_q8_kv",
    "attention_prefill_flash_tile_q8_kv",
    // MoE.
    "topk_f32",
    "apply_per_expert_scale_f32",
    "indexed_moe_mmvq_q4_k",
    "indexed_moe_mmvq_q4_k_r2",
    "indexed_moe_mmvq_q4_k_gate_up",
    "indexed_moe_mmvq_q5_k",
    "indexed_moe_mmvq_q5_k_r2_dp4a",
    "indexed_moe_mmvq_q2_k",
    "indexed_moe_mmvq_q2_k_r2_dp4a",
    "indexed_moe_mmvq_q3_k",
    "indexed_moe_mmvq_q3_k_r2_dp4a",
    "indexed_moe_mmq_q2_k_down_tile8_dp4a",
    "indexed_moe_mmq_q2_k_gate_up_tile8_dp4a",
    "indexed_moe_mmq_q3_k_down_tile8_dp4a",
    "indexed_moe_mmq_q3_k_gate_up_tile8_dp4a",
    "indexed_moe_mmvq_q6_k",
    "indexed_moe_mmvq_q6_k_r2_dp4a",
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
    // GPU-side sampler kernels for the chat hot path.
    "sampler_topk_softmax_f32",
    "sampler_apply_penalties_f32",
    // Gemma4 — final logit softcap.
    "apply_softcap_f32",
    // Gemma4 — GELU-based FFN + per-layer side-channel.
    "gelu_f32_to_f16",
    "gelu_mul_f32",
];

/// Load failure for [`Registry::new`].
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("module load for `{stem}` failed: {source}")]
    Load {
        stem: &'static str,
        #[source]
        source: anyhow::Error,
    },
}

/// Single-session registry of loaded kernel modules for backend `B`. Built once
/// at model init, passed by shared reference into every op call.
pub struct Registry<B: KernelBackend> {
    device_id: i32,
    modules: HashMap<&'static str, B::Module>,
    skipped: Vec<&'static str>,
}

impl<B: KernelBackend> Registry<B> {
    /// Load each [`KERNEL_STEMS`] entry that `B`'s compiled catalogue actually
    /// contains into a fresh module on `dev`, binding `dev` first so loads land
    /// on the right device.
    ///
    /// `KERNEL_STEMS` is the union over every backend; each backend compiles a
    /// different subset (CUDA hasn't ported the legacy single-row oracles / the
    /// out-of-scope IQ quants, HIP has them all). A stem absent from this
    /// backend's catalogue is **skipped**, not an error — the load-present
    /// contract for N partial catalogues. Drift safety moves to first use:
    /// [`expect_module`](Self::expect_module) errors at launch, naming the stem,
    /// if a *dispatched* op asks for one that was skipped. Inspect the skip set
    /// with [`skipped`](Self::skipped).
    /// # Errors
    /// [`RegistryError::Load`] if a present stem fails to load on the device.
    pub fn new(dev: &B::Device) -> Result<Self, RegistryError> {
        B::bind(dev).map_err(|e| RegistryError::Load {
            stem: "<bind>",
            source: anyhow::anyhow!("{e:?}"),
        })?;
        let mut modules = HashMap::with_capacity(KERNEL_STEMS.len());
        let mut skipped = Vec::new();
        for &stem in KERNEL_STEMS {
            let Some(image) = B::image(stem) else {
                skipped.push(stem);
                continue;
            };
            let module = B::load(dev, image).map_err(|e| RegistryError::Load {
                stem,
                source: anyhow::anyhow!("{e:?}"),
            })?;
            modules.insert(stem, module);
        }
        Ok(Self {
            device_id: dev.id(),
            modules,
            skipped,
        })
    }

    /// Stems from [`KERNEL_STEMS`] absent from this backend's compiled catalogue
    /// and therefore not loaded. Empty on a backend that compiles every stem
    /// (HIP today). For CUDA it lists the unported legacy / out-of-scope kernels;
    /// a dispatched op hitting one of these fails loudly at launch.
    pub fn skipped(&self) -> &[&'static str] {
        &self.skipped
    }

    /// Look up a loaded module by kernel stem.
    pub fn module(&self, stem: &str) -> Option<&B::Module> {
        self.modules.get(stem)
    }

    /// Lookup helper that errors with a useful message. Most call sites want
    /// this.
    /// # Errors
    /// Errors if `stem` was not loaded (catalogue drift).
    pub fn expect_module(&self, stem: &'static str) -> anyhow::Result<&B::Module> {
        self.modules
            .get(stem)
            .ok_or_else(|| anyhow::anyhow!("kernel stem `{stem}` not loaded in registry"))
    }

    /// Device id the registry was bound to.
    pub fn device_id(&self) -> i32 {
        self.device_id
    }
}
