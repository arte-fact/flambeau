//! `DirectCallKernel` catalogs for `sm_86`: every certified kernel that is not
//! a row of a `KernelDescriptor` table. Together with [`super::QMATMUL_SM86`]
//! they account for every cert under `certs/cuda/sm_86/`; the tests in
//! `super::tests` enforce both directions.

use flambeau_core::DirectCallKernel;

/// Kernels the runtime launches by name (`reg.expect_module("stem")`) rather
/// than through shape-based dispatch: one implementation per `(op, dtype)`.
pub const DIRECT_CALL_KERNELS_SM86: &[DirectCallKernel] = &[
    DirectCallKernel {
        impl_id: "add_f16",
        cert_rel_path: "certs/cuda/sm_86/add_f16.json",
    },
    DirectCallKernel {
        impl_id: "attention_decode_f16_batched_splitk_sm_86",
        cert_rel_path: "certs/cuda/sm_86/attention_decode_f16_batched_splitk_sm_86.json",
    },
    DirectCallKernel {
        impl_id: "attention_decode_f16_splitk_dpos_sm_86",
        cert_rel_path: "certs/cuda/sm_86/attention_decode_f16_splitk_dpos_sm_86.json",
    },
    DirectCallKernel {
        impl_id: "attention_decode_q8_kv_splitk_dpos_sm_86",
        cert_rel_path: "certs/cuda/sm_86/attention_decode_q8_kv_splitk_dpos_sm_86.json",
    },
    DirectCallKernel {
        impl_id: "cast_f16_f32",
        cert_rel_path: "certs/cuda/sm_86/cast_f16_f32.json",
    },
    DirectCallKernel {
        impl_id: "cast_f32_f16",
        cert_rel_path: "certs/cuda/sm_86/cast_f32_f16.json",
    },
    DirectCallKernel {
        impl_id: "embed_gather_f16_dpos_sm_86",
        cert_rel_path: "certs/cuda/sm_86/embed_gather_f16_dpos_sm_86.json",
    },
    DirectCallKernel {
        impl_id: "kv_append_v_unit_norm_f16_dpos_sm_86",
        cert_rel_path: "certs/cuda/sm_86/kv_append_v_unit_norm_f16_dpos_sm_86.json",
    },
    DirectCallKernel {
        impl_id: "l2_norm_f32",
        cert_rel_path: "certs/cuda/sm_86/l2_norm_f32.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_f16_mmq",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_f16_mmq.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_f16_mmq_tile",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_f16_mmq_tile.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_f16_mmvq",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_f16_mmvq.json",
    },
    DirectCallKernel {
        impl_id: "quantize_f16_q8_0_dpos_sm_86",
        cert_rel_path: "certs/cuda/sm_86/quantize_f16_q8_0_dpos_sm_86.json",
    },
    DirectCallKernel {
        impl_id: "rmsnorm_f32",
        cert_rel_path: "certs/cuda/sm_86/rmsnorm_f32.json",
    },
    DirectCallKernel {
        impl_id: "sampler_argmax_f32",
        cert_rel_path: "certs/cuda/sm_86/sampler_argmax_f32.json",
    },
    DirectCallKernel {
        impl_id: "scale_f32",
        cert_rel_path: "certs/cuda/sm_86/scale_f32.json",
    },
    DirectCallKernel {
        impl_id: "split_q_gate_f16",
        cert_rel_path: "certs/cuda/sm_86/split_q_gate_f16.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q4_k_gate_up_r4",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmvq_q4_k_gate_up_r4.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q4_k_r2",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmvq_q4_k_r2.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q4_k_r4",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmvq_q4_k_r4.json",
    },
];

/// Kernels no runtime path launches on sm_86, kept as A/B and cross-backend
/// references: a bench sweep (or autotune) is the only caller.
pub const BENCH_REFERENCE_KERNELS_SM86: &[DirectCallKernel] = &[
    DirectCallKernel {
        impl_id: "apply_per_expert_scale_f32",
        cert_rel_path: "certs/cuda/sm_86/apply_per_expert_scale_f32.json",
    },
    DirectCallKernel {
        impl_id: "apply_softcap_f32",
        cert_rel_path: "certs/cuda/sm_86/apply_softcap_f32.json",
    },
    DirectCallKernel {
        impl_id: "attention_decode_f16",
        cert_rel_path: "certs/cuda/sm_86/attention_decode_f16.json",
    },
    DirectCallKernel {
        impl_id: "attention_decode_f16_batched",
        cert_rel_path: "certs/cuda/sm_86/attention_decode_f16_batched.json",
    },
    DirectCallKernel {
        impl_id: "attention_decode_f16_paged",
        cert_rel_path: "certs/cuda/sm_86/attention_decode_f16_paged.json",
    },
    DirectCallKernel {
        impl_id: "attention_decode_f16_splitk",
        cert_rel_path: "certs/cuda/sm_86/attention_decode_f16_splitk.json",
    },
    DirectCallKernel {
        impl_id: "attention_decode_q8_kv",
        cert_rel_path: "certs/cuda/sm_86/attention_decode_q8_kv.json",
    },
    DirectCallKernel {
        impl_id: "attention_decode_q8_kv_splitk",
        cert_rel_path: "certs/cuda/sm_86/attention_decode_q8_kv_splitk.json",
    },
    DirectCallKernel {
        impl_id: "attention_prefill_f16",
        cert_rel_path: "certs/cuda/sm_86/attention_prefill_f16.json",
    },
    DirectCallKernel {
        impl_id: "attention_prefill_f16_paged",
        cert_rel_path: "certs/cuda/sm_86/attention_prefill_f16_paged.json",
    },
    DirectCallKernel {
        impl_id: "attention_prefill_flash_tile_f16",
        cert_rel_path: "certs/cuda/sm_86/attention_prefill_flash_tile_f16.json",
    },
    DirectCallKernel {
        impl_id: "attention_prefill_flash_tile_q8_kv",
        cert_rel_path: "certs/cuda/sm_86/attention_prefill_flash_tile_q8_kv.json",
    },
    DirectCallKernel {
        impl_id: "attention_prefill_q8_kv",
        cert_rel_path: "certs/cuda/sm_86/attention_prefill_q8_kv.json",
    },
    DirectCallKernel {
        impl_id: "causal_conv1d_f32",
        cert_rel_path: "certs/cuda/sm_86/causal_conv1d_f32.json",
    },
    DirectCallKernel {
        impl_id: "dense_gemv_f16_f16",
        cert_rel_path: "certs/cuda/sm_86/dense_gemv_f16_f16.json",
    },
    DirectCallKernel {
        impl_id: "dense_gemv_f16_f16_batched",
        cert_rel_path: "certs/cuda/sm_86/dense_gemv_f16_f16_batched.json",
    },
    DirectCallKernel {
        impl_id: "dense_gemv_f32_f16",
        cert_rel_path: "certs/cuda/sm_86/dense_gemv_f32_f16.json",
    },
    DirectCallKernel {
        impl_id: "dense_gemv_f32_f16_batched",
        cert_rel_path: "certs/cuda/sm_86/dense_gemv_f32_f16_batched.json",
    },
    DirectCallKernel {
        impl_id: "gdn_alpha_beta_f32",
        cert_rel_path: "certs/cuda/sm_86/gdn_alpha_beta_f32.json",
    },
    DirectCallKernel {
        impl_id: "gdn_assemble_conv_input_f32",
        cert_rel_path: "certs/cuda/sm_86/gdn_assemble_conv_input_f32.json",
    },
    DirectCallKernel {
        impl_id: "gdn_conv_trio_decode_f32_batched_slots",
        cert_rel_path: "certs/cuda/sm_86/gdn_conv_trio_decode_f32_batched_slots.json",
    },
    DirectCallKernel {
        impl_id: "gdn_split_qkv_f32",
        cert_rel_path: "certs/cuda/sm_86/gdn_split_qkv_f32.json",
    },
    DirectCallKernel {
        impl_id: "gdn_state_step_alphabeta_f32",
        cert_rel_path: "certs/cuda/sm_86/gdn_state_step_alphabeta_f32.json",
    },
    DirectCallKernel {
        impl_id: "gdn_state_step_alphabeta_f32_batched_slots",
        cert_rel_path: "certs/cuda/sm_86/gdn_state_step_alphabeta_f32_batched_slots.json",
    },
    DirectCallKernel {
        impl_id: "gdn_state_step_f32",
        cert_rel_path: "certs/cuda/sm_86/gdn_state_step_f32.json",
    },
    DirectCallKernel {
        impl_id: "gelu_f32_to_f16",
        cert_rel_path: "certs/cuda/sm_86/gelu_f32_to_f16.json",
    },
    DirectCallKernel {
        impl_id: "gelu_mul_f32",
        cert_rel_path: "certs/cuda/sm_86/gelu_mul_f32.json",
    },
    DirectCallKernel {
        impl_id: "gelu_mul_f32_to_q8_1",
        cert_rel_path: "certs/cuda/sm_86/gelu_mul_f32_to_q8_1.json",
    },
    DirectCallKernel {
        impl_id: "gelu_mul_f32_to_q8_1_mmq",
        cert_rel_path: "certs/cuda/sm_86/gelu_mul_f32_to_q8_1_mmq.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq1_m_down_tile8",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmq_iq1_m_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq1_m_gate_up_tile8",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmq_iq1_m_gate_up_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq1_s_down_tile8",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmq_iq1_s_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq1_s_gate_up_tile8",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmq_iq1_s_gate_up_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq2_s_down_tile8",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmq_iq2_s_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq2_s_gate_up_tile8",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmq_iq2_s_gate_up_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq2_xs_down_tile8",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmq_iq2_xs_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq2_xs_gate_up_tile8",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmq_iq2_xs_gate_up_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq2_xxs_down_tile8",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmq_iq2_xxs_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq2_xxs_gate_up_tile8",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmq_iq2_xxs_gate_up_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq3_s_down_tile8",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmq_iq3_s_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq3_s_gate_up_tile8",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmq_iq3_s_gate_up_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq3_xxs_down_tile8",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmq_iq3_xxs_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq3_xxs_gate_up_tile8",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmq_iq3_xxs_gate_up_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq4_nl_down_tile8",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmq_iq4_nl_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq4_nl_gate_up_tile8",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmq_iq4_nl_gate_up_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq4_xs_down_tile8",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmq_iq4_xs_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq4_xs_gate_up_tile8",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmq_iq4_xs_gate_up_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q3_k_down_tile8",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmq_q3_k_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q3_k_gate_up_tile8",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmq_q3_k_gate_up_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q4_0_down_tile8",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmq_q4_0_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q4_0_gate_up_tile8",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmq_q4_0_gate_up_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q4_1_down_tile8",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmq_q4_1_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q4_k_down_tile8",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmq_q4_k_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q4_k_gate_up_tile8",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmq_q4_k_gate_up_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q5_k_down_tile8",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmq_q5_k_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q5_k_gate_up_tile8",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmq_q5_k_gate_up_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q6_k_down_tile8",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmq_q6_k_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q6_k_gate_up_tile8",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmq_q6_k_gate_up_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q8_0_down_tile8",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmq_q8_0_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q8_0_gate_up_tile8",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmq_q8_0_gate_up_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_iq1_m_r2",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmvq_iq1_m_r2.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_iq1_s_r2",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmvq_iq1_s_r2.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_iq2_s_r2",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmvq_iq2_s_r2.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_iq2_xs_r2",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmvq_iq2_xs_r2.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_iq2_xxs_r2",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmvq_iq2_xxs_r2.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_iq3_s_r2",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmvq_iq3_s_r2.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_iq3_xxs_r2",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmvq_iq3_xxs_r2.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_iq4_nl_r2",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmvq_iq4_nl_r2.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_iq4_xs_r2",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmvq_iq4_xs_r2.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q3_k_r2",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmvq_q3_k_r2.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q4_0",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmvq_q4_0.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q4_0_gate_up",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmvq_q4_0_gate_up.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q4_1",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmvq_q4_1.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q5_k_r2",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmvq_q5_k_r2.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q6_k_r2",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmvq_q6_k_r2.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q8_0",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmvq_q8_0.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q8_0_gate_up",
        cert_rel_path: "certs/cuda/sm_86/indexed_moe_mmvq_q8_0_gate_up.json",
    },
    DirectCallKernel {
        impl_id: "kv_append_f16_batched_slots",
        cert_rel_path: "certs/cuda/sm_86/kv_append_f16_batched_slots.json",
    },
    DirectCallKernel {
        impl_id: "kv_append_f16_paged_prefill",
        cert_rel_path: "certs/cuda/sm_86/kv_append_f16_paged_prefill.json",
    },
    DirectCallKernel {
        impl_id: "kv_append_f16_paged_slots",
        cert_rel_path: "certs/cuda/sm_86/kv_append_f16_paged_slots.json",
    },
    DirectCallKernel {
        impl_id: "kv_append_v_unit_norm_f16",
        cert_rel_path: "certs/cuda/sm_86/kv_append_v_unit_norm_f16.json",
    },
    DirectCallKernel {
        impl_id: "mmvq_q4_0_kv_f16dst",
        cert_rel_path: "certs/cuda/sm_86/mmvq_q4_0_kv_f16dst.json",
    },
    DirectCallKernel {
        impl_id: "moe_combine_f16",
        cert_rel_path: "certs/cuda/sm_86/moe_combine_f16.json",
    },
    DirectCallKernel {
        impl_id: "moe_combine_no_residual_f16",
        cert_rel_path: "certs/cuda/sm_86/moe_combine_no_residual_f16.json",
    },
    DirectCallKernel {
        impl_id: "moe_combine_no_residual_f32",
        cert_rel_path: "certs/cuda/sm_86/moe_combine_no_residual_f32.json",
    },
    DirectCallKernel {
        impl_id: "moe_combine_two_residuals_f16",
        cert_rel_path: "certs/cuda/sm_86/moe_combine_two_residuals_f16.json",
    },
    DirectCallKernel {
        impl_id: "moe_sort_by_expert",
        cert_rel_path: "certs/cuda/sm_86/moe_sort_by_expert.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_iq1_m_mmq_wave64",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_iq1_m_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_iq1_s_mmq_wave64",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_iq1_s_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_iq2_s_mmq_wave64",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_iq2_s_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_iq2_s_mmvq_dp4a_row_tile_batched",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_iq2_s_mmvq_dp4a_row_tile_batched.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_iq2_xs_mmq_wave64",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_iq2_xs_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_iq2_xxs_mmq_wave64",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_iq2_xxs_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_iq3_s_mmq_wave64",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_iq3_s_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_iq3_s_mmvq_dp4a",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_iq3_s_mmvq_dp4a.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_iq3_s_mmvq_dp4a_row_tile_batched",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_iq3_s_mmvq_dp4a_row_tile_batched.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_iq3_xxs_mmq_wave64",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_iq3_xxs_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_iq3_xxs_mmvq_dp4a_row_tile_batched",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_iq3_xxs_mmvq_dp4a_row_tile_batched.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_iq4_nl_mmq_wave64",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_iq4_nl_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_iq4_nl_mmvq_dp4a",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_iq4_nl_mmvq_dp4a.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_iq4_xs_mmq_wave64",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_iq4_xs_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_iq4_xs_mmvq_dp4a",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_iq4_xs_mmvq_dp4a.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_iq4_xs_mmvq_dp4a_row_tile_batched",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_iq4_xs_mmvq_dp4a_row_tile_batched.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q2_k_mmq_wave64",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q2_k_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q3_k_mmq_wave64",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q3_k_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q3_k_mmvq_dp4a",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q3_k_mmvq_dp4a.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q4_0_gate_up_row_tile_mmvq_batched",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q4_0_gate_up_row_tile_mmvq_batched.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q4_0_gate_up_t128_dp4a",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q4_0_gate_up_t128_dp4a.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q4_0_mmq_4warp_lds",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q4_0_mmq_4warp_lds.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q4_0_mmq_wave64",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q4_0_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q4_0_mmvq_batched",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q4_0_mmvq_batched.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q4_0_mmvq_row_tile_batched",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q4_0_mmvq_row_tile_batched.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q4_0_mmvq_t128_dp4a",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q4_0_mmvq_t128_dp4a.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q4_0_mmvq_t128_vdr2",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q4_0_mmvq_t128_vdr2.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q4_1_mmq_4warp_lds",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q4_1_mmq_4warp_lds.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q4_1_mmq_wave64",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q4_1_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q4_1_mmvq_batched",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q4_1_mmvq_batched.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q4_k_mmq_lds_tiled",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q4_k_mmq_lds_tiled.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q4_k_mmq_wave64",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q4_k_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q4_k_mmvq_dp4a",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q4_k_mmvq_dp4a.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q4_k_mmvq_dp4a_row_tile_batched",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q4_k_mmvq_dp4a_row_tile_batched.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q5_0_mmq_wave64",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q5_0_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q5_1_mmq_wave64",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q5_1_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q5_k_gate_up_dp4a",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q5_k_gate_up_dp4a.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q5_k_mmq_wave64",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q5_k_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q5_k_mmvq_dp4a",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q5_k_mmvq_dp4a.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q5_k_mmvq_dp4a_row_tile_batched",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q5_k_mmvq_dp4a_row_tile_batched.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q5_k_r2_mmvq_batched",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q5_k_r2_mmvq_batched.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q6_k_mmq_wave64",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q6_k_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q6_k_mmvq_dp4a_row_tile_batched",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q6_k_mmvq_dp4a_row_tile_batched.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q8_0_gate_up_t128_vdr2",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q8_0_gate_up_t128_vdr2.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q8_0_mmq_oracle",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q8_0_mmq_oracle.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q8_0_mmq_wave64_tile16",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q8_0_mmq_wave64_tile16.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q8_0_mmvq_batched",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q8_0_mmvq_batched.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q8_0_mmvq_r2",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q8_0_mmvq_r2.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q8_0_mmvq_r4",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q8_0_mmvq_r4.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q8_0_mmvq_row_tile_batched",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q8_0_mmvq_row_tile_batched.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q8_0_mmvq_t128",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q8_0_mmvq_t128.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q8_0_mmvq_t128_vdr2",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q8_0_mmvq_t128_vdr2.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q8_k_mmq_wave64",
        cert_rel_path: "certs/cuda/sm_86/qmatmul_q8_k_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "quantize_f16_q8_0",
        cert_rel_path: "certs/cuda/sm_86/quantize_f16_q8_0.json",
    },
    DirectCallKernel {
        impl_id: "quantize_f16_q8_1",
        cert_rel_path: "certs/cuda/sm_86/quantize_f16_q8_1.json",
    },
    DirectCallKernel {
        impl_id: "quantize_f16_q8_1_mmq",
        cert_rel_path: "certs/cuda/sm_86/quantize_f16_q8_1_mmq.json",
    },
    DirectCallKernel {
        impl_id: "rmsnorm_f16",
        cert_rel_path: "certs/cuda/sm_86/rmsnorm_f16.json",
    },
    DirectCallKernel {
        impl_id: "rmsnorm_f16_add_residual",
        cert_rel_path: "certs/cuda/sm_86/rmsnorm_f16_add_residual.json",
    },
    DirectCallKernel {
        impl_id: "rmsnorm_f16_to_f16_add_residual",
        cert_rel_path: "certs/cuda/sm_86/rmsnorm_f16_to_f16_add_residual.json",
    },
    DirectCallKernel {
        impl_id: "rmsnorm_f32_to_f16",
        cert_rel_path: "certs/cuda/sm_86/rmsnorm_f32_to_f16.json",
    },
    DirectCallKernel {
        impl_id: "rmsnorm_f32_to_f16_add_residual",
        cert_rel_path: "certs/cuda/sm_86/rmsnorm_f32_to_f16_add_residual.json",
    },
    DirectCallKernel {
        impl_id: "rmsnorm_q8_1_fused",
        cert_rel_path: "certs/cuda/sm_86/rmsnorm_q8_1_fused.json",
    },
    DirectCallKernel {
        impl_id: "rmsnorm_rope_neox_partial_f16",
        cert_rel_path: "certs/cuda/sm_86/rmsnorm_rope_neox_partial_f16.json",
    },
    DirectCallKernel {
        impl_id: "rope_f16",
        cert_rel_path: "certs/cuda/sm_86/rope_f16.json",
    },
    DirectCallKernel {
        impl_id: "rope_neox_partial_f16",
        cert_rel_path: "certs/cuda/sm_86/rope_neox_partial_f16.json",
    },
    DirectCallKernel {
        impl_id: "sampler_apply_penalties_f32",
        cert_rel_path: "certs/cuda/sm_86/sampler_apply_penalties_f32.json",
    },
    DirectCallKernel {
        impl_id: "sampler_topk_softmax_f32",
        cert_rel_path: "certs/cuda/sm_86/sampler_topk_softmax_f32.json",
    },
    DirectCallKernel {
        impl_id: "scale_f16",
        cert_rel_path: "certs/cuda/sm_86/scale_f16.json",
    },
    DirectCallKernel {
        impl_id: "shared_expert_scale_f32",
        cert_rel_path: "certs/cuda/sm_86/shared_expert_scale_f32.json",
    },
    DirectCallKernel {
        impl_id: "sigmoid_mul_f16",
        cert_rel_path: "certs/cuda/sm_86/sigmoid_mul_f16.json",
    },
    DirectCallKernel {
        impl_id: "silu_f32",
        cert_rel_path: "certs/cuda/sm_86/silu_f32.json",
    },
    DirectCallKernel {
        impl_id: "softmax_masked_f16",
        cert_rel_path: "certs/cuda/sm_86/softmax_masked_f16.json",
    },
    DirectCallKernel {
        impl_id: "swiglu_f16",
        cert_rel_path: "certs/cuda/sm_86/swiglu_f16.json",
    },
    DirectCallKernel {
        impl_id: "swiglu_f32",
        cert_rel_path: "certs/cuda/sm_86/swiglu_f32.json",
    },
    DirectCallKernel {
        impl_id: "swiglu_f32_to_f16",
        cert_rel_path: "certs/cuda/sm_86/swiglu_f32_to_f16.json",
    },
    DirectCallKernel {
        impl_id: "swiglu_f32_to_q8_1",
        cert_rel_path: "certs/cuda/sm_86/swiglu_f32_to_q8_1.json",
    },
    DirectCallKernel {
        impl_id: "topk_f32",
        cert_rel_path: "certs/cuda/sm_86/topk_f32.json",
    },
    DirectCallKernel {
        impl_id: "v_unit_norm_per_head_f16",
        cert_rel_path: "certs/cuda/sm_86/v_unit_norm_per_head_f16.json",
    },
];
