# Sweeps par ligne de dispatch (HEAD 247a91f8)

Le CLI écrit dans ./certs/<backend>/<arch>/ du répertoire courant (dossiers créés au besoin) : lancer depuis un dossier de travail vide, par exemple ~/Workspace/flambeau-exp/certs-run/.

## sm_86 — 20 / 20 lignes résolues
```
CUDA_VISIBLE_DEVICES=0 ~/Workspace/flambeau-exp/impl/bin_STAB_CUDA sweep --arch sm_86 --op qmatmul --dtype IQ1_M_dp4a   # qmatmul_iq1_m_mmvq_dp4a
CUDA_VISIBLE_DEVICES=0 ~/Workspace/flambeau-exp/impl/bin_STAB_CUDA sweep --arch sm_86 --op qmatmul --dtype IQ1_S_dp4a   # qmatmul_iq1_s_mmvq_dp4a
CUDA_VISIBLE_DEVICES=0 ~/Workspace/flambeau-exp/impl/bin_STAB_CUDA sweep --arch sm_86 --op qmatmul --dtype IQ2_S_dp4a   # qmatmul_iq2_s_mmvq_dp4a
CUDA_VISIBLE_DEVICES=0 ~/Workspace/flambeau-exp/impl/bin_STAB_CUDA sweep --arch sm_86 --op qmatmul --dtype IQ2_XS_dp4a   # qmatmul_iq2_xs_mmvq_dp4a
CUDA_VISIBLE_DEVICES=0 ~/Workspace/flambeau-exp/impl/bin_STAB_CUDA sweep --arch sm_86 --op qmatmul --dtype IQ2_XXS_dp4a   # qmatmul_iq2_xxs_mmvq_dp4a
CUDA_VISIBLE_DEVICES=0 ~/Workspace/flambeau-exp/impl/bin_STAB_CUDA sweep --arch sm_86 --op qmatmul --dtype IQ3_S_dp4a   # qmatmul_iq3_s_mmvq_dp4a
CUDA_VISIBLE_DEVICES=0 ~/Workspace/flambeau-exp/impl/bin_STAB_CUDA sweep --arch sm_86 --op qmatmul --dtype IQ3_XXS_dp4a   # qmatmul_iq3_xxs_mmvq_dp4a
CUDA_VISIBLE_DEVICES=0 ~/Workspace/flambeau-exp/impl/bin_STAB_CUDA sweep --arch sm_86 --op qmatmul --dtype IQ4_NL_dp4a   # qmatmul_iq4_nl_mmvq_dp4a
CUDA_VISIBLE_DEVICES=0 ~/Workspace/flambeau-exp/impl/bin_STAB_CUDA sweep --arch sm_86 --op qmatmul --dtype IQ4_XS_dp4a   # qmatmul_iq4_xs_mmvq_dp4a
CUDA_VISIBLE_DEVICES=0 ~/Workspace/flambeau-exp/impl/bin_STAB_CUDA sweep --arch sm_86 --op qmatmul --dtype Q2_K_dp4a   # qmatmul_q2_k_mmvq_dp4a
CUDA_VISIBLE_DEVICES=0 ~/Workspace/flambeau-exp/impl/bin_STAB_CUDA sweep --arch sm_86 --op qmatmul --dtype Q3_K_dp4a   # qmatmul_q3_k_mmvq_dp4a
CUDA_VISIBLE_DEVICES=0 ~/Workspace/flambeau-exp/impl/bin_STAB_CUDA sweep --arch sm_86 --op qmatmul --dtype Q4_0   # qmatmul_q4_0_mmvq_dp4a
CUDA_VISIBLE_DEVICES=0 ~/Workspace/flambeau-exp/impl/bin_STAB_CUDA sweep --arch sm_86 --op qmatmul --dtype Q4_1   # qmatmul_q4_1_mmvq_dp4a
CUDA_VISIBLE_DEVICES=0 ~/Workspace/flambeau-exp/impl/bin_STAB_CUDA sweep --arch sm_86 --op qmatmul --dtype Q4_K_r2_dp4a   # qmatmul_q4_k_mmvq_dp4a
CUDA_VISIBLE_DEVICES=0 ~/Workspace/flambeau-exp/impl/bin_STAB_CUDA sweep --arch sm_86 --op qmatmul --dtype Q5_0   # qmatmul_q5_0_mmvq_dp4a
CUDA_VISIBLE_DEVICES=0 ~/Workspace/flambeau-exp/impl/bin_STAB_CUDA sweep --arch sm_86 --op qmatmul --dtype Q5_1   # qmatmul_q5_1_mmvq_dp4a
CUDA_VISIBLE_DEVICES=0 ~/Workspace/flambeau-exp/impl/bin_STAB_CUDA sweep --arch sm_86 --op qmatmul --dtype Q5_K_dp4a   # qmatmul_q5_k_mmvq_dp4a
CUDA_VISIBLE_DEVICES=0 ~/Workspace/flambeau-exp/impl/bin_STAB_CUDA sweep --arch sm_86 --op qmatmul --dtype Q6_K_dp4a   # qmatmul_q6_k_mmvq_dp4a
CUDA_VISIBLE_DEVICES=0 ~/Workspace/flambeau-exp/impl/bin_STAB_CUDA sweep --arch sm_86 --op qmatmul --dtype Q8_0_dp4a   # qmatmul_q8_0_mmvq_dp4a
CUDA_VISIBLE_DEVICES=0 ~/Workspace/flambeau-exp/impl/bin_STAB_CUDA sweep --arch sm_86 --op qmatmul --dtype Q8_K   # qmatmul_q8_k_mmvq_single_row
```

## gfx906 — 26 / 88 lignes résolues
```
HIP_VISIBLE_DEVICES=1 ~/Workspace/flambeau-exp/impl/bin_STAB sweep --arch gfx906 --op attention_decode   # attention_decode_f16_gfx906
HIP_VISIBLE_DEVICES=1 ~/Workspace/flambeau-exp/impl/bin_STAB sweep --arch gfx906 --op attention_decode_splitk   # attention_decode_f16_splitk_gfx906
HIP_VISIBLE_DEVICES=1 ~/Workspace/flambeau-exp/impl/bin_STAB sweep --arch gfx906 --op attention_decode_q8_kv   # attention_decode_q8_kv_gfx906
HIP_VISIBLE_DEVICES=1 ~/Workspace/flambeau-exp/impl/bin_STAB sweep --arch gfx906 --op attention_prefill_flash_tile   # attention_prefill_f16_gfx906
HIP_VISIBLE_DEVICES=1 ~/Workspace/flambeau-exp/impl/bin_STAB sweep --arch gfx906 --op causal_conv1d   # causal_conv1d_f32_gfx906
HIP_VISIBLE_DEVICES=1 ~/Workspace/flambeau-exp/impl/bin_STAB sweep --arch gfx906 --op gdn_state_step_alphabeta   # gdn_state_step_alphabeta_f32_s128_gfx906
HIP_VISIBLE_DEVICES=1 ~/Workspace/flambeau-exp/impl/bin_STAB sweep --arch gfx906 --op qmatmul --dtype IQ1_M_dp4a   # qmatmul_iq1_m_mmvq_dp4a
HIP_VISIBLE_DEVICES=1 ~/Workspace/flambeau-exp/impl/bin_STAB sweep --arch gfx906 --op qmatmul --dtype IQ1_S_dp4a   # qmatmul_iq1_s_mmvq_dp4a
HIP_VISIBLE_DEVICES=1 ~/Workspace/flambeau-exp/impl/bin_STAB sweep --arch gfx906 --op qmatmul --dtype IQ2_S_dp4a   # qmatmul_iq2_s_mmvq_dp4a
HIP_VISIBLE_DEVICES=1 ~/Workspace/flambeau-exp/impl/bin_STAB sweep --arch gfx906 --op qmatmul --dtype IQ2_XS_dp4a   # qmatmul_iq2_xs_mmvq_dp4a
HIP_VISIBLE_DEVICES=1 ~/Workspace/flambeau-exp/impl/bin_STAB sweep --arch gfx906 --op qmatmul --dtype IQ2_XXS_dp4a   # qmatmul_iq2_xxs_mmvq_dp4a
HIP_VISIBLE_DEVICES=1 ~/Workspace/flambeau-exp/impl/bin_STAB sweep --arch gfx906 --op qmatmul --dtype IQ3_S_dp4a   # qmatmul_iq3_s_mmvq_dp4a
HIP_VISIBLE_DEVICES=1 ~/Workspace/flambeau-exp/impl/bin_STAB sweep --arch gfx906 --op qmatmul --dtype IQ3_XXS_dp4a   # qmatmul_iq3_xxs_mmvq_dp4a
HIP_VISIBLE_DEVICES=1 ~/Workspace/flambeau-exp/impl/bin_STAB sweep --arch gfx906 --op qmatmul --dtype IQ4_NL_dp4a   # qmatmul_iq4_nl_mmvq_dp4a
HIP_VISIBLE_DEVICES=1 ~/Workspace/flambeau-exp/impl/bin_STAB sweep --arch gfx906 --op qmatmul --dtype IQ4_XS_dp4a_t128_vdr4   # qmatmul_iq4_xs_mmvq_dp4a_t128_vdr4
HIP_VISIBLE_DEVICES=1 ~/Workspace/flambeau-exp/impl/bin_STAB sweep --arch gfx906 --op qmatmul --dtype Q2_K_r2_dp4a   # qmatmul_q2_K_mmvq_r2_dp4a_gfx906
HIP_VISIBLE_DEVICES=1 ~/Workspace/flambeau-exp/impl/bin_STAB sweep --arch gfx906 --op qmatmul --dtype Q3_K_r2_dp4a   # qmatmul_q3_K_mmvq_r2_dp4a_gfx906
HIP_VISIBLE_DEVICES=1 ~/Workspace/flambeau-exp/impl/bin_STAB sweep --arch gfx906 --op qmatmul --dtype Q4_1_t128   # qmatmul_q4_1_mmvq_t128_gfx906
HIP_VISIBLE_DEVICES=1 ~/Workspace/flambeau-exp/impl/bin_STAB sweep --arch gfx906 --op qmatmul --dtype Q4_K_dp4a_t128_vdr2   # qmatmul_q4_k_mmvq_dp4a_t128_vdr2
HIP_VISIBLE_DEVICES=1 ~/Workspace/flambeau-exp/impl/bin_STAB sweep --arch gfx906 --op qmatmul --dtype Q5_K_dp4a_t128_vdr2   # qmatmul_q5_k_mmvq_dp4a_t128_vdr2
HIP_VISIBLE_DEVICES=1 ~/Workspace/flambeau-exp/impl/bin_STAB sweep --arch gfx906 --op qmatmul --dtype Q6_K_dp4a   # qmatmul_q6_k_mmvq_dp4a
HIP_VISIBLE_DEVICES=1 ~/Workspace/flambeau-exp/impl/bin_STAB sweep --arch gfx906 --op qmatmul --dtype all   # qmatmul_q8_0_mmvq_single_row_gfx906
HIP_VISIBLE_DEVICES=1 ~/Workspace/flambeau-exp/impl/bin_STAB sweep --arch gfx906 --op qmatmul --dtype Q8_K   # qmatmul_q8_k_mmvq_single_row
HIP_VISIBLE_DEVICES=1 ~/Workspace/flambeau-exp/impl/bin_STAB sweep --arch gfx906 --op rmsnorm_q8_1   # rmsnorm_q8_1_fused_gfx906
HIP_VISIBLE_DEVICES=1 ~/Workspace/flambeau-exp/impl/bin_STAB sweep --arch gfx906 --op rope_neox_partial   # rope_neox_partial_f16_gfx906
HIP_VISIBLE_DEVICES=1 ~/Workspace/flambeau-exp/impl/bin_STAB sweep --arch gfx906 --op shared_expert_scale   # shared_expert_scale_f32_gfx906
```
Non résolues (62) : add_f16_gfx906, cast_f16_f32_gfx906, cast_f32_f16_gfx906, dense_gemv_f32_f16_gfx906, gdn_alpha_beta_f32_gfx906, gdn_state_step_f32_s128_gfx906, indexed_moe_mmq_q4_k_gfx906, indexed_moe_mmvq_q4_0_gfx906, indexed_moe_mmvq_q4_k_gate_up_gfx906, indexed_moe_mmvq_q4_k_r2_gfx906, indexed_moe_mmvq_q5_0_gfx906, indexed_moe_mmvq_q5_1_gfx906, indexed_moe_mmvq_q6_k_gfx906, indexed_moe_mmvq_q8_0_gfx906, l2_norm_f32_gfx906, mmq_f16_q8_1_gfx906, mmq_f16_tile_gfx906, mmvq_f16_q8_1_gfx906, mmvq_q4_0_gfx906, mmvq_q5_0_gfx906, mmvq_q5_1_gfx906, moe_combine_f16_gfx906, qmatmul_iq1_m_mmq_wave64_gfx906, qmatmul_iq1_s_mmq_wave64_gfx906, qmatmul_iq2_s_mmq_wave64_gfx906, qmatmul_iq2_xs_mmq_wave64_gfx906, qmatmul_iq2_xxs_mmq_wave64_gfx906, qmatmul_iq3_s_mmq_wave64_gfx906, qmatmul_iq3_xxs_mmq_wave64_gfx906, qmatmul_iq4_nl_mmq_lds_tiled_gfx906, qmatmul_iq4_nl_mmq_lds_tiled_x8_gfx906, qmatmul_iq4_xs_mmq_lds_tiled_gfx906, qmatmul_iq4_xs_mmq_lds_tiled_x8_gfx906, qmatmul_q2_K_mmq_wave64_gfx906, qmatmul_q3_K_mmq_lds_tiled_gfx906, qmatmul_q3_K_mmq_lds_tiled_x8_gfx906, qmatmul_q4_0_mmq_4warp_lds_gfx906, qmatmul_q4_0_mmq_wave64_gfx906, qmatmul_q4_1_mmq_4warp_lds_gfx906, qmatmul_q4_1_mmq_wave64_gfx906, qmatmul_q4_K_mmq_lds_tiled_y64_gfx906, qmatmul_q4_K_mmq_lds_tiled_y64_x8_gfx906, qmatmul_q5_0_mmq_wave64_gfx906, qmatmul_q5_1_mmq_wave64_gfx906, qmatmul_q5_K_mmq_lds_tiled_gfx906, qmatmul_q5_K_mmq_lds_tiled_x8_gfx906, qmatmul_q6_K_mmq_lds_tiled_gfx906, qmatmul_q6_K_mmq_lds_tiled_x8_gfx906, qmatmul_q8_0_mmq_lds_tiled_gfx906, qmatmul_q8_0_mmq_oracle_gfx906, qmatmul_q8_K_mmq_wave64_gfx906, quantize_f16_q8_1_gfx906, rmsnorm_f16_gfx906, rmsnorm_f32_gfx906, rope_f16_gfx906, scale_f32_gfx906, silu_f32_gfx906, softmax_masked_f16_gfx906, split_q_gate_f16_gfx906, swiglu_f16_gfx906, swiglu_f32_gfx906, topk_f32_gfx906

