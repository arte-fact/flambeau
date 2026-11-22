# r5 — chemin de décodage CUDA de bin_O2_CUDA, 27B sans MTP (lecture du code, a45374a0)

Configuration lue : `rtx/run_rtx_fl.sh` avec `serve --backend cuda --devices cuda:0 --kv q8 --inflight-slots 1 --mtp 0`.
Modèle : `Qwen3.8-27B-UD-Q4_K_M.gguf`, arch `qwen35`. Il a 65 blocs : les 64 couches principales (48 GDN et 16 attention pleine, aux couches 3, 7, …, 63) plus la couche MTP 64, ignorée avec `--mtp 0`.
Dimensions : hidden 5120, FFN 17408, vocab 248320. Attention : 24 têtes Q, 4 têtes KV, head_dim 256, Q à porte (attn_q de 12288 lignes). GDN : 16 têtes K × 128, 48 têtes V × 128, d_inner 6144, conv_channels 10240, noyau de convolution 4.

## 0. Points structurants (à vérifier sur la trace)
- **Pas de graphe CUDA pour qwen35.**
  - La passe de capture (`solo.rs::forward_decode_graph`) n'est tentée qu'à partir de la position 256 (`DECODE_GRAPH_MIN_KV` = 257). Avant, le chemin est `forward_general`.
  - La tentative échoue toujours. En KV Q8, l'append de qwen35 (pas de V unit-norm) appelle `mark_uncapturable("kv_append: Q8 append not captured")`, avec `RetryAt::Never` (standard_attn.rs:668). En KV F16, la branche équivalente refuse aussi (« plain F16 append », standard_attn.rs:698).
  - Les variantes `_dpos` (`kv_append_f16_to_q8_dpos`, `attention_decode_q8_kv_splitk_dpos`) existent, mais seul le chemin gemma4 (V unit-norm) les utilise pour l'append.
  - Conséquence : **tous les tokens passent par `forward_general`**, en lancements individuels.
  - Coût unique par requête qui atteint la position 257 : une capture (enregistrement de la passe, rien n'est exécuté) puis l'instanciation du graphe, qui est jeté. Ensuite `graph_block` bloque les nouvelles tentatives pour ce slot.
- **Pas de MTP** : `--mtp 0` donne `config.mtp = None`, donc `mtp_anchor_f16` est NULL et `forward()` saute `mtp_stage_hidden` / `mtp_catchup`.
- **Pas de fusion AR** : en solo (`NoopHooks`), `supports_ar_residual_*` est faux. Chaque projection de sortie écrit en F32, puis passe par `cast_f32_f16` et `add_f16`. Le chemin « F16 direct » n'est utilisé que pour K et V.
- **MMVQ** : `qmatmul` à m = 1 prend la ligne de `dispatch/cuda/sm_86.toml` (QMATMUL_SM86, toutes en `m_range (1, MAX)`) ; `Ops::mmvq` (GDN) passe par la même résolution m = 1, mise en cache. La grille vaut ceil(n / rows_per_block) et le bloc vaut `threads` (launch/qmatmul.rs:2297).

| dtype | impl_id (ligne sm_86) | stem / entrée | bloc | lignes/bloc |
|---|---|---|---|---|
| Q4_K | qmatmul_q4_k_mmvq_dp4a | mmvq_q4_k_r2_dp4a / flambeau_mmvq_q4_k_r2_dp4a_q8_1 | 64 | 2 |
| Q5_K | qmatmul_q5_k_mmvq_dp4a | mmvq_q5_k_dp4a | 256 | 1 |
| Q6_K | qmatmul_q6_k_mmvq_dp4a | mmvq_q6_k_dp4a | 64 | 1 |
| Q8_0 | qmatmul_q8_0_mmvq_dp4a | mmvq_q8_0_dp4a | 256 | 1 |
| Q3_K | qmatmul_q3_k_mmvq_dp4a | mmvq_q3_k_dp4a | 256 | 1 |
| IQ4_XS | qmatmul_iq4_xs_mmvq_dp4a | mmvq_iq4_xs_dp4a | 256 | 1 |
| IQ4_NL | qmatmul_iq4_nl_mmvq_dp4a | mmvq_iq4_nl_dp4a | 256 | 1 |
| IQ3_S | qmatmul_iq3_s_mmvq_dp4a | mmvq_iq3_s_dp4a | 256 | 1 |

L'entrée `_f16` (sortie F16 saturée) a la même grille ; seul le nom d'entrée change (suffixe `_f16`).

Catalogue sm_86 (colonne « cat. » des tableaux) :
- **DC** = `DIRECT_CALL_KERNELS_SM86` ;
- **BR** = `BENCH_REFERENCE_KERNELS_SM86`.

Plusieurs noyaux lancés à chaque token sont classés BR (rmsnorm_q8_1_fused, quantize_*, attention_decode_q8_kv*, sigmoid_mul_f16, swiglu_*, silu_f32, gdn_*, causal_conv1d_f32). C'est la même mauvaise classification que celle relevée sur gfx906 (E4-audit, famille C).

## 1. Entrée (une fois par token)
| # | opération | noyau / copie | grille × bloc | impl_id (cat.) |
|---|---|---|---|---|
| E1 | embedding (token_embd Q4_K dans le GGUF, table F16 en VRAM, une ligne copiée) | `cuMemcpyDtoDAsync` 10 240 o (une ligne F16) | — | — |

(`embed_gather_f16_dpos` ne sert qu'en capture.)

## 2. Couche GDN (×48), `gdn.rs` → `DeltaNetLayer::forward_decode_with_ar_hook` (delta_net.rs:646)
| # | noyau (entrée) | poids | grille × bloc | impl_id (cat.) |
|---|---|---|---|---|
| G1 | flambeau_rmsnorm_q8_1_fused (attn_norm + quant Q8_1, k = 5120) | F32 | 1 × 1024 | rmsnorm_q8_1_fused (BR) |
| G2 | MMVQ attn_qkv, n = 10240, k = 5120 | Q4_K 29, Q5_K 12, IQ4_XS 6, IQ4_NL 1 | Q4_K 5120 × 64 ; autres 10240 × 256 | ligne sm_86 du dtype |
| G3 | MMVQ attn_gate (z), n = 6144, k = 5120 | Q4_K 24, Q5_K 12, IQ4_XS 12 | Q4_K 3072 × 64 ; autres 6144 × 256 | ligne sm_86 du dtype |
| G4 | flambeau_mmvq_q8_0_gate_up_t128_vdr2_q8_1 (alpha et beta fusionnés, 48 + 48 lignes) | Q8_0 ×48 | 48 × 128 | qmatmul_q8_0_gate_up_t128_vdr2 (BR) |
| G5 | flambeau_gdn_assemble_conv_input_f32 (40 960 éléments) | — | 160 × 256 | gdn_assemble_conv_input_f32 (BR) |
| G6 | flambeau_causal_conv1d_f32 | ssm_conv1d F32 | (40, 1) × 256 | causal_conv1d_f32 (BR) |
| — | historique de convolution | `cuMemcpyDtoDAsync` 122 880 o (3 × 10240 F32) | — | — |
| G7 | flambeau_silu_f32 (10240) | — | 40 × 256 | silu_f32 (BR) |
| G8 | flambeau_l2_norm_f32 Q (16 lignes de 128) | — | 16 × 256 | l2_norm_f32 (DC) |
| G9 | flambeau_l2_norm_f32 K | — | 16 × 256 | l2_norm_f32 (DC) |
| G10 | flambeau_scale_f32 Q (2048) | — | 8 × 256 | scale_f32 (DC) |
| G11 | flambeau_gdn_state_step_alphabeta_f32_s128 (h_v = 48, état 48 × 128 × 128 F32) | ssm_a, dt_bias F32 | (48, 1, 32) × (32, 4) | gdn_state_step_alphabeta_f32 (BR) |
| G12 | flambeau_rmsnorm_f32 (ssm_norm, 48 lignes de 128) | F32 | 48 × 256 | rmsnorm_f32 (DC) |
| G13 | flambeau_swiglu_f32_to_q8_1 (z · out, quant, 6144) | — | 192 × 32 | swiglu_f32_to_q8_1 (BR) |
| G14 | MMVQ ssm_out, n = 5120, k = 6144 | Q5_K 33, Q6_K 5, IQ4_XS 5, Q4_K 5 | Q4_K 2560 × 64 ; Q6_K 5120 × 64 ; autres 5120 × 256 | ligne sm_86 du dtype |
| G15 | flambeau_cast_f32_f16 (5120) | — | 20 × 256 | cast_f32_f16 (DC) |
| G16 | flambeau_add_f16 (résiduel, 5120) | — | 20 × 256 | add_f16 (DC) |

Par couche GDN : 16 noyaux + 1 copie DtoD.

## 3. Couche d'attention (×16), `standard_attn.rs` (n = 1, slot unique, KV Q8 contigu)
| # | noyau (entrée) | poids | grille × bloc | impl_id (cat.) |
|---|---|---|---|---|
| A1 | flambeau_rmsnorm_q8_1_fused (attn_norm) | F32 | 1 × 1024 | rmsnorm_q8_1_fused (BR) |
| A2 | MMVQ attn_q à porte, n = 12288 (Q et porte), k = 5120, sortie F32 | IQ4_XS 8, Q4_K 6, Q5_K 2 | Q4_K 6144 × 64 ; autres 12288 × 256 | ligne sm_86 du dtype |
| A3 | flambeau_cast_f32_f16 (12288) | — | 48 × 256 | cast_f32_f16 (DC) |
| A4 | flambeau_split_q_gate_f16 | — | (1, 24, 2) × 128 | split_q_gate_f16 (DC) |
| A5 | MMVQ attn_k, sortie F16 directe (entrée `_f16`), n = 1024 | Q6_K 9, Q5_K 5, Q4_K 1, Q8_0 1 | Q6_K 1024 × 64 ; Q4_K 512 × 64 ; autres 1024 × 256 | ligne sm_86 du dtype |
| A6 | MMVQ attn_v, F16 direct, n = 1024 | Q8_0 7, Q5_K 6, Q6_K 3 | idem A5 | ligne sm_86 du dtype |
| — | positions | `cuMemcpyHtoDAsync` 4 o depuis un `Vec<i32>` (mémoire paginée) | — | — |
| A7 | flambeau_rmsnorm_rope_neox_partial_f16_d256 Q (q_norm + RoPE partielle 64) | F32 | (1, 24, 1) × 256 | rmsnorm_rope_neox_partial_f16 (BR) |
| A8 | idem K | F32 | (1, 4, 1) × 256 | idem |
| A9 | flambeau_quantize_row_f16_q8_0 (append K dans le cache Q8, 1024) | — | 32 × 32 | quantize_f16_q8_0 (BR) |
| A10 | idem V | — | 32 × 32 | idem |
| A11a | si n_kv > 32 : flambeau_attention_decode_q8_kv_splitk_chunk | — | (24, n_chunks) × 64 | attention_decode_q8_kv_splitk (BR) |
| A11b | puis flambeau_attention_decode_q8_kv_splitk_combine | — | 24 × 256 | idem |
| A11' | si n_kv ≤ 32 : flambeau_attention_decode_q8_kv (une passe) | — | 24 × 64 | attention_decode_q8_kv (BR) |
| A12 | flambeau_sigmoid_mul_f16 (porte, 6144) | — | 24 × 256 | sigmoid_mul_f16 (BR) |
| A13 | flambeau_quantize_row_f16_q8_1 (6144) | — | 192 × 32 | quantize_f16_q8_1 (BR) |
| A14 | MMVQ attn_output, n = 5120, k = 6144, F32 | Q5_K 11, Q6_K 4, IQ4_XS 1 | Q6_K 5120 × 64 ; autres 5120 × 256 | ligne sm_86 du dtype |
| A15 | flambeau_cast_f32_f16 (5120) | — | 20 × 256 | cast_f32_f16 (DC) |
| A16 | flambeau_add_f16 (résiduel) | — | 20 × 256 | add_f16 (DC) |

Découpage de l'attention en blocs : chunk_size = n_kv si n_kv ≤ 32, sinon max(ceil(n_kv / 32), 32) arrondi au multiple de 4 (`launch/attention.rs:1086`). Donc n_chunks = ceil(n_kv / 32) jusqu'à n_kv = 1024, puis 32 au-delà. Le contexte est plafonné à 8192 par `--ctx-cap`.

Par couche d'attention : 16 noyaux (17 en split-K) + 1 copie HtoD de 4 o.

## 4. FFN dense (après chacune des 64 couches), `dense_ffn.rs`
| # | noyau (entrée) | poids | grille × bloc | impl_id (cat.) |
|---|---|---|---|---|
| F1 | flambeau_rmsnorm_q8_1_fused (post_attention_norm) | F32 | 1 × 1024 | rmsnorm_q8_1_fused (BR) |
| F2 | MMVQ ffn_gate, n = 17408, k = 5120 | IQ4_XS 31, Q4_K 14, Q5_K 12, Q3_K 4, IQ4_NL 2, IQ3_S 1 | Q4_K 8704 × 64 ; autres 17408 × 256 | ligne sm_86 du dtype |
| F3 | MMVQ ffn_up, n = 17408, k = 5120 | IQ4_XS 30, Q5_K 16, Q4_K 14, Q3_K 2, IQ4_NL 1, Q6_K 1 | Q4_K 8704 × 64 ; Q6_K 17408 × 64 ; autres 17408 × 256 | ligne sm_86 du dtype |
| F4 | flambeau_swiglu_f32_to_f16 (17408) | — | 68 × 256 | swiglu_f32_to_f16 (BR) |
| F5 | flambeau_quantize_row_f16_q8_1 (17408) | — | 544 × 32 | quantize_f16_q8_1 (BR) |
| F6 | MMVQ ffn_down, n = 5120, k = 17408 | IQ4_XS 24, Q5_K 22, Q4_K 10, IQ4_NL 3, IQ3_S 3, Q3_K 1, Q6_K 1 | Q4_K 2560 × 64 ; Q6_K 5120 × 64 ; autres 5120 × 256 | ligne sm_86 du dtype |
| F7 | flambeau_cast_f32_f16 (5120) | — | 20 × 256 | cast_f32_f16 (DC) |
| F8 | flambeau_add_f16 (résiduel) | — | 20 × 256 | add_f16 (DC) |

gate et up sont deux MMVQ séparés. La fusion n'existe que pour les paires Q4_0/Q4_0 et Q8_0/Q8_0, absentes de ce GGUF.

## 5. Tête (une fois par token), `output_head.rs`
| # | noyau | poids | grille × bloc | impl_id (cat.) |
|---|---|---|---|---|
| H1 | flambeau_rmsnorm_q8_1_fused (output_norm) | F32 | 1 × 1024 | rmsnorm_q8_1_fused (BR) |
| H2 | MMVQ output, n = 248320, k = 5120, F32 | Q6_K (`mmvq_q4_k_r4` n'est pris que pour Q4_K) | 248320 × 64 | qmatmul_q6_k_mmvq_dp4a |
| — | logits | `cuMemcpyDtoHAsync` 993 280 o vers `Vec<f32>` (mémoire paginée), puis `cuStreamSynchronize` | — | — |

Après la synchronisation, le côté hôte fait deux copies des logits (`ctx.logits().to_vec()` puis `out.extend_from_slice`) et l'échantillonnage. Le serveur ne fait aucun autre appel GPU par token (aucun memcpy ni synchronize dans `crates/server`).

## 6. Bilan par token (n_kv > 32, régime split-K)
- **Noyaux** : 48 × 16 (GDN) + 16 × 17 (attention) + 64 × 8 (FFN) + 2 (tête) = **1554** lancements (1538 si n_kv ≤ 32). Dont MMVQ : 48 × 4 + 16 × 4 + 64 × 3 + 1 = 449.
- **Copies** :
  - 1 DtoD de 10 240 o (embedding) ;
  - 48 DtoD de 122 880 o (historique de convolution) ;
  - 16 HtoD de 4 o (positions, depuis de la mémoire paginée) ;
  - 1 DtoH de 993 280 o (logits, vers de la mémoire paginée).
- **Synchronisations** :
  - 1 `cuStreamSynchronize` explicite (après les logits) ;
  - un DtoH vers de la mémoire paginée est bloquant pour l'hôte ;
  - les 16 HtoD paginés de 4 o peuvent provoquer une attente implicite du flux selon le pilote (à lire dans `cuda_api_sum` : durée de `cuMemcpyHtoDAsync`).
- **Graphe CUDA** : aucun noyau ne passe par un graphe (voir §0).
- **Une fois par requête** (position 257) : `cuStreamBeginCapture`, `cuStreamEndCapture`, `cuGraphInstantiate`, puis destruction du graphe.
