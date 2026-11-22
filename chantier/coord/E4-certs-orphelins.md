# Certificats orphelins — tri (HEAD 7a260d46, lecture seule, rien supprimé)

Orphelin = fichier de `certs/<backend>/<arch>/` qu'aucune ligne de `dispatch/*/<arch>.toml` ne référence.
« Régénérable » : gfx906 vérifié sur GPU (les 376 invocations de la CLI du 20/11 sur ececc040 ont réécrit ces fichiers) ; sm_86 par lecture (le nom est écrit par un sweep atteignable via `sweep --arch sm_86`).

| arch | fichiers | dispatchés | orphelins |
|---|---|---|---|
| gfx906 | 324 | 88 | **236** (244 hier − 7 jumeaux neutres désormais dispatchés par h2 − 1 Q4_K t256 retiré par a0f7301f) |
| sm_86 | 241 | 20 | **221** (222 hier : 220 non réécrits + 2 Q4_0 t128 réécrits par la passe ; − 1 t256) |

## gfx906 — 236, tous en (a) : enregistrés ET régénérables
Tests qui les exigent : `every_cert_on_disk_is_registered` (tout fichier doit être enregistré) et `every_descriptor_cert_path_exists_on_disk` (tout enregistrement doit avoir son fichier). Supprimer un certificat oblige à retirer son enregistrement, et inversement.

- (a1) `DIRECT_CALL_KERNELS_GFX906`, **24** : noyaux que le runtime lance directement, sans ligne de dispatch (choix par dtype). Ex. : `indexed_moe_mmq_q4_k_down_tile8_gfx906`, `indexed_moe_mmvq_q5_k_gfx906`, `attention_decode_q8_kv_splitk_gfx906`. **Garder** : ils sont utilisés à l'exécution.
- (a2) descripteur `KernelDescriptor` à `m_range (MAX, MAX)` (inactif), **12** : variantes remplacées gardées comme référence. Ex. : `qmatmul_q4_K_mmq_wave64_gfx906`, `qmatmul_q5_K_mmvq_nw1_r2_gfx906`, `qmatmul_q8_0_mmq_wave64_tile16_gfx906`. **Proposition** : supprimer certificat + descripteur + recette + Dtype de sweep, comme pour t256 (a0f7301f) ; supprimer le noyau seulement s'il ne sert plus ailleurs (3 d'entre eux — `qmatmul_q4_k_mmvq_dp4a`, `q5_k_mmvq_dp4a`, `iq4_xs_mmvq_dp4a` — sont dispatchés sur sm_86 et à garder).
- (a3) `BENCH_REFERENCE_KERNELS_GFX906`, **200** :
  - jumeaux multi-backend à nom neutre (`add_f16`, `attention_decode_f16`, `rmsnorm_f16`…), base de la comparaison bit à bit avec sm_86 : ~70. **Garder.**
  - variantes MMQ MoE tile8 / lds_tiled (`indexed_moe_mmq_q5_0_gate_up_tile8_gfx906`…) : 40 ; MMVQ MoE : 26 ; MMVQ par lots et gate+up : 22. Chemins MoE et par lots, à confronter aux appels du runtime avant toute décision.
  - oracles une ligne / r2 (`qmatmul_*_single_row_gfx906`, `*_nw1_r2_gfx906`) : CLAUDE.md les garde comme oracles HIP. **Garder.**
  - variantes MMQ non dispatchées (`qmatmul_q4_K_mmq_4warp_lds_gfx906`, `*_wave64_gfx906` dont le dtype dispatche `lds_tiled`…) : 32. **Proposition** : audit famille par famille (le runtime les lance-t-il hors dispatch ?) puis, pour celles qu'aucun chemin n'atteint, supprimer certificat + enregistrement + Dtype, et le noyau s'il est nul (règle 10).

## sm_86 — 221, aucun enregistrement exigé par un test (backend-cuda n'a ni catalogue direct-call ni test de réciprocité)
- (b) régénérables, non enregistrés, **201** : jumeaux multi-backend (`add_f16`, `attention_decode_f16_batched`…), MMQ/MMVQ MoE et par lots CUDA, variantes MMVQ dp4a hors dispatch, plus 6 ops propres à CUDA (`attention_decode_f16_splitk_dpos_sm_86`, `embed_gather_f16_dpos_sm_86`, `quantize_f16_q8_0_dpos_sm_86`…). **Proposition** : ajouter à backend-cuda le même catalogue + test de réciprocité que gfx906 (sinon rien n'empêche la dérive), et appliquer ensuite le même tri que (a3).
- (d) noms HIP dans le dossier CUDA, **18** : `qmatmul_iq1_m_mmvq_single_row_gfx906`, `qmatmul_iq2_s_mmvq_nw1_r2_gfx906`, `qmatmul_iq4_xs_mmvq_single_row_gfx906`… écrits par les variantes « dtype complet » de `sweep --arch sm_86 --dtype IQ*` qui lancent aussi les oracles HIP. CLAUDE.md : ces oracles restent HIP. **Proposition** : supprimer le certificat seul, et retirer ces Dtype des listes CUDA de la CLI pour qu'ils ne reviennent pas.
- (c) non régénérables, **2**, à ne PAS supprimer : ce sont des trous de la règle 2 :
  - `qmatmul_q4_0_mma` : descripteur ACTIF dans `QMATMUL_SM86` (m ≥ 128, MMQ Q4_0 en tensor core), choisi à l'exécution par `first_match`, mais sans ligne dans `dispatch/cuda/sm_86.toml` (donc hors cert-check) et sans sweep qui écrive ce nom.
  - `attention_prefill_mma` : lancé directement par `ops/launch/attention.rs:1532` (prefill CUDA d128/256/512, n ≥ 16), sans sweep ni enregistrement.
  **Proposition** : écrire leurs sweeps, ajouter la ligne de dispatch de `qmatmul_q4_0_mma`, enregistrer `attention_prefill_mma` (point ouvert, étape 5 RTX).

## Synthèse
Rien n'est irrégénérable sur gfx906. Suppressions sûres proposées : 12 descripteurs inactifs gfx906 (certificat + enregistrement), 18 certificats `_gfx906` du dossier sm_86. Le reste demande un audit d'usage par famille (MMQ non dispatchés, MoE, par lots) et, sur sm_86, un catalogue + test de réciprocité. Deux trous de certification réels sur sm_86, à traiter en priorité.
