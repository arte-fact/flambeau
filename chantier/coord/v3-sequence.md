# V3 : séquence quand le GPU revient (Opus ; préparée, rien d'exécuté)

v3 (local, `~/Workspace/flambeau-impl-o`) = v2 cd3ab132 → 6b69f618 (raccourcis mesurés) → 66c52ff7 (F0 `ActLayout`) → e2b00e94 (F1 quantification few) → 11db931c (F2 Q4_K few MMA). Rien n'est poussé.

`FE=~/Workspace/flambeau-exp`, `SP=<scratchpad>`, `IO=~/Workspace/flambeau-impl-o`, `B=$IO/target-o/release/flambeau`, `BC=$IO/target-cuda/release/flambeau`.

**Gardes** :
- MI50 : `wait_gpu1.sh`, jonction ≤ 54 °C et VRAM < 5 % (décision provisoire du 22/11) ;
- MI50 à deux GPU : `$SP/wait_mi50s.sh` (écrit le 22/11), jonction ≤ 56 °C et VRAM < 5 % sur les GPU 0 et 1 (la jonction au repos est à 54-55 °C) ;
- RTX : `wait_rtx.sh`, ≤ 50 °C et < 1 Go ;
- un seul processus GPU à la fois, tout arrêté avant chaque rapport ; sd-server et llama-monitor ne sont pas touchés.

**Durées** :
- celles marquées « mesuré » viennent de la session du 22/11 :
  - build incrémental : 1 min 10 pour HIP, 1 min 30 pour CUDA ;
  - qmatmul-bench : environ 1 min par lancement sm_86 ;
  - gfx906 : 1 à 2 min par lancement, plus environ 10 min de refroidissement à chaque fois avec la garde ;
- les autres sont des **estimations**.

## 1. session_* sur MI50, sur bin_V2, avant le push de V2 (les deux MI50)
**Construction des tests**, dans `$IO` positionné sur **v2 cd3ab132** (`git stash` inutile : arbre propre ; on revient sur v3 après), avec `CARGO_TARGET_DIR=$IO/target-o` :
- `cargo test --release --features hip --no-run -p flambeau-qwen35-v2 -p flambeau-gemma4-v2 -p flambeau-qwen35moe-v2`, en `-j 8` ;
- **estimation : 5 à 10 min**, la première compilation des binaires de test en release.

**Lancement** : `HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=`, avec `wait_mi50s.sh` avant chaque test, un test à la fois (`--test-threads=1`) :

| test | crate | modèle |
|---|---|---|
| `session_qwen35_9b_pp_size_2_argmax_matches_sd` | qwen35-v2 | Qwen3.5-9B |
| `session_qwen35_9b_tp_size_2_argmax_matches_sd` | qwen35-v2 | Qwen3.5-9B |
| `session_qwen35_9b_pp2tp2_argmax_matches_sd` | qwen35-v2 | Qwen3.5-9B. Il demande 4 rangs : il sera sauté s'il n'y a que 2 GPU, à vérifier au lancement |
| `session_qwen35_27b_q4_0_tp_size_2_runs` | qwen35-v2 | Qwen3.5-27B-Q4_0 |
| `qwen38_27b_tp2_matches_pp2` | qwen35-v2 | `FLAMBEAU_27B_GGUF=~/models/Qwen3.8-27B-UD-Q4_K_M.gguf` |
| `session_qwen35_9b_q3_k_s_sd_finite_logits` | qwen35-v2 (k_quant_smoke) | Qwen3.5-9B-Q3_K_S |
| `session_forward_mixed_smoke` | qwen35-v2 | `FLAMBEAU_MIXED_MICROBENCH_GGUF`, s'il est défini |
| `session_gemma4_31b_tp_size_2_runs` | gemma4-v2 | gemma-4-31B |
| `session_qwen36_a3b_q3_k_s_pp_size_2_runs` | qwen35moe-v2 | Qwen3.6-35B-A3B Q3_K_S, si présent |

- **Estimation : 1 à 3 min par test, plus 10 min de refroidissement chacun, soit environ 1 h 30 à 2 h** pour les 9 tests.
- **Rendu** :
  - le verdict par test, ou SKIP avec sa raison ;
  - tout échec est diagnostiqué contre v1 (2ecc291c) avant toute correction, selon la consigne.
- Le push de V2 reste suspendu à ce rendu et à l'accord de Fable.

## 2. Build de v3 (aucun GPU)
- **Branche** : retour sur v3 (11db931c).
- **Builds**, l'un après l'autre :
  - `CARGO_TARGET_DIR=$IO/target-o cargo build -j 8 --release -p flambeau-cli --features hip_serve,hip_sweep` ;
  - puis `target-cuda` avec les 4 features.
- **Durée** :
  - estimation 3 à 6 min chacun ;
  - côté CUDA, 3 nouveaux `.cu` (`quantize_f16_q8_1_few`, `mmq_few_q4_k_mma`) ; côté HIP, aucun noyau modifié.
- **Contrôle hsaco** : 0 différent sur les 276 de bin_V2.
- **Contrôle cubin** : 2 de plus (`quantize_f16_q8_1_few`, `mmq_few_q4_k_mma`). Le registre CUDA charge `mmq_few_q4_k_mma` seulement dans le binaire avec `bench`, c'est-à-dire le binaire de sweep.

## 3. Certs manquants (règle 2 : rien de v3 ne part avant)
| # | commande | GPU | durée |
|---|---|---|---|
| 3a | `$B sweep --arch gfx906 --op qmatmul_mmq --dtype Q4_K_lds_tiled_y64_x16 --out .` | MI50 GPU 1 | estimation 2 min + refroidissement |
| 3b | `$B sweep --arch gfx906 --op qmatmul_mmq --dtype Q3_K_lds_tiled_x8 --out .` (re-sweep : m = 2, 5, 9 en plus) | MI50 GPU 1 | idem |
| 3c | `$B sweep --arch gfx906 --op qmatmul_mmq --dtype Q4_K_lds_tiled_y64_x8 --out .` (re-sweep) | MI50 GPU 1 | idem |
| 3d | `$BC sweep --arch sm_86 --op quantize_f16_q8_1_few --out .` : égalité à l'octet près avec la référence hôte, `max_rel_err = 0` partout | RTX | estimation < 1 min |
| 3e | `$BC sweep --arch sm_86 --op qmatmul_mmq_few_q4_k --out .` (tolérance 3e-2 · max(\|ref\|, √k)) | RTX | estimation 2 à 5 min (référence f64 sur l'hôte) |

- **Durée totale estimée** : environ 45 min, refroidissements MI50 compris.
- **Si un cert échoue** : on s'arrête et on écrit le diagnostic avant toute correction.
- **Commit** : un seul commit des certs sur v3 (5 JSON).

## 4. cert-check, all-dispatch et tests
1. **cert-check** :
   - `$BC cert-check` (gfx906), attendu **90 lignes / 0 échec** : 86 + 4 lignes, à savoir row-tile Q4_K, row-tile Q5_K, x16 Q4_K et x8 Q4_K 17..32 (cette dernière sur un cert existant) ;
   - `$BC cert-check --backend cuda --arch sm_86`, attendu **33 / 0** : aucune ligne sm_86 n'est ajoutée.
2. **all-dispatch gfx906** : `$B sweep --arch gfx906 --all-dispatch --out $SP/v3_ad_gfx906` (GPU 1). Estimation 20 à 30 min, refroidissements compris.
3. **all-dispatch sm_86**, inchangé : RTX, estimation 10 min.
4. **Tests sans GPU** (`std2.sh v3`), dont :
   - backend-hip `small_batch_bands_*`, `dispatch_toml_roundtrip`, `every_descriptor_cert_path_exists_on_disk` ;
   - backend-cuda `every_registered_cert_path_exists_on_disk` ;
   - ops `recipe_table_tests` ;
   - cli `sweep_all` ;
   - quant `q8_1_few`.

   Attendu : les mêmes échecs « demande un GPU » que V2, rien de nouveau. Estimation 15 min de CPU (aucun timing du Vibe en parallèle).
5. **Tests GPU** du F0 (bit-exact attendu) :
   - CUDA : `parity_snapshot_dense_cuda`, `embed_q4k`, `mtp_*` ;
   - HIP (GPU 1) : `synth_*`, `parity_*`, `snapshot_restore_parity` ;
   - croisés : `parity_moe_q4k_cross_backend`, `qwen35-v2 real_gguf_cuda_parity` (même cosinus 0,997056 qu'en V2).

   Estimation 30 min.
6. **ops_smoke / cuda_registry_smoke** : 2 stems CUDA de plus, 0 nouveau « skipped » hors liste CUDA-only.
7. **Vibe** : textes identiques à V2 (RTX, MI50, ornith). C'est le contrôle bit-exact du F0 et des bandes gfx906 en bout de chaîne.

## 5. Bench des raccourcis restants, m = 7 et m > 16 (gfx906, GPU 1, rocprofv3 + seg2.py)
- **Raccourci row-tile** :
  - `$B qmatmul-bench --arch gfx906 --dtype D --shapes $S --m 2,3,4,5,6,7,8 --impl C0,C1` pour D = Q6_K, IQ4_XS, IQ3_S, IQ3_XXS, IQ2_S, 3 répétitions ;
  - pour chaque D, il faut une cellule C1 (x8) de ce dtype : Q6_K et IQ4_XS existent, mais IQ3_S, IQ3_XXS et IQ2_S n'ont pas de x8. Pour eux, C0 est comparé à leur ligne de table (le premier candidat sans raccourci), à ajouter à `impl_id_for` avant le build.
- **m = 7** : Q5_K et Q4_K, C0 contre C1.
- **m > 16** : Q4_K, `--m 17,20,24,32`, C1 contre C2.
- **Durée** :
  - mesuré : environ 12 min par lancement avec le refroidissement ;
  - 5 dtypes × 3 répétitions + 2 lancements, soit environ **3 h 30**.

  Pour la réduire, on peut regrouper les m dans un seul lancement par dtype et répétition, comme ici. Le refroidissement reste le poste dominant.
- **Décision** : selon la règle de mmq-few-resultats.md, appliquée dans un commit de bandes sur v3, avec certs si une bande commence plus bas que son cert.

## 6. C0 contre C6 à m = 2..24 (RTX, nsys + seg2.py)
- **Commande** : `$BC qmatmul-bench --arch sm_86 --dtype Q4_K --shapes $S,4096:14336,14336:4096 --m 2,3,4,5,6,8,9,12,16,17,24 --impl C0,C6`, 3 répétitions.
- **Durée** : mesuré, environ 1 min par lancement, soit **5 min** environ, analyse comprise.
- **Rendu** :
  - le tableau pondéré comme dans mmq-few-resultats.md ;
  - la comparaison à llama.cpp (déjà mesuré) sur 4096 × 14336 ;
  - la bande sm_86 Q4_K proposée.
- Si C6 ne bat pas C0, c'est un résultat nul, écrit comme tel avec le diagnostic (PMC nsys), avant toute suite F3/F4.

## Ordre et occupation (validé par Fable, décision provisoire)
| ordre | étape | GPU | durée |
|---|---|---|---|
| 1 | session_* | 2 MI50 | ≈ 2 h |
| 2 | build | — | ≈ 10 min |
| 3 | certs | MI50 GPU 1 puis RTX | ≈ 45 min |
| 4 | contrôles | MI50 GPU 1 et RTX | ≈ 1 h 15 |
| 5 | **étape 6 : C0/C6** (débloque F3/F4) | RTX | ≈ 5 min |
| 6 | **étape 5 : bench gfx906** ; F3/F4 codés pendant ce temps | MI50 GPU 1 | ≈ 3 h 30 |

- **Pendant le bench gfx906** : des lancements RTX sont permis pendant les refroidissements, jamais pendant qu'un processus MI50 tourne.
- **Anomalie Q3_K (R = 2 ou préchargement)** : elle passe après le port.

## Prochain créneau RTX, après « v3 terminé » (v3 0b54b98e)
Commits depuis bin_V3 (50864a2f) :
- 41982716 : buffer few, repli et quantification few F32 ;
- 791daf50 : câblage des 7 sites ;
- 89237dc6 : ligne de dispatch sm_86 Q4_K few, m = 3..24 ;
- 8afec00e : test des sites ;
- 542eb3be : F3 Q5_K ;
- 0b54b98e : F4 Q3_K.

F3 et F4 ont été écrits sans compilation.

1. **Build** : `target-o` puis `target-cuda`, `-j 8`. Côté CUDA, **tous les cubins se recompilent** (en-têtes `arch_primitives` modifiés, et `build.rs` hache désormais aussi les en-têtes de `src/kernels`). Estimation : 5 à 10 min.
2. **nvcc -Xptxas -v** sur `mmq_few_q{4,5,3}_k_mma` : 0 spill exigé. On compare le SASS de `mmq_few_q4_k_mma` à celui de 11db931c. S'il diffère, on refait le sweep Q4_K (le cert actuel vaut pour 11db931c).
3. **Sweeps sm_86, RTX** :
   - `quantize_f32_q8_1_few` (attendu : 0 octet différent) ;
   - `qmatmul_mmq_few_q5_k` ;
   - `qmatmul_mmq_few_q3_k` ;
   - `qmatmul_mmq_few_q4_k` si le point 2 l'exige.

   Puis un commit des certs.
4. **cert-check sm_86**, attendu **35/0** (33 + les lignes few Q4_K et batched ≥ 25) ; cert-check gfx906 90/0 inchangé.
5. **all-dispatch sm_86**.
6. **Tests** :
   - sans GPU : backend-cuda `impls` (14/14 une fois le cert F32 présent), ops `recipe_table_tests` (dont `sm86_q4_k_few_column_layouts`), forward `few_activation_sites` ;
   - GPU CUDA : forward `embed_q4k`, `mtp_*`, `parity_snapshot_dense_cuda`, `cuda_registry_smoke` ;
   - croisés : `real_gguf_cuda_parity` (critère IQ2_XXS) et `parity_moe_q4k_cross_backend`.
7. **C6 de Q5_K et Q3_K** : `$BC qmatmul-bench --arch sm_86 --dtype Q5_K|Q3_K --shapes $S,4096:14336,14336:4096 --m 2,3,4,5,6,8,9,12,16,17,24 --impl C0,C6`, nsys, 3 répétitions. Les bandes sont proposées selon la règle ; une ligne de dispatch par dtype seulement après mesure.
8. **Vibe** : MTP-4 de bout en bout sur ce binaire, textes inclus.
9. **MI50** : étape 6 (bench gfx906).

**Condition 1 de Fable, le repli batched** : le cert de `qmatmul_{q4,q5,q3}_k_mmvq_dp4a_t128_vdr2_batched` teste chacune des entrées n1 à n8 (k = 2048, 5120 et 15360 ; 1 à 512 lignes de poids), et l'all-dispatch sm_86 le refait. Une matmul repliée à m = 3..24 n'enchaîne que ces entrées, par paquets de ≤ 8 colonnes (8 + 1, …, 8 + 8 + 8). Les noyaux sont donc couverts. L'enchaînement lui-même (pointeurs de colonne, découpage) n'est couvert par aucun cert, pas plus qu'avant pour la ligne m ≥ 2. Proposition : ajouter au sweep batched un cas qmatmul complet à m = 9, 17 et 24, à valider.
