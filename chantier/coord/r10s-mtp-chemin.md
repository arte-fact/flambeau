# r10s — chemin d'un cycle MTP-4 sur CUDA (bin_R10s_CUDA, 1b31581f ; lecture du code seulement)

Modèle Qwen3.8-27B-UD-Q4_K_M, `--kv q8`, 1 slot. Cycle (`qwen35-v2/src/model.rs::mtp_cycle`) :
1. Brouillon : 4 pas n = 1 sur la couche nextn (`mtp_draft_local`).
2. `gdn_snapshot`.
3. Vérification : `verify_layers` sur 5 tokens (m = 5), puis `output_head_all` sur 5 lignes.
4. Échantillonnage côté hôte.
5. Si rejet, `gdn_replay_rows`.
6. `mtp_stage_hidden` et `mtp_catchup`, l'attention nextn sur les tokens acceptés.

Aucun de ces chemins ne passe par le graphe CUDA, puisque la capture ne concerne que le décodage n = 1 hors MTP, et qwen35 ne la réussit de toute façon pas (r5-chemin-cuda.md).

## 1. Réponse courte
- **Il n'existe pas de MMVQ batched R10s.** Pour m entre 2 et 8, `qmatmul()` (ops/launch/qmatmul.rs:57) choisit **en dur**, avant le dispatch, `mmvq_dp4a_row_tile_batched_k` pour Q4_K, Q5_K, Q6_K, IQ4_XS, IQ3_S, IQ3_XXS et IQ2_S (`rowtile_batched_k_owns`, l.444).
  - Noyau (`mmvq_dp4a_batched_k.cuh`) : 256 threads, 8 lignes par bloc, 32 lanes par ligne, **un seul super-bloc à la fois par ligne**. La bande Q8_1 des m colonnes passe par la LDS, avec deux `__syncthreads` par super-bloc (20 itérations pour k = 5120, 68 pour k = 17408).
  - En Q4_K, un lane traite 1 entier (les deux quartets), soit 8 poids : c'est l'ancien découpage, pas le VDR 2 de R10s (16 threads par super-bloc, 8 super-blocs en vol par ligne).
  - Grille ceil(n / 8) × 256. Entrée `..._batched_n5`.
  - impl_id `qmatmul_<t>_mmvq_dp4a_row_tile_batched`, certifié sm_86 mais classé BENCH_REFERENCE (c'est la famille C de l'audit E4 : choisie en dur, non dispatchée).
- **Q3_K, IQ4_NL et Q8_0 à m = 5** : pas de noyau batched (Q8_0 n'a de row-tile que pour m entre 2 et 4). Ils prennent la ligne de dispatch sm_86, donc le noyau R10s t128 pour Q3_K et IQ4_NL, et `qmatmul_q8_0_mmvq_dp4a` pour Q8_0. Mais la boucle `for i in 0..m` (qmatmul.rs:172) le lance **5 fois**, et les poids sont relus 5 fois.
- **Le brouillon** (couche nextn) est en Q6_K / Q8_0 et en F16. Q6_K n'est pas touché par R10s (64 threads, déjà à parité selon r5). La tête Q6_K de 248 320 lignes est lancée **4 fois par cycle** à m = 1, à environ 1,2 ms chacune (r5 : 1224 µs). Les gains R10s ne s'y appliquent pas.

## 2. Vérification (m = 5, `verify_layers` + `output_head_all`)

### 2a. Choix des MMVQ à m = 5 (par type de poids)
| type | noyau | grille × bloc (n lignes) | choix |
|---|---|---|---|
| Q4_K, Q5_K, Q6_K, IQ4_XS, IQ3_S | `flambeau_mmvq_<t>_dp4a_q8_1_batched_n5` (stem `mmvq_<t>_dp4a_batched`) | ceil(n / 8) × 256 | en dur (`rowtile_batched_k_owns`) |
| Q3_K | `flambeau_mmvq_q3_k_dp4a_t128_vdr2_q8_1` × 5 lancements | n × 128, ×5 | dispatch sm_86 (R10s) |
| IQ4_NL | `flambeau_mmvq_iq4_nl_dp4a_t128_vdr2_q8_1` × 5 | n × 128, ×5 | dispatch sm_86 (R10s) |
| Q8_0 | `flambeau_mmvq_q8_0_dp4a_q8_1` × 5 | n × 256, ×5 | dispatch sm_86 |

Exemples de grilles row-tile : 17408 lignes → 2176 × 256 ; 10240 → 1280 ; 12288 → 1536 ; 5120 → 640 ; 1024 → 128 ; tête 248320 → 31 040.

### 2b. Par couche GDN (×48), `forward_prefill_with_ar_hook` (delta_net.rs:1370), avec `mtp_spec_rows`
| # | noyau | poids | grille × bloc | impl_id / choix |
|---|---|---|---|---|
| 1 | flambeau_rmsnorm_f16 (attn_norm) | F32 | 5 × 256 | rmsnorm_f16 (DC) |
| 2 | flambeau_quantize_row_f16_q8_1 | — | 800 × 32 | quantize_f16_q8_1 |
| 3 | flambeau_quantize_f16_q8_1_mmq (**inutile** : aucun MMQ à m = 5) | — | (40, …) × 128 | quantize_f16_q8_1_mmq |
| 4 | attn_qkv n = 10240 | Q4_K 29, Q5_K 12, IQ4_XS 6 (row-tile) ; IQ4_NL 1 (×5) | 1280 × 256 | §2a |
| 5 | attn_gate n = 6144 | Q4_K 24, Q5_K 12, IQ4_XS 12 (row-tile) | 768 × 256 | §2a |
| 6 | ssm_alpha n = 48 | Q8_0 → **5 lancements** | 48 × 256 ×5 | dispatch |
| 7 | ssm_beta n = 48 | Q8_0 → **5 lancements** (pas de fusion alpha+beta à m > 1) | 48 × 256 ×5 | dispatch |
| — | 2 copies DtoD (historique + qkv dans conv_input) | — | — | — |
| 8 | flambeau_causal_conv1d_f32 | ssm_conv1d F32 | (40, 5) × 256 | causal_conv1d_f32 |
| — | 1 copie DtoD (décalage de l'historique) | — | — | — |
| 9 | flambeau_silu_f32 (5 × 10240) | — | 200 × 256 | silu_f32 |
| 10 | flambeau_gdn_split_qkv_f32 | — | 200 × 256 | gdn_split_qkv_f32 |
| 11-12 | flambeau_l2_norm_f32 Q, K (5 × 16 lignes) | — | 80 × 256 | l2_norm_f32 (DC) |
| 13 | flambeau_scale_f32 Q | — | 40 × 256 | scale_f32 (DC) |
| 14 | flambeau_gdn_state_step_alphabeta_f32_s128 (l = 5, boucle interne) | — | (48, 1, 32) × (32, 4) | gdn_state_step_alphabeta_f32 |
| 15 | flambeau_rmsnorm_f32 (5 × 48 lignes) | F32 | 240 × 256 | rmsnorm_f32 (DC) |
| 16 | flambeau_swiglu_f32 (5 × 6144) | — | 120 × 256 | swiglu_f32 |
| 17 | flambeau_quantize_row_q8_1 | — | 960 × 32 | quantize_q8_1 |
| 18 | flambeau_quantize_q8_1_mmq (**inutile**) | — | … × 128 | quantize_q8_1_mmq |
| 19 | ssm_out n = 5120, k = 6144 | Q5_K 33, Q6_K 5, IQ4_XS 5, Q4_K 5 (row-tile) | 640 × 256 | §2a |
| 20 | flambeau_cast_f32_f16 | — | 100 × 256 | cast_f32_f16 (DC) |
| 21 | flambeau_add_f16 (résiduel) | — | 100 × 256 | add_f16 (DC) |

Environ 29 lancements par couche GDN, dont 10 MMVQ Q8_0 de 48 lignes. Sur 48 couches : **≈ 480 lancements alpha/beta minuscules par vérification**.

### 2c. Par couche d'attention (×16), standard_attn n = 5 (`prefill_shape`, KV Q8)
| # | noyau | poids | grille × bloc | choix |
|---|---|---|---|---|
| 1-3 | rmsnorm_f16, quantize_f16_q8_1, quantize_f16_q8_1_mmq (le dernier inutile) | — | 5 × 256 / 800 × 32 / … | — |
| 4 | attn_q à porte n = 12288 (F32) | IQ4_XS 8, Q4_K 6, Q5_K 2 (row-tile) | 1536 × 256 | §2a |
| 5-6 | cast_f32_f16 (5 × 12288), split_q_gate_f16 | — | 240 × 256 ; (5, 24, 2) × 128 | DC |
| 7 | attn_k n = 1024 (F32, puis cast) | Q6_K 9, Q5_K 5, Q4_K 1 (row-tile) ; **Q8_0 1 (×5)** | 128 × 256 | §2a |
| 8 | attn_v n = 1024 (F32, puis cast) | Q5_K 6, Q6_K 3 (row-tile) ; **Q8_0 7 (×5)** | 128 × 256 | §2a |
| 9-10 | cast_f32_f16 K, V | — | 20 × 256 | DC |
| — | positions : HtoD de 20 o depuis de la mémoire paginée | — | — | — |
| 11-12 | flambeau_rmsnorm_rope_neox_partial_f16_d256 Q, K | F32 | (5, 24, 1) / (5, 4, 1) × 256 | — |
| 13-14 | flambeau_quantize_row_f16_q8_0 K, V (append Q8, 5 lignes) | — | 160 × 32 | quantize_f16_q8_0 |
| 15 | flambeau_attention_prefill_flash_tile_d256_br8_q8_kv (n_q = 5 ≥ 4) | — | (1, 24) × (32, 8) | attention_prefill_flash_tile_q8_kv (BR) |
| 16 | sigmoid_mul_f16 | — | 120 × 256 | — |
| 17-18 | quantize_f16_q8_1 ; quantize_f16_q8_1_mmq (inutile) | — | 960 × 32 / … | — |
| 19 | attn_output n = 5120, k = 6144 | Q5_K 11, Q6_K 4, IQ4_XS 1 (row-tile) | 640 × 256 | §2a |
| 20-21 | cast_f32_f16, add_f16 | — | 100 × 256 | DC |

### 2d. FFN dense (×64), n = 5
| # | noyau | poids | grille × bloc | choix |
|---|---|---|---|---|
| 1-2 | rmsnorm_f16, quantize_f16_q8_1 (le MMQ seulement si un poids le lit : non à m = 5) | — | 5 × 256, 2720 × 32 | — |
| 3 | ffn_gate n = 17408 | IQ4_XS 31, Q4_K 14, Q5_K 12, IQ3_S 1 (row-tile) ; **Q3_K 4, IQ4_NL 2 (×5)** | 2176 × 256 (row-tile) ; 17408 × 128 ×5 | §2a |
| 4 | ffn_up n = 17408 | IQ4_XS 30, Q5_K 16, Q4_K 14, Q6_K 1 (row-tile) ; **Q3_K 2, IQ4_NL 1 (×5)** | idem | §2a |
| 5 | swiglu_f32_to_f16 (5 × 17408) | — | 340 × 256 | — |
| 6 | quantize_f16_q8_1 | — | 2720 × 32 | — |
| 7 | ffn_down n = 5120, k = 17408 | IQ4_XS 24, Q5_K 22, Q4_K 10, IQ3_S 3, Q6_K 1 (row-tile) ; **IQ4_NL 3, Q3_K 1 (×5)** | 640 × 256 ; 5120 × 128 ×5 | §2a |
| 8-9 | cast_f32_f16, add_f16 | — | 100 × 256 | DC |

### 2e. Tête (une fois par cycle), `output_head_local(force_emit_all)`
- rmsnorm_q8_1_fused sur 5 lignes : 5 × 1024.
- MMVQ Q6_K à m = 5 (row-tile `flambeau_mmvq_q6_k_dp4a_q8_1_batched_n5`) : 31 040 × 256, choisi en dur.
- Copie DtoH de 5 × 993 280 o (4,97 Mo) vers de la mémoire paginée, puis synchronisation, puis `to_vec` côté hôte.

## 3. Brouillon : 4 pas n = 1 sur la couche nextn (blk.64), `mtp_draft_local` (mtp.rs:174)
| # | noyau / copie | poids | grille × bloc | choix |
|---|---|---|---|---|
| 1 | embedding : DtoD de 10 240 o | token_embd (F16 en VRAM) | — | — |
| 2-3 | rmsnorm_f16 (enorm), rmsnorm_f16 (hnorm) | F32 | 1 × 256 | DC |
| 4-5 | flambeau_dense_gemv_f16_f16_batched : eh_proj_e puis eh_proj_h, 5120 × 5120 chacun | **F16** (Q6_K dans le GGUF, déquantifié au chargement, loader.rs:296-320) : 105 Mo lus par pas, contre ≈ 43 Mo en Q6_K | (5120, 1) × 256 | dense_gemv_f16_f16_batched (BR), en dur |
| 6-7 | add_f32, cast_f32_f16 | — | 20 × 256 | DC |
| 8 | standard_attn de décodage (comme r5-chemin-cuda §3) | attn_q Q6_K (12288), attn_k Q8_0, attn_v Q8_0, attn_output Q6_K | MMVQ dispatch sm_86 m = 1 : Q6_K 12288 × 64 ; Q8_0 en F16 direct 1024 × 256 ; append Q8 et split-K | dispatch |
| 9 | add_f16 | — | 20 × 256 | DC |
| 10 | dense_ffn m = 1 (comme r5 §4) | ffn_gate, up, down **Q6_K** | 17408 × 64 (×2) ; 5120 × 64 | dispatch (Q6_K, non R10s) |
| 11-12 | rmsnorm_f16 (shared_head_norm), quantize_f16_q8_1 | — | 1 × 256 ; 160 × 32 | — |
| 13 | **tête Q6_K** m = 1, 248 320 lignes | Q6_K | 248 320 × 64 (≈ 1,2 ms) | dispatch |
| — | DtoH 993 280 o (paginée), synchronisation, échantillonnage hôte | — | — | — |

Par cycle : ×4. Donc 4 têtes Q6_K (≈ 4,8 ms), 8 GEMV F16 de 5120² (≈ 420 Mo lus), 4 synchronisations et 4 copies de logits de 970 Ko.

## 4. Autour de la vérification
- **`gdn_snapshot`** (engine.rs:1155) : pour chacune des 48 couches GDN, DtoD de l'état (48 × 128 × 128 × 4 = 3,1 Mo) et de l'historique (120 Ko), soit **≈ 151 Mo copiés par cycle**, suivis d'un `Stream::synchronize`.
- **`gdn_replay_rows`** (si rejet) : par couche GDN, restauration de l'état puis `gdn_state_step_alphabeta_f32_s128` sur accepted + 1 lignes.
- **`mtp_stage_hidden`, `capture_pending`** : rmsnorm_f16 et une copie DtoD.
- **`mtp_catchup`** (`mtp_attn_rows` sur les acceptés, n ≤ 4) : embedding, 2 rmsnorm, 2 GEMV F16 de 5120² en batched n, add, cast, puis attention nextn en n > 1 (row-tile Q6_K et Q8_0 ×n, flash_tile si n ≥ 4 sinon `attention_prefill_q8_kv`), puis add.

## 5. Ce qui est choisi en dur, et non par le dispatch sm_86
- m entre 2 et 8 : row-tile batched pour Q4_K, Q5_K, Q6_K, IQ4_XS, IQ3_S, IQ3_XXS et IQ2_S. Aussi Q8_0 et Q4_1 row-tile ou batched pour m entre 2 et 4, et Q4_0 row-tile (qmatmul.rs:43-130).
- `dense_gemv_f16_f16_batched` pour eh_proj.
- attention : flash_tile ou une passe selon n_q ≥ 4.
- quantize MMQ calculé systématiquement à n > 1 dans l'attention et le GDN.

Passent par le dispatch : tous les MMVQ à m = 1 (brouillon, et tête du brouillon), et ceux à m = 5 des types non couverts par le row-tile (Q3_K, IQ4_NL, Q8_0), mais en boucle sur les 5 lignes.

## 6. Pistes qui en découlent (à chiffrer sur la trace nsys MTP-4 du Vibe)
1. Row-tile batched dans le découpage R10s (VDR, plusieurs super-blocs en vol par ligne) pour IQ4_XS, Q5_K, Q4_K et Q6_K. C'est le gros de la vérification et de la tête m = 5.
2. Couverture m > 1 de Q3_K, IQ4_NL et Q8_0 pour supprimer les boucles ×5. Pour Q8_0 en particulier : les 480 lancements alpha/beta du GDN, ou alpha+beta fusionnés à m > 1.
3. Brouillon : tête Q6_K ×4. Le Q6_K à m = 1 est à parité avec llama.cpp ; le levier serait structurel (vocabulaire réduit pour le brouillon, ou argmax sur le GPU sans copie des logits).
4. eh_proj gardé en Q6_K (MMVQ) au lieu de F16 : 2,4 fois moins d'octets par pas.
5. Snapshot GDN de 151 Mo par cycle, et les quantize MMQ inutiles à n > 1.

## 7. Trace MTP-4 (r10s/mtp/mtp.sqlite) : d'où viennent les MMVQ m = 1

**Ce n'est pas un forward m = 1 par cycle : c'est le préremplissage de l'invite.**
- Les 4 087 appels IQ4_XS 17408 lignes en m = 1 tombent **tous avant la première vérification** (première `batched_n5` à t = 1 310 ms). C'est 61 projections IQ4_XS 17408 par forward (gate 31 + up 30) × 67 tokens d'invite = 4 087, exactement.
- Même chose pour Q5_K 17408 (28 × 67 ≈ 1 890) et Q4_K 17408 (28 × 67).
- Q3_K 17408 : 402 appels avant (6 × 67) et 1 332 après (6 couches × 5 lignes × 45 cycles, la boucle ×5 de la vérification, §2a).
- Pourquoi : `dispatch/cuda/sm_86.toml` n'a que des lignes MMVQ `m = "any"` (sauf Q4_0 en MMA). Un préremplissage de m = 67 passe donc par la boucle `for i in 0..m` de `qmatmul()` (qmatmul.rs:172) : 67 lancements MMVQ par projection.
- Coût : préremplissage de 67 tokens = **1,31 s** (1 276 ms de noyaux), soit 28 % de la durée client de 4,72 s. C'est un coût par requête, qui pèse sur le TTFT et non sur le cycle.
- Il n'y a ni repli m = 1 après rejet, ni rattrapage de la cible : `mtp_cycle` ne refait un forward m = 1 que si `mtp_pending` ne correspond pas. Or `capture_pending` le remet à (slot, pos + accepted) à chaque cycle, et 200 tokens en 45 cycles (4,4 par cycle) exclut des cycles de repli à 1 token.

**Coût réel par cycle, hors préremplissage** : de t = 1 310 ms à la fin, 3 378 ms pour 45 cycles, soit **75,1 ms par cycle** (66,4 ms de noyaux). Les 105 ms de r10s-mtp-analyse.md incluent le préremplissage ; llama.cpp ≈ 56 ms.

| poste (par cycle, après le préremplissage) | appels | ms |
|---|---|---|
| vérification row-tile batched_n5 (Q4_K/Q5_K/Q6_K/IQ4_XS/IQ3_S, tête m = 5 comprise) | 385 | **48,1** |
| Q6_K m = 1 : brouillon nextn (q, o, gate, up, down) et 4 têtes de 248 320 lignes | 23 | 6,1 |
| vérification Q3_K / IQ4_NL, boucle ×5 (t128 R10s) | 69 | 3,7 |
| petits noyaux (norm, quantize, cast, add, split, silu…) | 1 467 | 3,3 |
| attention (prefill flash_tile, décodage nextn) | 25 | 1,9 |
| vérification Q8_0 alpha/beta, boucle ×5 (48 lignes) | 469 | 1,3 |
| eh_proj F16 (dense_gemv) | 10 | 0,9 |
| gdn_state_step (vérification, replay) | 60 | 0,9 |
| Q8_0 m = 1 autres (k/v) | 47 | 0,4 |
| DtoD (snapshot GDN et copies) | 295 | 0,9 (231 Mo) |
| DtoH (logits : 4 du brouillon et 1 de vérification) | 5 | 1,9 (8,7 Mo) |
| trous (lancements, 5 synchronisations, échantillonnage hôte) | — | ≈ 6 |

## 8. Gains chiffrés (estimations, non mesurées)

**(a) « Supprimer les forwards m = 1 »** : **0 ms par cycle**. Ces forwards n'existent pas dans le cycle. Le vrai sujet est le préremplissage sm_86 en MMVQ ligne par ligne : 1,31 s pour 67 tokens, alors qu'un passage MMQ (ou batched par paquets de 8) le ramènerait à quelques dizaines de ms. Le gain porte sur le TTFT, et sur le t/s **seulement si** la mesure lp.py compte le préremplissage (200 / 4,72 = 42,4 t/s avec, 200 / 3,38 = 59 t/s sans). À vérifier avant toute comparaison avec llama.cpp (`predicted_per_second` exclut le préremplissage).

**(b) Batched R10s pour n de 2 à 5** : le seul gros levier par cycle.
- Aujourd'hui, la vérification coûte 48,1 (row-tile) + 3,7 (boucles Q3_K / IQ4_NL) + 1,3 (alpha/beta Q8_0) = **53 ms**. IQ4_XS n5 prend 131 µs, contre 49 µs à m = 1 (×2,7).
- Un noyau limité par la bande passante qui lit les poids une fois pour 5 colonnes devrait coûter ≈ 1,2 à 1,4 × m = 1. Les MMVQ m = 1 d'un forward cible font ≈ 22 ms, donc la vérification passerait à **≈ 27 à 31 ms**.
- **Gain ≈ 22 à 26 ms par cycle** : 75 → environ 50 ms par cycle, au niveau des ≈ 56 ms de llama.cpp. À 4,4 tokens par cycle, ≈ 85 à 90 t/s hors préremplissage (estimation).
- Mise en œuvre :
  - noyau `mmvq_<t>_vdrN_batched_row<OutT, THREADS, N>` qui reprend les corps R10s (une ligne par bloc, N accumulateurs, les N colonnes Q8_1 lues depuis la L2 ou la LDS) pour IQ4_XS, Q5_K, Q4_K, Q6_K, puis Q3_K, IQ4_NL et Q8_0 (ce qui supprime aussi les boucles ×5) ;
  - impl `qmatmul_<t>_mmvq_dp4a_t128_vdrN_batched`, cellule partagée avec une entrée par N de 2 à 8 ;
  - **dispatch** : lignes sm_86 `m = "2..8"` (le format sait déjà lire `lo..hi`), avec un nouveau `RecipeKind::MmvqBatched` ;
  - dans `qmatmul()`, la branche `rowtile_batched_k_owns` ne s'applique plus que si la recette dispatchée pour m n'est pas `MmvqBatched`. gfx906 n'a aucune ligne de ce genre, donc il garde exactement le chemin row-tile actuel ; seuls de nouveaux .hsaco s'ajoutent, sans toucher aux existants (contrôle `cmp`) ;
  - certificats : un par impl et par type sur sm_86, sweep à n = 2..8 avec comparaison au bit près contre n lancements m = 1 du même corps (même ordre de réduction par colonne).

## 9. Ordre recommandé
1. **(b) Batched R10s n = 2..8 sur sm_86**, IQ4_XS d'abord (le plus fréquent), puis Q5_K, Q4_K, Q6_K (la tête de vérification et ssm_out), puis Q3_K, IQ4_NL et Q8_0. Gain estimé de 22 à 26 ms sur 75 par cycle.
2. **Préremplissage sm_86** : lignes MMQ (ou batched par paquets de 8) pour m > 8. Gain : TTFT de 1,31 s à quelques dizaines de ms ; c'est (a) correctement posé. À faire passer avant (b) si la métrique de l'objectif inclut le préremplissage.
3. Ensuite : le brouillon (4 têtes Q6_K ≈ 4,8 ms par cycle, eh_proj en F16), les 1 467 petits lancements et le snapshot GDN (≈ 6 + 3 + 1 ms par cycle).
