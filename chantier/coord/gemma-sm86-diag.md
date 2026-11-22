# gemma-4 sur sm_86 : deux bogues antérieurs à V1 (lecture du code et des logs, rien d'exécuté ni de codé)

## (1) gemma-4-12b-it-Q4_0 : HTTP 500 au préremplissage, « kernel stem `mmvq_q4_k_r4` not loaded »

**Constat** (v1/fl_g1.log, et fl_g2.log à l'identique) : `chunked_prefill chunk [0..47] of 47: kernel stem mmvq_q4_k_r4 not loaded in registry`, sur `--backend cuda`.

**Qui demande ce stem : aucune ligne de dispatch, un chemin codé en dur.**
- Le modèle est presque entièrement en Q4_0 (322 tenseurs, plus 6 `ffn_down` en Q4_1). Mais `token_embd` est en **Q4_K** [262144, 3840] et sert de tête de sortie (tête liée).
- `crates/forward/src/core/composites/output_head.rs:82-99` : quand on n'émet qu'une ligne, que la tête est en Q4_K, que `vocab ≥ 131072` et que `hidden % 256 == 0`, le composite appelle directement `ops.mmvq_q4_k_r4(…)`.
- Cet op (`ops_impl.rs:175`) lance le stem fixe `mmvq_q4_k_r4` / `flambeau_mmvq_q4_k_r4_q8_1`, sans passer par `qmatmul` ni par la table. Le commentaire du composite justifie ce choix par un gain mesuré sur E4B-Q4_0 (HIP).
- Le préremplissage n'émet qu'une ligne (la dernière), donc le chemin est pris dès la première requête.

**Pourquoi le stem n'est pas chargé sur CUDA.** `mmvq_q4_k_r4` est un noyau HIP écrit à la main (kernels-hip). Il n'a pas de cellule dans `instantiations.toml` ni de port CUDA. Le registre CUDA le range donc parmi ses 26 stems sautés (`cuda_registry_smoke` : « skipped 26 of 281 », `mmvq_q4_k_r4` en fait partie), et `expect_module` échoue au lancement.

**Ce n'est donc pas un trou de la table, mais un choix de variante fait hors de la table**, comme le raccourci row-tile de `qmatmul` (règles 1 et 13 : un test implicite d'arch par présence de module). Sur gfx906, il choisit en silence une variante sans ligne de dispatch ni cert de dispatch.

**Correctif proposé** (un commit, sans nouveau noyau) :
1. Supprimer la branche `use_r4_lmhead` d'`output_head_local` : la tête Q4_K passe par `lm_head.qmatmul(…)`. Sur sm_86, elle prend alors la ligne `m = 1` Q4_K (`qmatmul_q4_k_mmvq_dp4a_t128_vdr2`).
2. Si r4 vaut quelque chose sur gfx906, le mesurer avec `qmatmul-bench` (Q4_K, `--shapes 262144:3840,131072:2560 --m 1`, contre la ligne `m = 1..2` actuelle) :
   - s'il gagne, en faire une vraie ligne gfx906 Q4_K `m = 1`, avec une bande `n` si la table la porte, sinon pour toute largeur, avec cert et all-dispatch ;
   - sinon, on ne garde rien.
   - La méthode d'op `Ops::mmvq_q4_k_r4` sert aussi à d'autres ? Si elle n'a plus d'appelant, on la retire.
3. **Test qui le prouve :**
   - un test forward `output_head_q4k_large_vocab_{hip,cuda}` : tête Q4_K synthétique avec `vocab = 131072`, `hidden = 256`, une ligne. Il compare `output_head_local` à `qmatmul` Q4_K à `m = 1` sur la même entrée (égalité au bit près, puisque c'est le même noyau après correctif) et échouerait aujourd'hui sur CUDA avec « not loaded » ;
   - côté Vibe, la section E de la consigne (gemma-4-12b-it-Q4_0 sur la RTX) doit passer de HTTP 500 à 5 textes.

## (2) gemma4-v2 `real_gguf_cuda_parity` (gemma-4-12B-it-qat-UD-Q4_K_XL) : CUDA s'écarte de llama.cpp au 5e token

**Constat :**
- CUDA donne `[100, 45518, 107, 101, 236777, …]`, la référence llama.cpp `[100, 45518, 107, 101, 818, …]`.
- La séquence est identique sur V1 (2ecc291c) et sur V2 : c'est antérieur.
- Le test s'arrête sur cet assert, avant la comparaison HIP : on ne sait donc pas aujourd'hui si HIP colle encore à llama.cpp.
- Le test était vert, en « exact match », à b538847e (2026-06-19). Il y a 142 commits jusqu'à 2ecc291c.

**Le modèle :**
- Malgré son nom, le GGUF ne contient **que du Q4_0** pour les matmuls (q, k, o, v ×40, ffn ×3, `token_embd`). Pas un seul tenseur Q4_K.
- hidden 3840 ; couches globales en head_dim 512, couches SWA en 256 ; KV F16 dans le test ; softcap 30.
- Invite de 33 tokens, puis décodage token par token.
- Les 4 premiers tokens concordent : l'argmax du préremplissage est correct, et l'écart s'accumule.

**Ops gemma dont le chemin diffère entre sm_86 et gfx906.** Ce sont des candidats, du plus suspect au moins suspect, d'après les dates :
1. **Attention de préremplissage (m = 33).**
   - sm_86 : `attention_prefill_mma` (ligne m ≥ 16), MMA f16 sur tensor cores pour d128, d256 et d512. Elle est arrivée le **2026-06-20**, soit le lendemain du test vert : 182ef2f5 pour d256/d128, e586c6a9 pour d512 « +10,7 % RTX prefill ». Elle est routée par la table depuis 42626448 (2026-11-21), mais elle était déjà prise auparavant dès que son module était chargé.
   - gfx906 : `attention_prefill_f16_gfx906` (scalaire).
   - Le cert sm_86 couvre d = 128, 256 et 512 avec une tolérance relative de 0,005 (max mesuré 2,4e-4).
   - Le préremplissage écrit le KV de toutes les couches, à travers les sorties d'attention des couches précédentes. Un écart ici déplace tout le décodage qui suit : **c'est le premier suspect**.
2. **Q4_0 en décodage, sortie F16 directe** (q / k / v / o à m = 1). Depuis ae330c82 (2026-11-20, « F16-output decode runs the m = 1 dispatch kernel »), le noyau F16 direct est celui de la recette `m = 1` (`mmvq_m1_recipe`, fixée à `mmvq_q4_0` pour Q4_0). Il faut vérifier si sm_86 lançait un autre noyau avant ce commit.
3. **Matmuls Q4_0 du préremplissage (m = 33).**
   - sm_86 : ligne `m = 1..127` → `qmatmul_q4_0_mmvq_dp4a`, MMVQ ligne par ligne sur l'activation Q8_1 standard.
   - gfx906 : `qmatmul_q4_0_mmq_wave64_gfx906` (m ≥ 32).
   - L'ordre d'accumulation diffère, mais le chemin sm_86 est le même noyau qu'en décodage.
4. **GELU de la FFN en préremplissage** : 5fffa12a (2026-07-08) fusionne gelu → Q8_1 au format MMQ, et la quantification inutilisée est sautée selon la disposition lue. Sur sm_86, la disposition lue est la standard : à vérifier que la branche « dead layout » choisit bien la bonne.
5. **Attention de décodage F16** : `attention_decode_f16` des deux côtés, mais réécrite pour la table (f722655e, 2026-11-21) et en positions sur le device (5be9b035, 2026-07-08). Ce serait le suspect si (1) était innocent.

**Test qui tranchera** (au prochain créneau GPU, après l'étape 0) :
- **Bisect** du test lui-même sur b538847e..2ecc291c, limité aux chemins `kernels-cuda`, `kernels-shared`, `ops`, `backend-cuda`, `dispatch/cuda` et `forward`, comme pour IQ2_XXS. Environ 8 builds en `-j 8`. Seule l'assertion CUDA contre llama.cpp sert : on lance avec `HIP_VISIBLE_DEVICES=-1`, la partie HIP se met alors en « SKIP » d'elle-même, ce qu'il faudra vérifier.
- **Contrôle direct du suspect 1, à faire avant le bisect parce qu'il coûte un seul build** : un test `cfg(test)` qui fait tourner le préremplissage sm_86 avec `attention_prefill_flash_tile_f16` pour toute valeur de m (dispatch de test), puis le même assert. Si la séquence redevient celle de llama.cpp, c'est l'attention MMA.
- **Correctif selon le résultat :**
  - **si c'est l'attention MMA** : refaire le cert de `attention_prefill_mma` avec des formes gemma (d512, m = 33, contexte 33) et la comparer au flash_tile en logits ;
    - si l'écart vient d'une accumulation P·V en f16, passer à une accumulation f32, puis refaire le cert ;
    - sinon, faire commencer sa bande à la frontière où le gain mesuré justifie l'écart, et relâcher le test comme IQ2_XXS (préfixe ≥ N + cosinus) si l'écart est dans la tolérance du cert.
  - **si c'est un autre commit** : un correctif ciblé sur le premier commit fautif, avec un assert de parité de l'op concerné.
- **En parallèle**, faire tourner la partie HIP seule (`CUDA_VISIBLE_DEVICES=`) pour savoir si gfx906 suit encore llama.cpp, ce qui confirmera que l'écart est propre à sm_86.

## Les 79 échecs des tests « sans GPU »
- **Ce sont tous des tests qui demandent un GPU**, et pour certains un modèle en plus. Le script masque les GPU (`HIP_VISIBLE_DEVICES=-1`, `CUDA_VISIBLE_DEVICES=`), si bien que la création du device échoue : 65 « no ROCm-capable device » et 18 `CUDA_ERROR_NO_DEVICE` / « no CUDA-capable device » dans les messages.
- **Répartition :**
  - 39 tests unitaires GPU de `model-ops` (`ops::*::tests`, via `model-ops/src/testing.rs`) ;
  - environ 25 tests forward synthétiques (`synth_*`, `parity_*`, `snapshot_*`, `mtp_*`, `embed_q4k`, `gemma_ring_swa_*`) ;
  - 2 tests de registre ou de capture (`cuda_registry_smoke`, `cuda_graph_capture_replay_*`) et `nccl_single_rank_allreduce_is_identity` ;
  - 9 tests sur modèle réel (`session_*`, `qwen38_27b_tp2_matches_pp2`, les deux `real_gguf_cuda_parity`, `k_quant_smoke`), qui demandent un GPU et un GGUF.
- **La liste est la même en V1 et en V2** : c'est la référence (`failed_o2` / `failed_v1`), et on vérifie seulement qu'aucun nouvel échec n'apparaît.
- **Les tests GPU pertinents sont relancés à part, GPU visibles**, et sont verts, sauf les deux `real_gguf_cuda_parity` décrits ici et dans v1-parity-iq2xxs.md.
- Les 9 tests sur modèle réel et les tests TP ou PP à 2 MI50 (`session_*`) n'ont pas été relancés sur GPU en V1 ni en V2.
