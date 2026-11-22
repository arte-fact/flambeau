# Port de `mul_mat_q_few` sur sm_86 : découpage en commits (Opus, rien de codé)

**Point de départ** : v3 6b69f618, branche locale qui part de v2. Décision prise sur mesure, voir coord/mmq-few-resultats.md :
- aucun noyau existant ne bat le batched MMVQ (C0) à m = 5 ni à m = 9 ;
- sur la même forme (4096 × 14336), llama.cpp reste plat de m = 4 à 16 (45 à 50 µs en Q4_K), alors que C0 croît avec m : 56,9 µs à m = 5, 126,5 à m = 9, 165,5 à m = 16.

**Cible** : la bande m = 2..24 (jusqu'à 3 groupes de 8 colonnes). m = 1 reste en MMVQ. La borne basse de chaque dtype se mesure ; elle n'est pas reprise des seuils de llama.cpp, réglés sur RTX 5090 (Q3_K et Q5_K ≥ 3, Q4_K, Q6_K et IQ4_NL ≥ 4, Q8_0 et IQ3_S ≥ 5, IQ4_XS ≥ 6).

## Référence : la structure de llama.cpp (`ggml-cuda/mmq-few.cu`, HEAD ffc87c3a7)
- **Bloc CUDA** : 16 lignes de poids (32 avec `nmt = 2`), 4 warps qui se partagent K par unités de 256 valeurs.
  - Chaque warp copie les octets bruts de ses lignes de poids dans son propre anneau `cp.async` : 2 étages, 3 en Q6_K. Un étage contient 1 ou 2 unités.
  - Le noyau décode ensuite depuis la mémoire partagée et fait des `mma.sync` int8 en m16n8k32, plus une variante k16.
  - Il termine par une réduction des 4 warps en mémoire partagée.
- **Activation** : quantifiée au format du fragment B (`quantize_mmq_few_q8_1`). Par unité de 256 et par lane, elle contient `qs` (int4[4]) et `ds` (int4[4]). On distingue deux dispositions :
  - DS4 (somme comprise) pour Q4_K et Q5_K, à cause des mins ;
  - D4 pour les autres types.
- **Gabarit** `<type, ncg, nmt>` :
  - `ncg` = nombre de groupes de 8 colonnes, de 1 à 3 ;
  - `nmt = 2` (tuile de 32 lignes) seulement si `ncg = 2` et `rows ≥ 2 × 16 × nSM` (2 624 lignes sur une 3090 à 82 SM), et seulement pour Q3_K, Q4_K, Q5_K, IQ3_S, IQ4_NL et IQ4_XS.
- **Conditions d'emploi** :
  - `k ≥ 8 × 256` ;
  - `k` multiple de `nunits_stage × 256` ;
  - poids alignés sur 16 octets, ou sur 4 ou 2 octets pour les types dont la taille ne tombe pas juste ;
  - activation alignée sur 16 octets.

## Ce qu'on réutilise
- `kernels-cuda/src/arch_primitives/mma_int8.cuh` : `tile_i16` et `tile_i8`, `load_generic`, `load_ldmatrix`, et `mma` m16n8k32 s8 (déjà utilisés par `qmatmul_q4_0_mma`).
  - Il manque la variante k16 (m16n8k16 s8) et les helpers `cp.async` (`cp.async.cg` 16 octets, `commit_group`, `wait_group`).
  - Ces deux ajouts vont dans `arch_primitives/sm_80.cuh` ou `mma_int8.cuh` : ce sont des intrinsèques NVIDIA, donc hors de `kernels-shared` (règle 5).
- **Les décodeurs de blocs** (Q4_K, Q5_K, Q3_K…) : `kernels-shared` a les siens dans la MMQ LDS-tiled. Le décodage au format fragment de llama.cpp (`mmq_few_vec_dot_*`) est différent et sera porté.

## Commits

### F0 : la disposition d'activation devient un type (prérequis, aucun noyau)
- Aujourd'hui, `qmatmul_reads_mmq_activation` renvoie un `bool` (DS4 ou Q8_1 standard), lu par 7 sites :
  - `dense_ffn` ×3 ;
  - `standard_attn` ×2 ;
  - `mtp` ;
  - `output_head` ;
  - `delta_net` ×2.
- Une troisième disposition (fragment) ne doit pas venir s'ajouter en un second booléen (règle 14 ; la règle 6 aussi : la disposition est un type).
- **Changement** :
  - `qmatmul_act_layout(shape, dtype) -> ActLayout { Q8_1, Mmq, Few }` ;
  - un seul helper `quantize_for(layout, …)` que les 7 sites appellent ;
  - `resolve_dispatch_recipe` porte la disposition de la recette (nouveau `RecipeKind::MmqFew`) ;
  - le `bool` disparaît.
- **Tests** : la disposition renvoyée est identique à celle d'aujourd'hui pour toutes les lignes des deux tables (test unitaire sur chaque descripteur × m de 1 à 64) ; all-dispatch et cert-check inchangés ; textes du Vibe identiques (RTX, MI50, ornith).

### F1 : quantification au format fragment, avec son cert
- **Impl** `quantize_f16_q8_1_few` en deux variantes, D4 et DS4 :
  - en F16, parce que les activations de flambeau sont en F16 (llama.cpp quantifie du F32) ;
  - sans intrinsèque, donc dans `kernels-shared` (règle 5) ;
  - une ligne `quantize_f16_q8_1` sur sm_86 (avec le choix de disposition) ;
  - dispatchée seulement quand une recette `MmqFew` la lit.
- **Cert** (`sweep --op quantize_f16_q8_1 --dtype few_d4|few_ds4`) :
  - comparaison à une référence CPU de la même disposition : int8 exacts, `d` et `s` en tolérance f16 ;
  - m de 1 à 24, k de 2 048 à 17 408 ;
  - colonnes partielles du dernier groupe (m = 9, 17) incluses.
- **Test** : un test unitaire de disposition (position de chaque int8 et de chaque `ds` pour une lane donnée), écrit contre la description de `quantize.cu` de llama.cpp et non contre le noyau lui-même.

### F2 : noyau `mmq_few_mma` + Q4_K
- **Noyau** `kernels-cuda/src/kernels/mmq_few_mma.cu`, gabarit `<dtype, ncg, nmt>`. Impl `qmatmul_q4_k_mmq_few_mma`, recette `MmqFew`, descripteur sm_86.
- **Entrées** : ncg de 1 à 3 avec nmt = 1, plus ncg = 2 avec nmt = 2.
- **Choix ncg / nmt à l'exécution, dans la recette** : `ncg = ceil(m/8)`, et `nmt = 2` selon le nombre de lignes et de SM. Ce n'est pas une ligne de table, parce que ce choix découle de m et de la géométrie, comme le découpage n8 + n1 du batched aujourd'hui.
- **Formes non servies** (`k < 2048`, `k % (nunits_stage × 256) ≠ 0`, alignement) : re-dispatch vers le batched MMVQ dans `resolve_dispatch_recipe`. Le précédent existe : le re-dispatch à `k % 128 ≠ 0` des recettes DS4.
- **Cert** (`sweep --op qmatmul_mmq --dtype Q4_K_few_mma`) :
  - m = 2, 3, 4, 5, 8, 9, 12, 16, 17, 24 ;
  - formes Qwen3.8-27B et 4096 × 14336 ;
  - tolérance des MMQ (5e-2) contre la référence CPU.
- **Bench** : candidat C6 dans qmatmul-bench, contre C0, à m = 2..24, 3 répétitions, nsys. La bande sm_86 se fixe à la frontière mesurée.
- **Tests de bout en bout** :
  - `qwen35-v2 real_gguf_cuda_parity` (critère IQ2_XXS) ;
  - all-dispatch et cert-check ;
  - Vibe : MTP-4 sur la RTX (tok/s, acceptation, textes identiques) contre v3.

### F3 : Q5_K
Même chemin q45 (DS4). On ajoute le décodeur des bits hauts, le cert, une ligne de bench et la bande mesurée.

### F4 : Q3_K
Disposition D4, décodeur Q3_K. Cert, bench et bande. L'anomalie n3/n4 du batched Q3_K (109 µs contre 69 à n5) n'a plus d'effet si la bande few commence à m = 3. On la consigne, sans la corriger dans ce commit.

### F5 et suivants : les autres dtypes, un commit par dtype
- Q6_K (3 étages, pas de tuile 32 lignes) ;
- Q8_0 : `ssm_alpha` et `ssm_beta`, +817 µs par cycle MTP-4 dans l'analyse nsys ;
- IQ3_S, IQ4_NL, IQ4_XS.

Chacun a son cert, sa ligne de bench et sa bande mesurée.

## Ce qui ne bouge pas
- **gfx906** n'est pas concerné : le noyau est NVIDIA seulement, comme chez llama.cpp. Les bandes gfx906 viennent du commit des raccourcis.
- **MUL_MAT_ID** (MoE) est hors périmètre de ce port.
- **Aucun gain projeté** avant le bench de F2.

## Risques connus
- **Registres** : `ncg = 3` déborde chez llama.cpp avec `nmt = 2`, d'où `nmt = 1` à `ncg = 3`. On vérifie à chaque cellule avec `cuobjdump -res-usage` (0 spill exigé).
- **Ordre d'accumulation** : 4 warps sur K plus une réduction partagée, à comparer au batched MMVQ. Les tests de parité réels (critère IQ2_XXS) doivent rester verts. Un écart de tokens dans les tolérances du cert est diagnostiqué avant tout relâchement.
- **Graphes CUDA** : la vérification MTP est capturée. `cp.async` ne pose pas de problème de capture, mais on vérifie avec `cuda_graph_capture_replay_*`.
