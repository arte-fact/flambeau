# Microbench de l'étape 0 (qmatmul-bench) : résultats et décision (Opus, 2026-11-22)

## Conditions
- **Binaires** : v2 cd3ab132 (265ea8ef + C5), build incrémental. `target-o` pour gfx906, `target-cuda` pour sm_86.
- **Formes** : les 7 formes de vérification de Qwen3.8-27B, plus 4096:14336 et 14336:4096 pour comparer à llama.cpp.
- **m** : 2, 3, 4, 5, 6, 8, 9, 12, 16. Candidats C0 à C4, 200 itérations après 3 appels de chauffe, **3 répétitions**.
- **Temps** : noyau seul, médiane par appel, issue des traces nsys (`cuda_gpu_trace`) et rocprofv3 (`--kernel-trace`). Un appel peut lancer 2 noyaux (C0 sm_86 m ≥ 9 = n8 + n…, C0 gfx906 Q3_K m = 2 = r2 ×2) : on les additionne.
- **Gardes** :
  - RTX : ≤ 50 °C et < 1 Go, avant chaque lancement.
  - GPU1 MI50 : jonction ≤ 50 °C pour la répétition 1 (Q5_K, Q4_K, Q3_K), puis **≤ 54 °C** et VRAM < 5 % à partir de la tête de sortie de la répétition 1 (décision provisoire de Fable : la jonction au repos était à 53 °C).
  - Un seul processus GPU à la fois.
- **Limites** :
  - Poids et activations remplis de zéros (`upload_zeros`), pas d'octets aléatoires.
  - La quantification d'activation n'est pas lancée par le bench, donc pas comptée.
  - Dispersion entre répétitions : < 2 % presque partout (les crochets des tableaux).
- **Données** : `impl/mmqf/` (p_*.json par répétition, logs, W_*.txt, llama_ref.txt).

Coût de vérification = Σ (temps par forme × nombre de tenseurs de cette forme dans Qwen3.8-27B), en µs, médiane des 3 répétitions [écart max − min].

## sm_86 (RTX 3090)
| dtype | m | C0 (MMVQ batched) | C1 x8 | C2 x16 | C3 x64 | C4 wave64 |
|---|---|---|---|---|---|---|
| Q5_K | 5 | **7 127** [62] | 23 343 | 25 756 | 42 042 | 79 389 |
| Q5_K | 9 | **15 775** [180] | 32 481 | 25 839 | 42 173 | 117 439 |
| Q5_K | 16 | **20 155** | 33 309 | 26 176 | 42 452 | 118 204 |
| Q4_K | 5 | **5 661** [26] | 9 089 | 13 304 | 31 767 | 52 510 |
| Q4_K | 8 | **7 932** | 9 486 | 13 363 | 31 887 | 74 212 |
| Q4_K | 9 | **12 307** [26] | 15 115 | 13 612 [113] | 32 013 | 78 442 |
| Q4_K | 16 | 16 040 [550] | 15 660 | **13 888** [74] | 32 282 | 80 626 |
| Q3_K | 5 | **500** | 1 366 | 1 582 | 3 353 | 4 352 |
| Q3_K | 9 | **1 017** | 2 296 | 1 594 | 3 385 | 6 296 |

(Tous les m sont dans `impl/mmqf/W_sm86.txt`.)

- **À m = 5 comme à m = 9, aucun candidat existant ne bat C0**, sur aucun dtype ni aucune forme. Écart du meilleur candidat à m = 5 : Q5_K ×3,3, Q4_K ×1,6, Q3_K ×2,7.
- **Seule exception** : Q4_K à m = 16, où C2 bat C0 de 13 % (13 888 contre 16 040). Ce m est hors des deux points de décision.

**Même forme que llama.cpp** (rows × k = 4096 × 14336, µs par appel ; llama.cpp = `test-backend-ops perf`, quantification de l'activation comprise) :
| dtype | m | C0 | llama.cpp |
|---|---|---|---|
| Q4_K | 2 / 3 / 4 | 45,9 / 50,1 / 52,4 | 44,2 / 52,6 / 45,3 |
| Q4_K | 5 / 8 | 56,9 / 82,4 | 45,6 / 46,0 |
| Q4_K | 9 / 16 | 126,5 / 165,5 | 49,8 / 50,5 |
| Q5_K | 4 / 5 / 8 | 58,5 / 63,6 / 91,7 | 54,0 / 54,0 / 55,1 |
| Q3_K | 4 / 5 / 8 | 77,7 / 61,2 / 82,3 | 44,9 / 45,7 / 46,2 |

- Sur ces formes, llama.cpp reste plat de m = 4 à m = 16 (≈ 45 à 50 µs). C0 croît avec m.
- À m = 9, C0 enchaîne n8 puis n1 et relit donc deux fois les poids : 126,5 µs contre 82,4 à m = 8.

**Anomalie Q3_K batched sur sm_86, non diagnostiquée** :
- n4 est plus lent que n5 : 17408 × 5120 à 109,1 µs contre 69,3 ; 4096 × 14336 à 77,7 contre 61,2.
- `cuobjdump` : n1 à n4 = 128 registres, n5 à n8 = 168, 0 spill des deux côtés. Ce n'est donc ni un spill ni une question de registres.

### Décision sm_86 (règle du plan)
**Port de `mul_mat_q_few`**, dans l'ordre Q5_K, Q4_K, Q3_K, selon la section « Sinon » du plan. Aucune ligne de dispatch à ajouter avec les noyaux existants. Le gain possible se mesure sur la forme commune : à m = 5, Q4_K 56,9 → 45,6 µs et Q5_K 63,6 → 54,0 µs.

## gfx906 (MI50, GPU 1)
C0 est le chemin de production tel quel :
- Q4_K et Q5_K, m = 2..8 : raccourci row-tile batched (`mmvq_*_dp4a_q8_1_batched_nX`) ; m ≥ 9 : x8 de la table ;
- Q3_K, m = 2 : 2 lancements `mmvq_q3_k_r2` (ligne `m = 1..2`) ; m = 3..16 : x8.

| dtype | m | C0 | C1 x8 | C2 x16 | C3 x64 | C4 wave64 |
|---|---|---|---|---|---|---|
| Q5_K | 2 | **18 713** | 33 741 | 52 374 | 117 825 | 52 749 |
| Q5_K | 4 | **22 782** | 33 851 | 52 473 | 118 118 | 139 120 |
| Q5_K | 5 | **26 001** [8] | 33 911 [19] | 52 502 | 118 297 | 165 678 |
| Q5_K | 6 | **26 017** | 33 931 | 52 541 | 118 393 | 200 841 |
| Q5_K | 8 | 37 376 [7] | **34 385** [34] | 53 228 | 119 848 | 265 071 |
| Q5_K | 9 | 47 605 (= C1) | **47 614** | 53 646 | 120 050 | 291 150 |
| Q5_K | 16 | 48 061 (= C1) | **48 048** | 55 500 | 120 884 | 489 525 |
| Q4_K | 2 | **17 638** | 23 953 | 30 989 | 93 237 | 28 166 |
| Q4_K | 4 | **21 746** [1] | 23 993 [23] | 31 074 | 93 556 | 100 299 |
| Q4_K | 5 | 24 167 [6] | **24 012** [12] | 31 078 | 93 749 | 125 915 |
| Q4_K | 6 | 26 536 | **24 015** | 31 105 | 93 849 | 154 395 |
| Q4_K | 8 | 33 965 | **24 073** | 31 213 | 94 816 | 200 549 |
| Q4_K | 9 | 40 976 (= C1) | 40 962 [18] | **31 268** [15] | 95 075 | 218 979 |
| Q4_K | 16 | 41 521 (= C1) | 41 518 | **32 224** | 95 668 | 420 716 |
| Q3_K | 2 | 3 105 [1] | **2 054** [2] | 3 656 | 8 954 | 2 660 |
| Q3_K | 5 | 2 071 (= C1) | **2 071** | 3 676 | 9 022 | 7 090 |
| Q3_K | 9 | 3 685 (= C1) | 3 685 | **3 680** | 9 141 | 22 126 |
| Q3_K | 16 | 3 684 (= C1) | **3 683** | 3 870 | 9 251 | 35 711 |

(Tous les m sont dans `impl/mmqf/W_gfx906.txt`.)

### Row-tile (C0) contre x8 (C1), m = 2..8
**Q5_K :**
- row-tile meilleur à m = 2..6 : −23 % à m = 5 ;
- x8 meilleur à m = 8 : −8 % ;
- m = 7 non mesuré.

**Q4_K :**
- row-tile meilleur à m = 2..4 : −9 % à m = 4 ;
- à m = 5, x8 est en avance de 0,6 %, ce qui dépasse la dispersion de 6 à 12 µs ;
- x8 meilleur à m = 6 (−9,5 %) et à m = 8 (−29 %).

**Le gagnant dépend de la forme** (µs, m = 5) :
- x8 gagne sur les matrices hautes à k = 5120 :
  - Q4_K 10240 × 5120 : 158,6 contre 204,6 ;
  - Q5_K 10240 × 5120 : 171,7 contre 194,6 ;
  - Q4_K 17408 × 5120 : 303,8 contre 329,3 ;
- le row-tile gagne sur k = 17408, k = 6144 et rows = 1024 :
  - Q5_K 5120 × 17408 : 339,2 contre 504,2 ;
  - 1024 × 5120 : 45 à 48 contre 120 à 143.

### x16 (C2) contre deux tuiles x8 (C1), m = 9..16
- **Q4_K** : C2 meilleur, de −24 % à m = 9 à −22 % à m = 16.
- **Q5_K** : C1 meilleur (C2 coûte +13 %).
- **Q3_K** : égalité à m = 9, C1 meilleur au-delà.

### Q3_K, m = 2
La ligne `m = 1..2` (r2) lance 2 noyaux. Elle coûte 3 105 µs, contre **2 054** pour x8 (−34 %) et 2 660 pour C4.

### Décision gfx906 (bandes seulement, aucun nouveau noyau)
1. **Q4_K, m = 9..16 et plus** : ligne `qmatmul_q4_K_mmq_lds_tiled_y64_x16`, avec cert et all-dispatch. C'est la seule bande x16 justifiée.
   - Sa borne haute reste à mesurer au-delà de 16 : la ligne x8 couvre aujourd'hui 3..32.
   - La cellule x16 sort du cfg bench pour gfx906.
2. **Suppression du raccourci row-tile pour Q4_K et Q5_K**, remplacé par de vraies lignes de table :
   - Q5_K : `mmvq batched` à m = 2..6, x8 à partir de 7 ou 8 (m = 7 est à mesurer) ;
   - Q4_K : `mmvq batched` à m = 2..4, x8 à partir de 5.
   - L'effet dépend de la forme, il faudrait donc une bande `n` (rows) si la table la porte. Sinon, on prend les frontières pondérées ci-dessus, qui sont celles du modèle cible.
   - Les autres dtypes du raccourci (Q6_K, IQ4_XS, IQ3_S, IQ3_XXS, IQ2_S) n'ont pas été mesurés. Il faut les passer au bench (C0 contre C1) avant de supprimer le raccourci pour eux.
3. **Q3_K, m = 2** : la bande x8 (aujourd'hui 3..32) passe à 2..32, et la ligne r2 se limite à `m = 1`.

## Tête de sortie Q4_K, m = 1 (gfx906), C0 contre C5
| forme | C0 (`mmvq_q4_k_dp4a_t128_vdr2`, ligne m = 1) | C5 (`mmvq_q4_k_r4`, chemin codé en dur) |
|---|---|---|
| 262144 × 3840 | **1 686,7** [1,0] | 3 454,9 [0,9] |
| 131072 × 5120 | **1 158,7** [9,2] | 2 234,4 [0,5] |

C5 est 2,0 fois plus lent que la ligne de table. **Décision : supprimer la branche `use_r4_lmhead` d'`output_head`** (elle tombe alors sur la ligne m = 1), et retirer `Ops::mmvq_q4_k_r4` s'il n'a plus d'appelant. Cela corrige aussi le HTTP 500 de gemma Q4_0 sur sm_86 (gemma-sm86-diag.md (1)).

## Suites (rien n'est codé)
- **Commit des raccourcis**, une fois que Fable l'aura validé :
  - décisions gfx906 1 à 3 et suppression de C5 ;
  - bench C0 contre C1 sur Q6_K et les i-quants du raccourci ;
  - mesure de m = 7 et de m > 16 pour les bandes x16.
- **sm_86** : port de `mul_mat_q_few` (plusieurs sessions). Rien n'est projeté au-delà des écarts mesurés ici.
- **Anomalie Q3_K batched n3/n4 sur sm_86** : à profiler (ncu), indépendamment du port.

## Anomalie Q3_K batched sur sm_86 (n4 plus lent que n5) : lecture seule, hypothèse
**Le chemin de lancement est identique d'une entrée à l'autre.** Il est décrit dans `instantiations.toml` (`mmvq_q3_k_dp4a_t128_vdr2_batched`) et `mmvq_vdr_batched.cuh` :
- les 8 entrées (n1 à n8) appellent le même gabarit `mmvq_vdr_batched_row<Q3K, N, 32, 4>` ;
- un bloc fait 32 threads (`launch_bounds 32`) et couvre 4 lignes, la grille compte `ceil(rows / 4)` blocs ;
- le noyau suit le schéma VDR 2 de llama.cpp : 8 threads par super-bloc, 4 super-blocs par passage ;
- seule N (le nombre de colonnes) change. C0 lance n3, n4 ou n5 selon m, sans autre différence de géométrie.

**`ptxas -v` (sm_86)** : aucun spill nulle part ; n1 à n4 utilisent 128 registres, n5 à n8 en utilisent 168.

**SASS**, en LDG / IDP / branchements par entrée :

| entrée | LDG | IDP | branchements |
|---|---|---|---|
| n3 | 96 | 192 | 3 |
| n4 | 108 | 256 | 3 |
| n5 | 120 | 320 | 6 |

Le nombre de chargements croît de 12 par colonne : il n'y a pas de déroulage double de la boucle K.

Ce qui change, c'est l'ordre dans le corps de boucle :
- **n5** émet **78 chargements d'affilée** avant le premier `IDP`, puis calcule ;
- **n4** n'en émet que **51**, puis entrelace le reste avec le calcul (paquets de 18, 9, 3, 2, 1…). Chaque paquet attend donc la mémoire avant le calcul suivant.

**Hypothèse** :
- À 128 registres, ptxas ne peut pas garder en vol tous les chargements d'un passage (poids des 4 lignes et activations des N colonnes). Il les étale, et la latence mémoire se retrouve exposée à chaque paquet.
- À N ≥ 5, il passe à 168 registres et émet presque tout le passage d'un coup.
- Avec des blocs d'un seul warp, un SM ne porte de toute façon que 16 blocs (sm_86) : l'occupation est déjà bridée à 16 warps par le nombre de blocs (12 à 168 registres). Plus de registres coûte donc peu d'occupation et rapporte du parallélisme mémoire.
- C'est cohérent avec la mesure : n4 à 109,1 µs contre n5 à 69,3 µs sur 17408 × 5120 ; n3 à 76,9 ; n2 à 54,3 (deux colonnes tiennent encore dans 128 registres).

**Test qui trancherait** (un nvcc isolé et un bench, au prochain créneau RTX) :
- `-maxrregcount` ne sert à rien ici : c'est un plafond, et ptxas a déjà jusqu'à 255 registres disponibles. Les deux leviers réels sont :
  - 2 lignes par bloc (R = 2), ce qui divise par deux les registres de poids ;
  - un préchargement explicite des poids du passage suivant dans `mmvq_vdr_batched_row` ;
- on recompile n3 et n4 avec l'un des deux ;
- vérifier dans le SASS que les ≈ 78 chargements sont groupés ;
- remesurer C0 à m = 3, 4 avec qmatmul-bench.

Si n4 tombe vers 70 µs, l'hypothèse est confirmée. La correction devient alors un choix de géométrie (R) dans `instantiations.toml`, avec son cert. Elle reste sans objet si la bande few (F2) couvre m ≥ 3 en Q3_K (F4).

## F2 : C0 (MMVQ batched) contre C6 (few MMA Q4_K), sm_86, v3 11db931c (bin_V3_CUDA)
- **Conditions** : qmatmul-bench sous nsys, 3 répétitions, garde RTX ≤ 50 °C. Temps noyau par appel, découpés par `seg2.py`. Données dans impl/mmqf/p_c6_*.json.
- **Cert** : C6 est certifié sur sm_86, avec une erreur max de 8,4e-4 pour une tolérance de 3e-2.

**Coût de vérification Q4_K de Qwen3.8-27B, pondéré par le nombre de tenseurs** (µs, médiane des 3 répétitions [écart]) :

| m | C0 | C6 | C6 / C0 |
|---|---|---|---|
| 2 | **4 570** [5] | 4 703 [16] | 1,029 |
| 3 | 4 860 [13] | **4 695** [17] | 0,966 |
| 4 | 5 334 [39] | **4 702** [10] | 0,882 |
| 5 | 6 030 [19] | **4 713** [10] | 0,782 |
| 6 | 6 879 [72] | **4 722** [15] | 0,686 |
| 8 | 8 532 [145] | **4 739** [22] | 0,555 |
| 9 | 13 064 [195] | **5 832** [95] | 0,446 |
| 12 | 13 912 [255] | **5 866** [18] | 0,422 |
| 16 | 17 273 [49] | **5 933** [14] | 0,343 |
| 17 | 21 626 [125] | **6 163** [17] | 0,285 |
| 24 | 25 906 [282] | **6 289** [36] | 0,243 |

**Même forme que llama.cpp** (4096 × 14336, µs par appel ; llama.cpp `test-backend-ops`, mesuré plus haut) :

| m | C0 | C6 | llama.cpp |
|---|---|---|---|
| 2 | 46,0 | 42,5 | 44,2 |
| 4 | 53,3 | 42,9 | 45,3 |
| 5 | 59,2 | **43,2** | 45,6 |
| 8 | 87,4 | **43,2** | 46,0 |
| 9 | 131,5 | 51,1 | **49,8** |
| 16 | 177,6 | 51,8 | **50,5** |

- **Lecture** :
  - C6 est plat par groupe de 8 colonnes : il lit une seule fois les poids pour 1 à 8 colonnes, puis pour 9 à 16 colonnes. C0 croît avec m.
  - Contre llama.cpp : C6 est devant à m ≤ 8 (−5 % à m = 5 et 8). Il est 2,6 % derrière à m = 9..16, où llama.cpp prend la tuile de 32 lignes (`nmt = 2`) que F2 n'instancie pas (spill).
- **Bande proposée sur sm_86, Q4_K** : few MMA de **m = 3 à 24** ; m = 2 reste en MMVQ batched (+2,9 % pour C6). La ligne de dispatch elle-même attend le branchement du forward (le buffer few dans les 7 sites, l'étape « câblage »).
- **Suite** : F3 (Q5_K) et F4 (Q3_K), sur la même structure. La tuile de 32 lignes (spill) est le levier mesurable pour m = 9..16.

## F3/F4 : C0 (MMVQ batched) contre C6 (few MMA), Q5_K et Q3_K, sm_86, v3 b0a8d697 (bin_V4_CUDA)
Même protocole que F2 (nsys, seg2.py, 3 répétitions, formes complètes de Qwen3.8-27B sur un GPU ; noyau seul, quantification non comptée).
Coût de vérification pondéré par le nombre de tenseurs (µs, médiane [écart]) :

| m | Q5_K C0 | Q5_K C6 | Q3_K C0 | Q3_K C6 |
|---|---|---|---|---|
| 2 | **6 102** [9] | 6 299 [6] | **379** [1] | 402 [0] |
| 3 | 6 361 [21] | **6 304** [11] | 554 [0] | **403** [1] |
| 4 | 6 784 [40] | **6 313** [12] | 803 [1] | **403** [0] |
| 5 | 7 510 [162] | **6 331** [9] | 542 [1] | **404** [9] |
| 6 | 8 339 [101] | **6 350** [10] | 606 [12] | **405** [6] |
| 8 | 10 600 [197] | **6 366** [18] | 741 [20] | **406** [8] |
| 9 | 16 497 [194] | **7 161** [49] | 1 086 [22] | **538** [23] |
| 12 | 17 307 [95] | **7 251** [79] | 1 527 [44] | **508** [4] |
| 16 | 21 446 [419] | **7 297** [17] | 1 430 [20] | **525** [2] |
| 17 | 27 339 [105] | **7 595** [68] | 1 790 [5] | **572** [2] |
| 24 | 32 004 [157] | **7 713** [22] | 2 178 [5] | **574** [3] |

- C6 est plat par groupe de 8 colonnes, comme pour Q4_K. C0 gagne à m = 2 sur les deux dtypes.
- Q5_K à m = 3 : C6 en avance de 0,9 % seulement (57 µs, écart 11 à 21), noyau seul. Une deuxième quantification (site mixte) peut l'annuler.
- Bandes proposées (aucune ligne avant accord, et après le v4 du Vibe qui mesure le coût réel de la ligne Q4_K 3..24 quantification comprise) :
  - Q3_K few : 3..24 ;
  - Q5_K few : 4..24 (m = 3 laissé au batched, marge trop faible) ;
  - à revoir si v4 montre que la double quantification mange le gain Q4_K.
