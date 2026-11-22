# nsys d'un cycle MTP-4 sur RTX 3090 : flambeau V1 contre llama.cpp HEAD

## Protocole
- **Modèle et matériel :** Qwen3.8-27B-UD-Q4_K_M sur une RTX 3090 (sm_86).
- **Requête :** invite « code » de lp_g, gloutonne, `max_tokens` 400 ; elle s'arrête à 203 tokens des deux côtés. Une requête de chauffe, puis `nsys start/stop` autour de la requête mesurée (`--trace=cuda,osrt,nvtx --cuda-graph-trace=node`).
- **flambeau :** `impl/bin_V1_CUDA serve --kv q8 --ctx-cap 8192 --mtp 4`. 43 cycles.
- **llama.cpp :** ffc87c3a7, build-cuda-hd, ligne D2 (`--spec-type draft-mtp --spec-draft-n-max 4`, q8_0). 44 cycles, 158 brouillons acceptés sur 176.
- **Référence S7 :** `bin_S7_CUDA`, même protocole.
- **Gardes :** RTX ≤ 50 °C et < 1 Go avant chaque serveur, un seul processus GPU, SIGTERM. Il ne reste aucun processus GPU (sd-server non touché).
- **Analyse :** les cycles sont découpés aux copies D2H qui marquent chaque cycle (flambeau : les 16 o des brouillons ; llama.cpp : les 4 966 400 o de logits de vérification). Les chiffres sont des moyennes sur les cycles stables (36 côté flambeau, 38 côté llama.cpp), en µs par cycle. Les rapports et les scripts sont dans `<scratchpad>/nsys_v1/` (fl, lc, s7 en `.nsys-rep` et `.sqlite`, plus flcyc.py, lccyc.py et bykern.py).
- **Biais :** le traçage CUPTI alourdit chaque lancement. flambeau lance environ 2 170 noyaux par cycle sans graphe, llama.cpp passe par des graphes. Les trous de lancement de flambeau sont donc probablement surestimés ici.

## Cycle : flambeau 43,55 ms, llama.cpp 40,66 ms, soit +2,89 ms (moyennes sous nsys)

| phase | flambeau (mur / GPU / trous) | llama.cpp (mur / GPU / trous) | écart mur |
|---|---|---|---|
| vérification n = 5 (cible + tête) | 33 945 / 32 158 / 1 786 | 27 461 / 27 374 / 141 | **+6 484** |
| relecture : ids + logits de la ligne bonus (fl) ; logits des 5 lignes (lc) | 404 / 209 / 194 | 1 051 / 1 051 / 0 | −647 |
| sélection GDN (fl) ; hôte entre brouillons et vérification (lc) | 188 / 16 / 172 | 847 / — / 847 | −659 |
| 4 brouillons (rattrapage compris) | 9 144 / 8 909 / 234 | 11 300 / 7 392 / 3 891 | −2 156 |

Toute l'avance de llama.cpp vient de la **vérification** : +6,5 ms, dont +4,8 ms de temps GPU des noyaux et +1,65 ms de trous de lancement. flambeau reprend 3,6 ms sur les brouillons et la relecture, parce que llama.cpp échantillonne sur l'hôte à chaque pas (environ 1 ms de trou par pas) et relit les logits des 5 lignes. La somme des phases donne 2,97 ms, contre 2,89 ms mesurés.

### Vérification : temps GPU par catégorie (µs par cycle)
| catégorie | flambeau | llama.cpp | écart | détail |
|---|---|---|---|---|
| matmul | 26 074 | 22 311 | **+3 763** | voir le tableau par dtype ci-dessous |
| attention | 1 681 | 353 | **+1 328** | fl : `attention_prefill_flash_tile_d256_br8_q8_kv` à n = 5, ≈ 99 µs par couche × 16 ; lc : `flash_attn_ext_f16`, ≈ 16 µs par couche |
| quantification des activations | 744 | 266 | +478 | fl lance `quantize_row_f16_q8_1` puis `quantize_f16_q8_1_mmq` sur la même entrée avant les MMVQ n5 |
| GDN | 1 553 | 1 217 | +336 | kstates 1 174 contre `gated_delta_net_cuda` 1 082 ; le reste vient de conv, l2, split et scale |
| autres | 1 098 | 2 525 | −1 427 | lc : concat_non_cont 659, k_bin_bcast 436, get_rows 330… |

### Matmul de la vérification par dtype (µs par cycle)
| dtype | flambeau (noyau) | llama.cpp (noyau) | écart |
|---|---|---|---|
| Q5_K | 8 081 (MMVQ batched n5) | 6 381 (`mul_mat_q_few`) | **+1 700** |
| Q4_K | 6 459 (MMVQ batched n5) | 4 737 (`mul_mat_q_few`) | **+1 722** |
| Q8_0 (ssm_alpha et ssm_beta : 96 lancements, grille 12) | 1 340, ≈ 14 µs chacun | 523 (`mul_mat_q_few`, grille 3), ≈ 5,4 µs | **+817** |
| Q3_K | 560 | 333 (`mul_mat_q_few`) | +227 |
| IQ3_S / IQ4_NL | 351 / 441 | 219 / 394 (`mul_mat_q_few`) | +132 / +47 |
| Q6_K (tête de vérification comprise) | 1 910 | 1 868 | +42 |
| IQ4_XS | 6 693 (MMVQ n5) | 6 882 (`mul_mat_vec_q`) | **−189** |

À n = 5 sur sm_86, llama.cpp fait passer les K-quants, Q8_0 et les IQ autres qu'IQ4_XS par une MMQ « few ». IQ4_XS y reste en `mul_mat_vec_q`, et c'est le seul dtype où flambeau fait jeu égal. L'écart tient donc au **type de noyau** (MMVQ batched contre MMQ few), pas à l'occupation du GPU en général.

### Brouillons : temps GPU (µs par cycle)
- **Tête Q6_K de 248 320 lignes, 4 par cycle :** 1 230 µs chacune contre 1 175 chez llama.cpp, soit +220 par cycle.
- **eh_proj :** stockée en Q6_K dans le GGUF (43 Mo). flambeau la fait passer en `dense_gemv_f16_f16_batched`, ≈ 10 appels de 97 µs, soit 968 µs par cycle sur des poids F16 deux fois plus gros (105 Mo). llama.cpp la garde en Q6_K.
- **Total GPU des brouillons :** 8 894 contre 7 392, soit +1 502, compensé par 3 657 µs de trous côté hôte chez llama.cpp.

## Référence S7 : la copie GDN supprimée par V1
| | S7 | V1 |
|---|---|---|
| cycle moyen / médian (µs) | 44 537 / 43 422 | 43 841 / 42 923 |
| copies d'état GDN par cycle (D2D de 3 Mio) | 81,2, soit 608 µs GPU | 1,3, soit 2 µs |
| noyaux GDN de l'étape d'état (µs) | 882 (rejeu compris) | 1 201 (écrit 5 états par couche) |
| `cuStreamSynchronize` sur la requête | 302 appels, 759 ms | 131 appels, 217 ms |

V1 retire 0,70 ms par cycle (−1,6 %), ce qui concorde avec les +1,4 % de débit mesurés par le Vibe (X4 contre X3). Le noyau kstates coûte +0,32 ms de GPU par cycle de plus que l'étape de S7.

## Les 256 Mio d'écart VRAM en mtp 4 (calcul −1 226, mesure −970)
- `mtp_gdn_kstates` est alloué sur `gdn_state.len()` (scratch.rs:1691).
- `gdn_state` contient **une entrée par couche** de `config.num_layers` (scratch.rs:1564), pas seulement pour les 48 couches GDN.
- Avec 65 couches (64 cibles + MTP, la même base que le snapshot de S7), on a 65 × 5 × 3 Mio = 975 Mio au lieu de 48 × 5 × 3 = 720 Mio, soit **+255 Mio**, contre 256 Mio mesurés.
- C'est mon erreur de calcul dans e6d1f1ea : les 17 couches non GDN reçoivent des K états jamais lus.
- Même cause, déjà présente dans S7 et V1 : l'état vivant et l'historique de conv sont aussi alloués pour les 17 couches non GDN, soit 17 × 3,12 Mio ≈ 53 Mio.

## Ordre des corrections proposé
Classé par écart mesuré contre llama.cpp sur ce cycle. Ce ne sont pas des gains projetés : chaque ligne demande un cert et une mesure.
1. **Matmul de vérification à n = 5 (K-quants, IQ3_S, IQ4_NL)** : une impl MMQ « few » (petit m, tuilage LDS, activations au format MMQ) pour 2 ≤ m ≤ 8 sur sm_86, choisie par une ligne de dispatch. Écart +3,6 ms (Q5_K, Q4_K, Q3_K) et +0,18 ms (IQ). IQ4_XS reste en MMVQ.
2. **Graphe CUDA de la vérification n = 5 et des pas de brouillon (V5)** : 1,65 ms de trous de lancement en vérification sous nsys. À remesurer hors traçage avant d'en tirer un chiffre.
3. **Attention de vérification à n = 5** : le noyau de préremplissage à tuiles coûte 99 µs par couche contre 16 µs. Il faut une impl décodage multi-lignes (split-K) pour n ≤ 8. Écart +1,33 ms.
4. **Q8_0 des petites matrices ssm_alpha et ssm_beta** (grille 12) : 14 µs par lancement contre 5,4 µs. Écart +0,82 ms. Pistes : fusionner alpha et beta en un seul lancement, ou une ligne de dispatch adaptée aux petits n_rows.
5. **eh_proj en Q6_K** (ne plus la déquantiser en F16) : 968 µs par cycle aujourd'hui, et 62 Mio de VRAM en moins.
6. **Quantification double des activations** avant les MMVQ n5 : +0,48 ms. Il faut vérifier si la sortie au format MMQ est lue sur ce chemin.
7. **Copie de 993 Ko de logits de la ligne bonus** (`logits_row_to_host`, model.rs) à chaque cycle, même en glouton sans logprobs : 209 µs de GPU et ≈ 400 µs de mur. À faire seulement si l'appelant en a besoin.
8. **VRAM des K états** : allouer seulement pour les couches GDN, avec un index dense. −255 Mio. C'est un correctif de V1 sans effet sur le débit, mais petit et sans noyau. On peut faire de même pour l'état vivant (−53 Mio).
