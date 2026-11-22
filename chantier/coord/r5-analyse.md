# r5 — analyse de la trace nsys (Fable, 20/11)

Requête « math », 200 tokens, sans MTP, RTX 3090 froide. Source : r5/fl.sqlite, r5/lc.sqlite.

## Le graphe n'est pas la cause principale
- flambeau : 9,60 s côté client, **9,07 s de noyaux GPU** (GPU occupé ≈ 94 %). llama.cpp : 5,07 s, 4,54 s de noyaux.
- L'écart vient du **temps des noyaux, ×2,0**, pas des trous entre lancements. Le graphe CUDA (r5-graphe-qwen35.md) récupère au plus ≈ 0,5 s sur 9,6 (≤ 5 %).

## MMVQ : ≈ 8,3 s chez flambeau contre ≈ 3,8 s chez llama.cpp (91 % du temps GPU flambeau)
Même forme, même type, µs par appel (flambeau / llama.cpp sans fusion) :
| type | lignes | flambeau | llama.cpp | rapport |
|---|---|---|---|---|
| IQ4_XS | 17408 | 115,4 (256 thr) | 55,8 | **×2,07** |
| Q5_K | 17408 | 121,1 (256 thr) | 71,4 | ×1,70 |
| Q5_K | 5120 | 65,8 | 43,2 (fusionné) | ×1,5 |
| Q6_K | 248320 (tête) | 1224 (64 thr) | 1175 | ×1,04 |
| Q6_K | 5120 | 41,8 | 37,5 (fusionné) | ×1,1 |
| Q4_K r2 | 17408 (grille 8704) | 95,6 | 59,1 | ×1,6 |
llama.cpp : blocs 32×4 = 128 threads, une ligne par bloc. flambeau : 256 threads par ligne pour IQ4_XS / Q5_K / Q3_K, 64 threads r2 pour Q4_K. Q6_K est à parité : le défaut est propre aux noyaux IQ4_XS, Q5_K, Q4_K, Q3_K sur sm_86. Même diagnostic que R10b sur gfx906 (répartition des poids par thread).

## Fusion gate+up (llama.cpp `mul_mat_vec_q<…, has_fusion=1>`)
llama.cpp fait gate et up dans un seul noyau (IQ4_XS 17408 : 113 µs pour les deux, contre 2 × 115 µs chez flambeau). Deuxième levier, après la répartition.

## Ordre proposé
1. R10b sur sm_86 pour IQ4_XS, Q5_K, Q4_K, Q3_K (puis IQ4_NL, IQ3_S) : microbanc par sweep aux formes 17408×5120 et 5120×17408, cible ≤ temps llama.cpp ci-dessus ; lignes de dispatch sm_86 et certificats.
2. Fusion gate+up sm_86.
3. Graphe CUDA qwen35 (≤ 5 %).
Gain attendu si (1) atteint llama.cpp : ≈ 4 s sur 9,1 s de GPU → ≈ 40 t/s sans MTP (estimation, à mesurer).
