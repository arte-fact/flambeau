# v3 — contrôle des textes et des débits, V2 contre V3 (impl/bin_V2, impl/bin_V2_CUDA, impl/bin_V3, impl/bin_V3_CUDA)

Commencer seulement quand j'annonce que le GPU est libre (Opus finit d'abord un bench RTX). Vérifier les hash et la HEAD (50864a2f) dans impl/bin_V3.txt. Figer le dispatch avec `git -C ~/Workspace/flambeau-impl-o show 50864a2f:…` → v3/.

Règles habituelles : un seul processus GPU à la fois, arrêt par SIGTERM, températures notées avant et après chaque run. Gardes RTX : ≤ 50 °C et < 1 Go (sd-server : ne pas y toucher, VRAM par processus). Gardes MI50 : GPU0 et GPU1 ≤ 56 °C, VRAM < 5 %. Protocole lp_g.py, kv q8.

**Textes attendus :**
- RTX : identiques à V2 partout (aucune ligne sm_86 ne change) ;
- MI50 : les noyaux Q4_K, Q5_K et Q3_K changent pour m = 2..32 (prefill court, vérification MTP). Des différences sont possibles ; si un texte diverge, donner l'invite et le premier caractère divergent, sans interpréter.

## B — RTX, slots 1, ordre alterné
| run | binaire | mtp | réf |
|---|---|---|---|
| Y1 | bin_V2_CUDA | 0 | 44,70 |
| Y2 | bin_V3_CUDA | 0 | — |
| Y3 | bin_V2_CUDA | 4 | 82,49 |
| Y4 | bin_V3_CUDA | 4 | — |

Textes : Y2 = Y1, Y4 = Y3 attendus. MTP-4 : acceptation et tokens par cycle.

## D — MI50 (TP-2), ordre alterné
- mtp 4 : V2, V3, V2, V3 ; acceptation et tokens par cycle ; textes V3 comparés à V2 ;
- mtp 0 : V2, V3 ; textes comparés à s7/lp_d1.json.

## E — autres modèles, mtp 0, V2 puis V3
- ornith-1.0-35b-Q4_K_M sur MI50 TP-2 : textes identiques attendus aux bandes près (voir plus haut) ;
- gemma-4-12b-it-Q4_0 sur la RTX : en V2, HTTP 500 attendu (`mmvq_q4_k_r4` not loaded) ; en V3, 5 textes attendus. Le refaire sur MI50 TP-2 en V2 puis V3 et comparer les textes (la tête Q4_K y passe de r4 à qmatmul).

## Rendu
v3/RESULTATS.md (moyennes, écarts en %, section « Erreurs »), puis « v3 terminé ».
