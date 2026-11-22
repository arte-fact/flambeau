# s1 — contrôle de l'échantillonnage MTP sur GPU (branche s1-draft-gpu-sampling, HEAD 993a8f2f ; binaires impl/bin_S1, impl/bin_S1_CUDA)

Règles habituelles : un seul processus GPU à la fois. Gardes : MI50 jonction ≤ 50 °C et VRAM < 5 % ; RTX ≤ 50 °C et < 1 Go. Arrêt par SIGTERM ; relever horloges et températures avant et après chaque run. Ne rien modifier dans les worktrees. Vérifier dans `git worktree list` que la branche s1-draft-gpu-sampling est à HEAD = 993a8f2f.

## A — certificats
1. `sweep --all-dispatch` sur gfx906 (bin_S1, GPU 1) : attendu 86/86.
2. `sweep --all-dispatch` sur sm_86 (bin_S1_CUDA) : attendu 27/27.
3. Sweep de `sampler_argmax_f32` sur les deux arch (impl DIRECT_CALL, hors dispatch ; commande dans le commit 993a8f2f ou à demander) : attendu PASS.

## B — RTX, slots 1, kv q8, lp.py, ordre alterné
| run | binaire | mtp | réf |
|---|---|---|---|
| P1 | bin_R10m_CUDA | 0 | 40,88 |
| P2 | bin_S1_CUDA | 0 | — |
| P3 | bin_R10m_CUDA | 4 | 71,97 |
| P4 | bin_S1_CUDA | 4 | — |

Pour P3 et P4, relever l'acceptation et les tokens par cycle. Textes : P2 doit être identique à P1, et P4 à P3 ; donner le premier caractère de divergence s'il y en a.

## C — MI50 (le chemin MTP HIP change), TP-2, kv q8, lp.py, ordre alterné
| run | binaire | mtp | réf |
|---|---|---|---|
| Q1 | bin_R10m | 0 | 29,80 |
| Q2 | bin_S1 | 0 | — |
| Q3 | bin_R10m | 4 | — |
| Q4 | bin_S1 | 4 | — |

Pour Q3 et Q4, relever l'acceptation et les tokens par cycle. Textes : Q2 doit être identique à Q1, et Q4 à Q3.

## D — trace nsys MTP-4 RTX (même protocole que r10m/D)
bin_S1_CUDA, invite « math », 200 tokens. À relever :
- la durée client et le temps GPU total des noyaux ;
- le coût par cycle hors préremplissage (réf. 39,0 ms de noyaux, cycle mur 47,7 ms) ;
- le temps GPU inactif par cycle (réf. 8,6 ms) ;
- le top 15 des noyaux.

## Rendu
s1/RESULTATS.md, puis « s1 terminé ».
