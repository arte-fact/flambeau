# r10m — contrôle du MMVQ batché m ≥ 2 sur sm_86 (branche r10b-sm86, HEAD 81632285 ; binaires impl/bin_R10m, impl/bin_R10m_CUDA)

Règles habituelles : un seul processus GPU à la fois. Gardes : MI50 jonction ≤ 50 °C et VRAM < 5 % ; RTX ≤ 50 °C et < 1 Go. Arrêt par SIGTERM. Relever horloges et températures avant et après chaque run. Ne rien modifier dans les worktrees. Vérifier que la worktree de r10b-sm86 (`git worktree list`) a HEAD = 81632285. Dispatch figé : `git show 81632285:dispatch/…` → r10m/.

## A — passe complète
`sweep --all-dispatch` sur gfx906 (bin_R10m, GPU 1) ; attendu 86/86. Puis sur sm_86 (bin_R10m_CUDA) ; attendu 27/27 (20 + 7 lignes batchées). Lister toute ligne non PASS.

## B — RTX, slots 1, kv q8, lp.py (même protocole que r10s/B)
Ordre alterné, garde ≤ 50 °C avant chaque serveur :

| run | binaire | mtp | réf |
|---|---|---|---|
| M1 | bin_R10s_CUDA | 0 | 40,83 |
| M2 | bin_R10m_CUDA | 0 | — |
| M3 | bin_R10s_CUDA | 4 | 45,50 |
| M4 | bin_R10m_CUDA | 4 | — |

Pour M3 et M4 : acceptation et tokens par cycle. Pour M2 et M4 : premier caractère de divergence de chaque invite par rapport à M1 et M3, sans interprétation. Relever aussi le TTFT moyen par invite pour les quatre runs (le préremplissage passe désormais par le batché).

## C — MI50, non-régression rapide
bin_R10m, mtp 0, lp.py. Référence r10s/C : 29,79. Comparer les textes à r10s.

## D — trace nsys MTP-4 (même protocole que r10s/mtp)
bin_R10m_CUDA, mtp 4, invite « math », 200 tokens. À relever :
- la durée client et le temps GPU total des noyaux ;
- le top 15 des noyaux ;
- le coût par cycle hors préremplissage (réf. 75,1 ms), avec la part de la vérification (réf. 53 ms) ;
- le temps GPU du préremplissage (67 tokens).

## Rendu
Écrire r10m/RESULTATS.md, puis « r10m terminé ».
