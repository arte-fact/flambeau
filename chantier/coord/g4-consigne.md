# g4 — grille split-K figée sous capture (branche g4-grid-baked, HEAD 2661d7bf ; binaires impl/bin_G4, impl/bin_G4_CUDA)

Règles habituelles : un seul processus GPU à la fois ; gardes MI50 (jonction ≤ 50 °C, VRAM < 5 %) et RTX (≤ 50 °C, < 1 Go) ; arrêt par SIGTERM ; températures relevées avant et après chaque run. Ne rien modifier dans les worktrees. Vérifier que le hash de bin_G4.txt et HEAD de g4-grid-baked valent tous deux 2661d7bf. Dispatch figé avec `git show 2661d7bf:…` → g4/.

## A — certificats
all-dispatch : gfx906 (bin_G4), attendu 86/86 ; sm_86 (bin_G4_CUDA), attendu 27/27.

## B — RTX, slots 1, ordre alterné
| run | binaire | kv | mtp | script | réf |
|---|---|---|---|---|---|
| N1 | bin_G3a_CUDA | q8 | 0 | lp_g.py | 43,78 |
| N2 | bin_G4_CUDA | q8 | 0 | lp_g.py | — |
| N3 | bin_G3a_CUDA | q8 | 0 | lp.py | 43,64 |
| N4 | bin_G4_CUDA | q8 | 0 | lp.py | — |
| N5 | bin_G4_CUDA | q8 | 4 | lp_g.py | 78,44 |
| N6 | bin_G4_CUDA | q8 | 0 | lp_g.py | répétition de N2 |
| N7 | bin_G4_CUDA | f16 | 0 | lp_g.py | (pas de réf. récente : lancer aussi bin_G3a_CUDA en f16 juste avant, run N7a) |

Textes attendus : N2 = N1, N4 = N3, N5 = g3/L6, N7 = N7a. Pour N5 : acceptation et tokens par cycle.

## C — MI50 (TP-2, kv q8, mtp 0, lp_g.py)
Un run bin_G4. Réf. g3/C 29,72. Textes comparés à g1/H1.

## Rendu
g4/RESULTATS.md, puis « g4 terminé ».
