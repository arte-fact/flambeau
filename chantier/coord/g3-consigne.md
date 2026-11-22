# g3 — argmax glouton sur GPU en mtp 0 (branche g3-device-argmax ; a8ccae8c = (a) seul ; eba8486f = (a) + pipeline (b))

Binaires :
- impl/bin_G3a, impl/bin_G3a_CUDA, sur a8ccae8c, candidat au push ;
- impl/bin_G3, impl/bin_G3_CUDA, sur eba8486f, pour documenter le résultat nul de (b).

Règles habituelles : un seul processus GPU à la fois ; gardes MI50 (jonction ≤ 50 °C, VRAM < 5 %) et RTX (≤ 50 °C, < 1 Go) ; arrêt par SIGTERM ; températures relevées avant et après. Ne rien modifier dans les worktrees. Vérifier les hash dans bin_G3a.txt et bin_G3.txt.

## A — certificats
all-dispatch avec bin_G3a : gfx906, attendu 86/86 ; sm_86, attendu 27/27.

## B — RTX, slots 1, kv q8, ordre alterné
| run | binaire | mtp | script | réf |
|---|---|---|---|---|
| L1 | bin_G2_CUDA | 0 | lp_g.py | 42,91 |
| L2 | bin_G3a_CUDA | 0 | lp_g.py | — |
| L3 | bin_G3_CUDA | 0 | lp_g.py | — |
| L4 | bin_G2_CUDA | 0 | lp.py (pénalité 1,05) | — |
| L5 | bin_G3a_CUDA | 0 | lp.py | — |
| L6 | bin_G3a_CUDA | 4 | lp_g.py | 78,21 |
| L7 | bin_G3a_CUDA | 0 | lp_g.py | répétition de L2 |
Textes : L2 et L3 doivent être identiques à L1, L5 à L4, et L6 à g2/K4. Pour L6 : acceptation et tokens par cycle.

## C — MI50 (TP-2, kv q8, mtp 0, lp_g.py)
Un run bin_G3a, réf. g2/C 29,57 et g1/H1 29,75 ; textes comparés à g1/H1.

## Rendu
g3/RESULTATS.md, puis « g3 terminé ».
