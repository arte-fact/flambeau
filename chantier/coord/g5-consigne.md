# g5 — attention de décodage vec_dpos sur sm_86 (branche g5-attn-vec, HEAD ee32a49c ; impl/bin_G5, impl/bin_G5_CUDA) + contrôle stop-trim (impl/bin_ST, impl/bin_ST_CUDA)

Règles habituelles : un seul processus GPU à la fois ; gardes MI50 (jonction ≤ 50 °C **sur le GPU utilisé**, VRAM < 5 %) et RTX (≤ 50 °C, < 1 Go) ; SIGTERM ; températures avant et après. Note : au repos, la jonction GPU 0 reste à 51 °C ; ne garder que le GPU utilisé. Ne rien modifier dans les worktrees. Vérifier les hash dans bin_G5.txt et bin_ST.txt. Dispatch figé `git show ee32a49c:…` → g5/.

## A — certificats (bin_G5)
all-dispatch gfx906 : attendu 86/86. sm_86 : attendu 29/29 (ou le nombre annoncé dans bin_G5.txt). Lister toute ligne non PASS.

## B — RTX, slots 1, kv q8, lp_g.py, ordre alterné
| run | binaire | mtp | réf |
|---|---|---|---|
| V1 | bin_G4_CUDA | 0 | 44,11 |
| V2 | bin_G5_CUDA | 0 | — |
| V3 | bin_G4_CUDA | 4 | 78,36 |
| V4 | bin_G5_CUDA | 4 | — |
| V5 | bin_G5_CUDA | 0 | répétition V2 |
Textes : V2 contre V1 et V4 contre V3 peuvent diverger (ordre des sommes) : donner le premier caractère de divergence, sans interpréter. V5 = V2 attendu. MTP : acceptation et tokens par cycle.

## C — RTX, contexte long (le gain attendu y est plus fort)
Invite d'environ 4 000 tokens (prendre un texte long existant dans flambeau-exp, ou concaténer les 5 invites ×N), 400 tokens générés, lp_g, mtp 0 : bin_G4_CUDA puis bin_G5_CUDA, puis à nouveau bin_G4_CUDA. Relever t/s hors TTFT et TTFT.

## D — MI50 non-régression (bin_G5, TP-2, kv q8, mtp 0, lp_g)
Réf. g4/C 29,87 ; textes comparés à g1/H1.

## E — stop-trim (bin_ST_CUDA, RTX, kv q8, mtp 0)
lp_g : débit (réf. de la base de la branche, voir bin_ST.txt) et textes comparés au binaire de base ; seuls des espaces ou sauts de ligne finaux peuvent différer. Relever chaque différence.

## Rendu
g5/RESULTATS.md, puis « g5 terminé ».
