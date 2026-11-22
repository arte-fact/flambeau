# r10s — contrôle de R10b sur sm_86 (branche r10b-sm86, HEAD 1b31581f ; binaires impl/bin_R10s, impl/bin_R10s_CUDA)

Règles habituelles : un seul processus GPU à la fois, gardes MI50 (jonction ≤ 50 °C, VRAM < 5 %) et RTX (≤ 50 °C, < 1 Go), SIGTERM, horloges et températures relevées avant et après chaque run. Ne rien modifier dans les worktrees. Retrouve la worktree de r10b-sm86 avec `git worktree list` et vérifie que son HEAD = 1b31581f. Dispatch figé : `git show 1b31581f:dispatch/…` → r10s/.

## A — passe complète
`sweep --all-dispatch` sur gfx906 (bin_R10s, GPU 1) ; attendu 86/86. Puis sur sm_86 (bin_R10s_CUDA) ; attendu 20/20. Liste toute ligne non PASS.

## B — RTX, slots 1, kv q8, lp.py (même protocole que eo2/Q3-Q4)
Ordre alterné, avec garde ≤ 50 °C avant chaque serveur :
| run | binaire | mtp | réf |
|---|---|---|---|
| S1 | bin_O2_CUDA | 0 | 27,48 |
| S2 | bin_R10s_CUDA | 0 | — |
| S3 | bin_O2_CUDA | 4 | 43,22 |
| S4 | bin_R10s_CUDA | 4 | — |
Pour S2 et S4 : acceptation et tokens par cycle, ainsi que le premier caractère de divergence de chaque invite par rapport à S1 et S3 (les noyaux changent, une divergence tardive est possible ; ne pas l'interpréter).

## C — MI50, non-régression rapide
bin_R10s, mtp 0, lp.py ; réf. eo2/Q1 29,31. Textes comparés à eo2.

## D — trace nsys courte (comme r5/B, côté flambeau seulement)
bin_R10s_CUDA, invite « math », 200 tokens : durée client, temps GPU total des noyaux, top 15 des noyaux.

## Rendu
r10s/RESULTATS.md, puis écris « r10s terminé ».
