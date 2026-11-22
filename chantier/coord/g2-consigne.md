# g2 — graphe CUDA du décodage qwen35 (branche g2-decode-graph, HEAD b9b69d2a ; binaires impl/bin_G2, impl/bin_G2_CUDA)

Règles habituelles : un seul processus GPU à la fois ; gardes MI50 (jonction ≤ 50 °C, VRAM < 5 %) et RTX (≤ 50 °C, < 1 Go) ; arrêt par SIGTERM ; températures relevées avant et après. Ne rien modifier dans les worktrees. Vérifier que HEAD de g2-decode-graph = b9b69d2a (`git worktree list`). Dispatch figé `git show b9b69d2a:…` → g2/.
**Le protocole de référence est désormais lp_g.py** (glouton pur, g1/lp_g.py).

## A — certificats
all-dispatch gfx906 (bin_G2, GPU 1) : attendu 86/86. sm_86 (bin_G2_CUDA) : attendu 27/27. Lister toute ligne non PASS.

## B — RTX, slots 1, kv q8, lp_g.py, ordre alterné
| run | binaire | mtp | réf |
|---|---|---|---|
| K1 | bin_S1_CUDA | 0 | 41,13 |
| K2 | bin_G2_CUDA | 0 | — |
| K3 | bin_S1_CUDA | 4 | 78,56 |
| K4 | bin_G2_CUDA | 4 | — |
| K5 | bin_G2_CUDA | 0 | répétition de K2 |
Textes : K2 identique à K1 et K4 identique à K3 attendus ; sinon donner le premier caractère de divergence. Pour les runs MTP : acceptation et tokens par cycle.

## C — MI50, non-régression rapide
bin_G2, mtp 0, lp_g.py ; réf. g1/H1 29,75 ; textes comparés à g1/H1.

## D — trace nsys sans MTP (bin_G2_CUDA, invite « math », 200 tokens, lp_g / sans pénalité)
- durée client ; temps GPU total des noyaux ;
- **GPU inactif par token en décodage** (réf. ≈ 1,07 ms d'après coord/g-nomtp.md ; llama.cpp 0,11) ;
- nombre et coût des captures de graphe (cudaGraphInstantiate / cudaStreamEndCapture dans cuda_api_sum) et positions où elles ont lieu ;
- top 15 des noyaux.

## Rendu
g2/RESULTATS.md, puis « g2 terminé ».
