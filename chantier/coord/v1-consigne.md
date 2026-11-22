# v1 — mtp0-noload + embed Q4_K sur device + brouillon D4 enchaîné + GDN par K états (branche mtp-gdn-kstates ; impl/bin_V1, impl/bin_V1_CUDA)

Les binaires et le .txt sont produits par Opus (coord/v1-sequence.md). Commencer seulement quand impl/bin_V1.txt existe et qu'Opus a dit que le build est terminé (aucune compilation en parallèle).

Règles habituelles : un seul processus GPU à la fois, SIGTERM, températures avant et après. Gardes RTX ≤ 50 °C et < 1 Go. Gardes MI50 GPU1 ≤ 52 °C, GPU0 ≤ 54 °C (repos 51/53), VRAM < 5 %. Vérifier les hash et la HEAD dans bin_V1.txt ; dispatch figé `git show <HEAD>:…` → v1/. Protocole lp_g.py.

Tous les changements sont annoncés bit-exacts : **les textes doivent être identiques à S7 partout**. Donner le premier caractère de toute divergence, sans interpréter.

## A — certificats
all-dispatch gfx906 (bin_V1) et sm_86 (bin_V1_CUDA) : nombres attendus dans bin_V1.txt. Lister toute ligne non PASS.

## B — RTX, slots 1, kv q8, ordre alterné
| run | binaire | mtp | réf |
|---|---|---|---|
| X1 | bin_S7_CUDA | 0 | 44,73 |
| X2 | bin_V1_CUDA | 0 | — |
| X3 | bin_S7_CUDA | 4 | 78,97 |
| X4 | bin_V1_CUDA | 4 | — |
| X5 | bin_V1_CUDA | 4 | répétition X4 |
Textes X2 = X1 et X4 = X3 = X5. MTP : acceptation et tokens par cycle.

## C — mémoire RTX (nvidia-smi, serveur chargé, après une requête)
bin_S7_CUDA et bin_V1_CUDA, chacun en mtp 0 puis mtp 4. Calcul d'Opus : V1 − S7 ≈ −2 137 Mio en mtp 0 et ≈ −1 226 Mio en mtp 4. Donner les quatre valeurs et les deux écarts.

## D — MI50 (bin_V1, TP-2, kv q8)
mtp 0 (réf. s7 29,75) et mtp 4 (réf. s7 41,62) ; textes comparés à s7/lp_d1.json et s7/lp_d2.json (ou aux noms de fichiers réels de s7/D). Relever la VRAM par carte avec rocm-smi, serveur chargé, pour S7 et V1 en mtp 4.

## E — autres arch dont le token_embd est en Q4_K
gemma-4-12b-it-Q4_0.gguf et ornith-1.0-35b-Q4_K_M.gguf, mtp 0, lp_g, bin_S7 puis bin_V1 (RTX si le modèle tient, sinon MI50). Textes identiques attendus ; débits pour information.

## Rendu
v1/RESULTATS.md, puis « v1 terminé ». Ensuite le GPU revient à Opus pour le nsys.
