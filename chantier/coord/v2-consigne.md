# v2 — K états sur les seules couches GDN, eh_proj en Q6_K, sans copie des logits bonus, quantification unique, parité IQ2_XXS relâchée, harnais qmatmul-bench (branche v2 ; impl/bin_V2, impl/bin_V2_CUDA)

Les binaires et le .txt sont produits par Opus. Commencer seulement quand impl/bin_V2.txt existe et qu'Opus a dit que le build est terminé : aucune compilation ne doit tourner en parallèle.

Règles habituelles : un seul processus GPU à la fois, arrêt par SIGTERM, températures relevées avant et après chaque run. Gardes RTX : ≤ 50 °C et < 1 Go (sd-server tourne : ne pas y toucher ; mesurer la VRAM par processus). Gardes MI50 : GPU1 ≤ 52 °C, GPU0 ≤ 54 °C, VRAM < 5 %. Vérifier les hash et la HEAD dans bin_V2.txt. Figer le dispatch avec `git show <HEAD>:…` → v2/. Protocole lp_g.py.

**Textes attendus :**
- en mtp 0, identiques à V1 ;
- en MTP-4, les tokens vérifiés ne devraient pas changer, mais eh_proj passe d'une activation F16 à Q8_1, donc les brouillons et l'acceptation peuvent bouger ;
- comparer quand même les textes MTP-4 à V1. Donner le premier caractère de toute divergence, l'acceptation et les tokens par cycle, sans interpréter.

## A — certificats
all-dispatch gfx906 (bin_V2) et sm_86 (bin_V2_CUDA) : les nombres attendus sont dans bin_V2.txt. Lister toute ligne non PASS.

## B — RTX, slots 1, kv q8, ordre alterné
| run | binaire | mtp | réf |
|---|---|---|---|
| Y1 | bin_V1_CUDA | 0 | 44,69 |
| Y2 | bin_V2_CUDA | 0 | — |
| Y3 | bin_V1_CUDA | 4 | 79,95 |
| Y4 | bin_V2_CUDA | 4 | — |
| Y5 | bin_V2_CUDA | 4 | répétition de Y4 |

Textes : Y2 = Y1 attendu. Y4 comparé à Y3. Y5 = Y4 attendu.

## C — mémoire RTX (nvidia-smi, serveur chargé, après une requête ; mémoire par processus)
bin_V1_CUDA et bin_V2_CUDA, chacun en mtp 0 puis mtp 4. Calcul d'Opus pour V2 − V1 : ≈ −53 Mio en mtp 0 et ≈ −370 Mio en mtp 4 (−255 pour les K états, −53 pour l'état vivant, −62 pour eh_proj). Donner les quatre valeurs et les deux écarts.

## D — MI50 (TP-2, kv q8), ordre alterné
mtp 4 : V1, V2, V1, V2 ; puis mtp 0 : V2 seul (réf. v1 29,81 ; textes comparés à s7/lp_d1.json). Pour mtp 4 : acceptation et tokens par cycle ; textes de V2 comparés à ceux de V1. VRAM par carte avec rocm-smi, serveur chargé, V1 et V2 en mtp 4.

## E — ornith-1.0-35b-Q4_K_M (MI50 TP-2, mtp 0)
V1 puis V2. Textes identiques attendus ; débits pour information.

## Rendu
v2/RESULTATS.md (avec une section « Erreurs »), puis « v2 terminé ». Le GPU revient ensuite à Opus pour l'étape 0 du plan MMQ.
