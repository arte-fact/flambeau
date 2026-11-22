# v5 — contrôle après correction gfx906 et lignes sm_86 Q5_K / Q3_K (impl/bin_V5, impl/bin_V5_CUDA)

Commencer seulement quand impl/bin_V5.txt existe et que j'annonce le GPU libre. Vérifier hash et HEAD dans bin_V5.txt ; figer le dispatch (`git show <HEAD>:…`) → v5/.

Règles habituelles : un seul processus GPU, SIGTERM, températures avant et après. Gardes RTX ≤ 50 °C et < 1 Go (sd-server intact, VRAM par processus) ; MI50 GPU0 et GPU1 ≤ 56 °C, VRAM < 5 %. lp_g.py, kv q8.

## D — MI50 TP-2, Qwen3.8-27B, MTP-4, alterné
V2, V5, V2, V5 (impl/bin_V2, impl/bin_V5). Acceptation, tokens par cycle, t/s par invite. Si les lignes gfx906 reviennent toutes à V2, textes identiques à V2 attendus ; sinon, donner l'invite et le premier caractère divergent. Puis mtp 0 : V5 seul, textes comparés à s7/lp_d1.json.

## B — RTX, Qwen3.8-27B, slots 1, MTP-4, alterné
V4, V5, V4, V5 (impl/bin_V4_CUDA, impl/bin_V5_CUDA). Le 27B n'a ni Q3_K ni (à vérifier dans le rendu) de Q5_K hors bande : textes V5 = V4 attendus si aucune ligne prise ne change ; sinon premier caractère divergent.

## E — ornith-1.0-35b-Q4_K_M, MI50 TP-2, mtp 0
V2 puis V5, textes comparés.

Rendu : v5/RESULTATS.md (moyennes, écarts en %, section « Erreurs »), puis « v5 terminé ».
