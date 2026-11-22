# s7 — contrôle de la série 7 avant push (branche serie-7, HEAD f5d29f30 ; impl/bin_S7, impl/bin_S7_CUDA)

Contenu, 11 commits sur ee32a49c : certs gfx906 aux formes 17408 ; stop-trim (non-streaming = concaténation des deltas) ; r2-sm86 (sweeps Q4_0 mma et attention_prefill_mma, ligne Q4_0 coupée, [[attention_prefill]] par la table) ; IQ3_S batched m ≥ 2 sur sm_86.

Règles habituelles : un seul processus GPU à la fois, et aucune charge CPU lourde en parallèle (Opus ne compile pas). Gardes MI50 (jonction ≤ 50 °C sur le GPU utilisé, VRAM < 5 %) et RTX (≤ 50 °C, < 1 Go). Arrêt par SIGTERM. Vérifier les hash dans bin_S7.txt et HEAD serie-7 = f5d29f30. Dispatch figé `git show f5d29f30:…` → s7/.

## A — certificats
all-dispatch gfx906 (bin_S7) : attendu 86/86. sm_86 (bin_S7_CUDA) : attendu 33/33 (ou le nombre de bin_S7.txt). Lister toute ligne non PASS.

## B — RTX, slots 1, kv q8, lp_g.py, ordre alterné
| run | binaire | mtp | réf |
|---|---|---|---|
| W1 | bin_G5_CUDA | 0 | 44,66 |
| W2 | bin_S7_CUDA | 0 | — |
| W3 | bin_G5_CUDA | 4 | 77,96 |
| W4 | bin_S7_CUDA | 4 | — |
| W5 | bin_S7_CUDA | 4 | répétition W4 |
Textes : IQ3_S batched change les sommes à m ≥ 2 (préremplissage et vérification), donc des divergences sont possibles. Donner le premier caractère de divergence sans interpréter. W5 doit être identique à W4. MTP : acceptation et tokens par cycle.

## C — RTX, KV f16, mtp 0, lp_g (l'attention de préremplissage passe désormais par la table)
bin_G5_CUDA puis bin_S7_CUDA ; textes identiques attendus.

## D — MI50 (bin_S7, TP-2, kv q8, lp_g)
mtp 0 (réf. g5/D 29,77) et mtp 4 (réf. g1/H2 41,63) ; textes comparés à g5 et à g1/H2.

## E — streaming
Sur bin_S7_CUDA, mtp 0, les 5 invites de lp_g en `"stream": true` : la concaténation des deltas `content` doit être identique au caractère près au texte non streamé de W2.

## Rendu
s7/RESULTATS.md, puis « s7 terminé ». Ensuite, dflash-base selon coord/dflash-base-commandes.md (scripts préparés dans dflash-base/), rendu dflash-base/RESULTATS.md, puis « dflash-base terminé ».
