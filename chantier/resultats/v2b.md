# v2b — MI50 mtp 0, TP-2, kv q8, lp_g, alterné V1/V2/V1/V2/V1/V2
Binaires : `impl/bin_V1` (de0b3cfe…), `impl/bin_V2` (d4b19ba9…). RTX non touchée. Un seul serveur à la fois, SIGTERM.
**Garde thermique** : GPU0/GPU1 ≤ 56 °C (relâchée depuis ≤ 52/≤ 54 : GPU1 au repos à 53 °C puis GPU0 à 54-55 °C, la garde initiale ne passait jamais) + VRAM < 5 % ; même garde pour V1 et V2 (runs alternés).

| run | binaire | jonction avant | jonction après | t/s par invite (code/math/chat/story/list) | moyenne | textes vs s7/lp_d1.json |
|---|---|---|---|---|---|---|
| Z1 | bin_V1 | 52/53 °C | 88/89 °C | 30,74 / 30,55 / 29,33 / 27,28 / 24,91 | **28,56** | identiques (5/5) |
| Z2 | bin_V2 | 54/52 °C | 90/90 °C | 30,72 / 29,61 / 28,91 / 26,35 / 24,99 | **28,12** | identiques (5/5) |
| Z3 | bin_V1 | 56/55 °C | 90/90 °C | 30,73 / 30,37 / 27,54 / 26,39 / 24,25 | **27,86** | identiques (5/5) |
| Z4 | bin_V2 | 56/55 °C | 90/91 °C | 30,82 / 30,38 / 27,98 / 26,39 / 24,31 | **27,98** | identiques (5/5) |
| Z5 | bin_V1 | 56/55 °C | 90/91 °C | 30,74 / 30,42 / 27,85 / 26,45 / 24,16 | **27,92** | identiques (5/5) |
| Z6 | bin_V2 | 57/55 °C | 90/90 °C | 30,81 / 30,43 / 27,81 / 26,32 / 24,17 | **27,91** | identiques (5/5) |

**Moyennes : V1 = 28,11 t/s (28,56 / 27,86 / 27,92) ; V2 = 28,00 t/s (28,12 / 27,98 / 27,91) → écart V2−V1 = −0,40 %**, dans le bruit. Le 28,64 de v2/D5 était donc un artefact de run unique (parti à 54/52, non alterné). Textes bit-exacts vs s7/D1 sur les 6 runs × 5 invites.

## Erreurs
Aucune. ERROR=0 dans fl_z1..z6.log ; 6 serveurs arrêtés par SIGTERM ; aucun processus GPU en fin (VRAM 0 %).
