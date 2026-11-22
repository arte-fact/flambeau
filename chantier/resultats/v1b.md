# v1b — remesure MI50 MTP-4, alterné S7 / V1 / S7 / V1 (TP-2, kv q8, lp_g)
Binaires : `impl/bin_S7` (sha b2636898… == bin_S7.txt) et `impl/bin_V1` (sha de0b3cfe… == bin_V1.txt). Garde avant chaque run : jonction GPU1 ≤ 52 °C, GPU0 ≤ 54 °C, VRAM < 5 %. RTX non utilisée. Un seul serveur à la fois, SIGTERM.

| run | binaire | jonction avant | jonction après | t/s par invite (code/math/chat/story/list) | moyenne | acc / tok-cyc | textes vs s7/lp_d2.json |
|---|---|---|---|---|---|---|---|
| Y1 | bin_S7 | 54/51 °C | 86/85 °C | 59,33 / 51,64 / 36,60 / 26,69 / 31,91 | **41,23** | 58,1 % (923/1588), 3,28 | **identiques (5/5)** |
| Y2 | bin_V1 | 54/52 °C | 86/86 °C | 59,14 / 51,31 / 36,22 / 26,30 / 31,73 | **40,94** | 58,1 %, 3,28 | **identiques (5/5)** |
| Y3 | bin_S7 | 54/52 °C | 86/86 °C | 58,90 / 51,30 / 36,00 / 26,09 / 31,04 | **40,67** | 58,1 %, 3,28 | **identiques (5/5)** |
| Y4 | bin_V1 | 55/52 °C | 86/87 °C | 59,14 / 51,35 / 36,17 / 26,24 / 31,53 | **40,89** | 58,1 %, 3,28 | **identiques (5/5)** |

S7 : 41,23 / 40,67 (moy 40,95) ; V1 : 40,94 / 40,89 (moy 40,92) → écart **−0,1 %**, dans le bruit. Toutes les températures de départ égales (54-55/51-52 °C). Textes bit-exacts vs s7/D2 sur les 4 runs × 5 invites.

## Erreurs
Aucune. ERROR=0 dans fl_y1..y4.log ; 4 serveurs arrêtés par SIGTERM ; aucun processus GPU en fin (VRAM MI50 0 %).
