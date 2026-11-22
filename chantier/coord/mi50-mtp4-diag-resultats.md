# Diagnostic MI50 MTP-4, V2 contre V3 : résultats D1/D2 (Opus, 2026-11-22)

**Conditions** :
- `impl/bin_V2` et `impl/bin_V3`, serveur TP-2, kv q8, `--mtp 4`, mêmes options que v3/run_v3.sh ;
- rocprofv3 `--kernel-trace` ;
- ordre V2, V3, V2, V3, `wait_mi50s` avant chaque lancement ;
- invites `code` (43 cycles) et `math` (97 cycles), à séquence de cycles identique entre V2 et V3 ;
- fenêtre de décodage par requête (first_token → fin), par GPU.

**Données** : scratchpad `diag/`, d2_all.md, d2_rows.json, et les traces v2a, v3a, v2b, v3b.

## Par cycle et par GPU (µs, moyenne des 2 traces par binaire)
| | code GPU0 | code GPU1 | math GPU0 | math GPU1 |
|---|---|---|---|---|
| Δ occupé (V3 − V2) | **+2 320** | **+2 354** | **+2 323** | **+2 365** |
| Δ lancements / cycle | +60 | +60 | +60 | +60 |
| **H2** : matmul Q4_K (row-tile V2 → tuile x8 V3) | **+2 118** | **+2 147** | **+2 077** | **+2 103** |
| **H1** : quantification DS4 (12,9 → 92,0 lancements) | +285 | +280 | +284 | +279 |
| **H1** : quantification Q8_1 (316,0 → 296,5 lancements) | −91 | −82 | −92 | −84 |
| GDN | −58 | −62 | −53 | −86 |
| reste (IQ4_XS, Q5_K, Q6_K, Q8_0, attention, comm, autres) | < ±25 chacun | | | |

- **Mur sous traceur** : +3,1 à +3,6 ms par cycle. Sans traceur, le Vibe mesurait +2,4 ms. L'écart est le coût de traçage des lancements supplémentaires.
- **Bilan** :
  - **H2 explique environ 90 %** de l'écart d'occupation ;
  - **H1 en explique environ 8 %** (+0,19 ms net, plus 60 lancements par cycle) ;
  - H3 (horloge) est exclue : les 4 traces concordent à ±1 %.

## H2 forme par forme : Q4_K à m = 5, formes vues par chaque GPU en TP-2
Médianes par lancement (µs). Nombre de tenseurs par vérification lu dans le GGUF.

| forme par GPU (lignes × k) | tenseurs | row-tile (V2) | x8 (V3) | Δ par vérification |
|---|---|---|---|---|
| 3072 × 5120 (attn_gate) | 24 | 77,5 | 135,5 | +1 392 |
| 5120 × 5120 (attn_qkv) | 29 | 116,8 | 144,8 | +812 |
| 5120 × 3072 (ssm_out) | 5 | 76,2 | 144,8 | +343 |
| 6144 × 5120 (attn_q) | 6 | 133,1 | 144,3 | +67 |
| 512 × 5120 (attn_k) | 1 | 48 | 138 | +90 |
| 5120 × 8704 (ffn_down) | 10 | 192,0 | 144,6 | **−474** |
| 8704 × 5120 (ffn_gate / ffn_up) | 28 | 184,6 | 161,8 | **−638** |
| **total** | | **13 564** | **15 155** | **+1 592** |

- **Mécanisme** :
  - la tuile x8 (MMQ_Y = 64) coûte à peu près le même temps (135 à 145 µs) sur toutes les formes de 5 120 lignes ou moins, quel que soit k ;
  - elle lance lignes / 64 blocs, soit 8 à 96 blocs pour 60 CU : moins d'une vague, donc un noyau limité par la latence ;
  - le row-tile, lui, suit k ;
  - aux formes complètes (un GPU), le nombre de blocs double et le x8 gagnait (10240 × 5120 : 158,6 contre 204,6). En TP-2, il ne gagne plus que sur les formes hautes (8 704 lignes) ou longues (k = 8 704).
- **Écart avec la colonne de famille** : Δ famille +2,12 ms contre +1,59 ms ici. Le reste vient des lancements hors vérification (couche MTP, rattrapage) et de l'appariement approché des formes de 5 120 lignes.

## Correction proposée (D3, à valider)
1. **gfx906 Q4_K : row-tile à m = 2..8**, comme en V2, et x8 à partir de 9 (ou x16, voir le point 3).
   - Gain attendu : l'essentiel des +2,3 ms par cycle.
   - Le défaut de quantification (H1) disparaît avec : à m ≤ 8, plus aucun lecteur DS4 Q4_K.
2. **Avant de fixer la borne** : qmatmul-bench gfx906 aux formes TP-2 ci-dessus, quantification comptée, C0 row-tile contre C1 x8 contre C2 x16, m = 5..16, pour Q4_K et Q5_K. Cela dira :
   - si la borne est 8 ou plus ;
   - si x16 à 9..16 tient encore en TP-2 (lignes / 64 blocs, même risque) ;
   - si Q5_K x8 à 8..32 doit bouger.
3. **Q3_K x8 à 2..32** (27B UD-Q3_K_XL, TP-2) : même mécanisme possible, à mesurer avec le reste.
4. **À plus long terme** :
   - une bande de lignes (n) dans la table : le x8 gagne sur les 8 704 lignes et le row-tile sur les petites ;
   - ou un MMQ_Y plus petit pour les formes TP-2.
