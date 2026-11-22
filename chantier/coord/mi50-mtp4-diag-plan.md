# Diagnostic MI50 MTP-4, V2 contre V3 (plan, Opus, 2026-11-22)

## Ce que disent déjà les logs du Vibe (v3/fl_m*.log, sans GPU)
- Décodage par cycle, TTFT exclu :
  - `code` : 77,57 / 77,57 ms (V2, m1 et m3) contre 79,97 / 79,93 (V3) ;
  - `math` : 78,96 / 78,99 contre 81,49 / 81,38.
- Sur ces deux invites, le nombre de cycles et les tokens par cycle sont identiques entre V2 et V3 (43 et 97 cycles), et le texte de `code` est identique.
- La régression est donc **+3,1 % de temps par cycle**, soit environ +2,4 ms, à séquence de cycles égale. La baisse d'acceptation (57,9 → 56,2 %) ne vient que des trois invites dont le texte diverge. Le « +1,6 % par cycle » de RESULTATS.md est une moyenne sur des cycles différents.

## Ce qui a changé à la vérification (m = 5), Qwen3.8-27B-UD-Q4_K_M
- Le modèle mélange Q4_K, Q5_K, IQ4_XS, Q6_K et Q8_0 ; il n'a pas de Q3_K.
- **Q4_K** :
  - V2 : raccourci row-tile, activation Q8_1 standard ;
  - V3 : tuile x8 MMQ, activation DS4 (Mmq).
- **Q5_K** : row-tile dans V2 comme dans V3 ; seul le chemin change (raccourci, puis ligne de table).
- **Autres dtypes** : inchangés.
- Les bandes m = 2..4 ne servent que pendant le rattrapage de la couche MTP.

## Hypothèses, toutes deux absentes du microbench
- **H1, double quantification.** Un site dont les consommateurs mélangent Q4_K et un autre dtype doit maintenant quantifier l'activation dans les deux formats :
  - qkv ou q avec k/v Q5_K, alpha/beta Q8_0 ;
  - gate Q4_K avec up IQ4_XS ;
  - etc.

  C'est environ 1 à 2 noyaux de quantification de plus par couche, soit de l'ordre de 100 lancements de plus par vérification et par GPU. Le microbench ne lançait pas la quantification (mmq-few-resultats.md, « Limites »).
- **H2, formes TP-2.** Le microbench a pesé les formes complètes du modèle sur un seul GPU. En TP-2, chaque GPU voit des formes réduites de moitié :
  - en lignes pour qkv, gate, up et q ;
  - en k pour down, ssm_out et attn_output (k = 8704 et 3072).

  Or le gagnant à m = 5 dépend de la forme : le row-tile gagne à k = 17408 ou 6144 et sur les petites matrices, et Q4_K a été tranché sur 0,6 % d'écart. À ces formes, le row-tile peut gagner.
- **H3, bruit d'horloge ou de température.** Peu probable : l'écart est identique au µs près sur deux alternances.

## D1 : traces (les deux MI50, `wait_mi50s.sh` avant chaque lancement)
- **Binaires** : `impl/bin_V2` et `impl/bin_V3`, ceux mesurés par le Vibe, sans rebuild.
- **Serveur** : mêmes options que `run_v3.sh` (TP-2, kv q8, `--mtp 4`, ctx-cap 8192, ubatch 512), sous :

  ```
  rocprofv3 --kernel-trace --output-format csv -d $SP/diag/<tag> -- <bin> serve …
  ```
- **Requêtes** : invites `code` puis `math` de lp_g.py (gloutonnes, 400 tokens), puis arrêt du serveur par SIGINT pour que rocprofv3 vide sa trace.
- **Ordre** : V2, V3, V2, V3.
- **Estimation** : environ 3 min par lancement (chargement du 27B compris), plus le refroidissement jusqu'à 56 °C, soit environ 1 h au total.
- **Risque** : si rocprofv3 ne vide pas la trace à l'arrêt du serveur, je passe à `--collection-period` ou à un arrêt propre ; à valider au premier lancement.

## D2 : analyse (CPU)
- **Fenêtre** : décodage de chaque requête, entre first_token et finish du log ; pour chaque GPU séparément.
- **Mesures** :
  - temps GPU occupé par cycle ;
  - nombre de lancements par cycle ;
  - temps inoccupé, c'est-à-dire mur moins occupé.
- **Ventilation par famille** :
  - matmul Q4_K par noyau (row-tile n5 ou x8), Q5_K, IQ4_XS, Q6_K, Q8_0 ;
  - quantification Q8_1 contre DS4 ;
  - attention, GDN, reste.
- **Rendu** :
  - tableau Δ V3 − V2 en µs par cycle et par famille ;
  - somme des Δ comparée aux +2,4 ms du mur ;
  - noyau Q4_K à m = 5 forme par forme, row-tile contre x8, aux formes TP-2 réelles.
- **Lecture attendue** :
  - H1 vérifiée si les lancements de quantification par cycle augmentent ;
  - H2 vérifiée si le x8 Q4_K coûte plus que le row-tile de V2 sur la somme des formes TP-2.

## D3 : correction, seulement après D2 et ton accord
- **Si H1 ou H2 explique l'écart**, deux options :
  - remettre Q4_K sur le row-tile jusqu'à m = 8, comme dans V2 ;
  - ou borner à 2..5.

  Avant de choisir la borne, un qmatmul-bench gfx906 (GPU 1) C0 contre C1, m = 5, 6, 8, **aux formes TP-2, quantification comptée**. Si le bench ne sait pas compter la quantification, je l'ajoute (code seulement).
- **Suite plus large**, à décider :
  - la règle de bande doit compter le coût d'un site mixte, c'est-à-dire la seconde quantification ;
  - les décisions de mmq-few-resultats.md (Q5_K 8..32, Q4_K x16 9..16) sont à revoir aux formes TP-2 si le modèle cible tourne en TP-2.
- **Puis** : commit de bande sur v3, cert-check, et MTP-4 du Vibe V2 contre V3'. Pas de push de v3 avant.
- **Si D2 ne montre aucun Δ de noyau** : l'écart est côté hôte (lancements, synchronisations), et je trace le côté hôte avant toute autre proposition.
