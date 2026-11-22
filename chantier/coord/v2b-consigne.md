# v2b — MI50 mtp 0, V1 contre V2 en alternance (impl/bin_V1, impl/bin_V2)

Point à vérifier, signalé par v2 : en mtp 0, D5 (bin_V2) donne 28,64, contre 29,81 en v1/D1. Ce n'était qu'un seul run, non alterné, parti à 54/52 °C.

Commencer seulement quand j'annonce que le GPU est libre (Opus fait d'abord un microbench). Règles habituelles : un seul processus GPU à la fois, arrêt par SIGTERM, garde avant chaque run (GPU1 ≤ 52 °C, GPU0 ≤ 54 °C, VRAM < 5 %), températures notées avant et après. Ne pas toucher à la RTX.

TP-2, kv q8, lp_g, mtp 0. Ordre : V1, V2, V1, V2, V1, V2.
Pour chaque run : t/s par invite, moyenne, températures, et textes comparés à s7/lp_d1.json.

Rendu : v2b/RESULTATS.md, avec les moyennes V1 et V2, l'écart en %, et une section « Erreurs ». Puis dire « v2b terminé ».
