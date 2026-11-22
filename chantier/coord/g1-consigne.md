# g1 — glouton pur (sans pénalité de répétition), pour comparer à llama.cpp à conditions égales

Contexte (coord/s1-penalite.md) : à température 0, flambeau impose par défaut une pénalité de répétition de 1,05 ; llama-server n'en applique aucune (1,00). Les références llama.cpp (RTX 43,5 / 78,0, MI50 25,7 / 37,3) sont en glouton pur.

Règles habituelles : un seul processus GPU à la fois. Gardes : MI50 jonction ≤ 50 °C et VRAM < 5 % ; RTX ≤ 50 °C et < 1 Go. Arrêt par SIGTERM ; relever les températures avant et après chaque run. Ne rien modifier dans les worktrees, ni lp.py.

## Préparation
Copier lp.py en lp_g.py et y ajouter `"repetition_penalty":1.0` au corps de la requête, sans rien changer d'autre. Au premier run, vérifier dans fl_*.log ou dans la réponse que la pénalité est bien de 1,0 ; si le serveur ignore le champ, arrêter et me le dire.

## A — RTX, slots 1, kv q8, ordre alterné (lp.py = avec pénalité 1,05, lp_g.py = sans pénalité)
| run | binaire | mtp | script |
|---|---|---|---|
| G1 | bin_S1_CUDA | 0 | lp_g.py |
| G2 | bin_S1_CUDA | 4 | lp_g.py |
| G3 | bin_S1_CUDA | 0 | lp.py (contrôle, réf. 40,88) |
| G4 | bin_S1_CUDA | 4 | lp_g.py (répétition de G2) |

Pour les runs MTP, relever l'acceptation et les tokens par cycle, ainsi que le nombre de tokens générés par invite.

## B — MI50 TP-2, kv q8, ordre alterné
| run | binaire | mtp | script |
|---|---|---|---|
| H1 | bin_S1 | 0 | lp_g.py |
| H2 | bin_S1 | 4 | lp_g.py |
| H3 | bin_R10m | 0 | lp.py (lever le doute sur le −1,0 % de s1/Q2) |
| H4 | bin_S1 | 0 | lp.py |

Pour les runs MTP, relever les mêmes champs qu'en A.

## Rendu
g1/RESULTATS.md, avec un tableau final qui met côte à côte flambeau sans pénalité et llama.cpp (références de l'ETAT). Terminer par « g1 terminé ».
