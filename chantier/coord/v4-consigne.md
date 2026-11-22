# v4 — RTX, MTP-4 de bout en bout avec le few MMA Q4_K (impl/bin_V3_CUDA contre impl/bin_V4_CUDA)

Commencer seulement quand impl/bin_V4.txt existe et que j'annonce que le GPU est libre : Opus fait d'abord sa séquence RTX (build, certs, tests). Vérifier hash et HEAD dans bin_V4.txt ; figer le dispatch sm_86 avec `git show <HEAD>:…` → v4/.

RTX seulement, Qwen3.8-27B-UD-Q4_K_M, slots 1, kv q8, lp_g.py. Gardes : ≤ 50 °C et < 1 Go (sd-server : ne pas y toucher, VRAM par processus). Un seul processus GPU, SIGTERM, températures avant et après.

| run | binaire | mtp |
|---|---|---|
| W1 | bin_V3_CUDA | 4 |
| W2 | bin_V4_CUDA | 4 |
| W3 | bin_V3_CUDA | 4 |
| W4 | bin_V4_CUDA | 4 |
| W5 | bin_V4_CUDA | 0 |

Pour chaque run : t/s par invite, moyenne, acceptation et tokens par cycle (MTP-4).
Textes : W4 = W2 attendu ; W2 comparé à W1 (la vérification change de noyau à m = 3..24 : des différences sont possibles ; donner l'invite et le premier caractère divergent, sans interpréter) ; W5 comparé à v3 y2 (identique attendu : le décodage m = 1 ne change pas).
Référence : llama.cpp HEAD MTP-4 = 89,2.

Rendu : v4/RESULTATS.md (moyennes, écart V4 − V3 en %, section « Erreurs »), puis « v4 terminé ».
