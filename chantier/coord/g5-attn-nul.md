# g5 — attention_decode_q8_kv_vec_dpos : résultat nul (d2a40acd, branche g5-attn-vec, locale)

Microbench RTX (GQA 24/4, D 256, moyenne de 200 lancements) contre le split-K à la grille de 32 chunks du graphe. Le sweep est vert (14 formes).

| Queue de KV | n_par | vec | split-K | Rapport |
|---|---|---|---|---|
| 33 | 1 | 21,0 µs | 32,7 µs | ×1,56 |
| 181 | 1 | 39,5 µs | 32,7 µs | **×0,83** |
| 1024 | 1 | 149,7 µs | 39,1 µs | ×0,26 |
| 1024 | 4 | 37,7 µs | 34,1 µs | ×0,90 |
| 4001 | 8 | 110,5 µs | 142,0 µs | ×1,28 |
| 8192 | 8 | 214,9 µs | 274,0 µs | ×1,28 |

**Diagnostic.** La boucle de 32 tokens par warp sort par `break` sur la queue de KV, donc elle n'est pas déroulée : chaque token enchaîne une charge de K, un dp4a et 5 shuffles, et le noyau est limité par la latence. Avec un bloc par tête Q, 24 blocs n'occupent que 24 des 82 SM. llama.cpp déroule les 32 tokens à la compilation (masqués, sans break) et choisit le nombre de blocs parallèles selon l'occupation : ce sont les deux leviers si on reprend le noyau.

**Rangement.** La surface Ops est passée sous cfg(unverified). Le .cu reste compilé, et le sweep garde le cert et la microbench (référence de bench). Le prérequis f722655e (lecture de la table [[attention_decode]]) reste utile, avec son cert cb régénéré (Q8_0/F16).
