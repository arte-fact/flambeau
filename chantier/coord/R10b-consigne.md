# R10b — MMVQ une ligne : plus de poids par thread, comme llama.cpp

Rédigé par Fable (coordination), pour Opus. Remplace l'hypothèse de R10a.

## Ce que R10a a montré

Passer de 256 à 128 threads par ligne ne gagne qu'environ 7 % par appel (trace
`r10a/trace_b4`) :

| Appel | 256 threads | 128 threads (R10a) | llama.cpp, même K |
|---|---|---|---|
| Q5_K, 5 120 lignes × K 5 120 | 56,0 µs (10,9 ns/ligne) | 52,3 µs (10,2) | ≈ 6,4 ns/ligne |
| IQ4_XS, 8 704 lignes × K 5 120 | 76,7 µs (8,8) | 70,7 µs (8,1) | ≈ 5,7 ns/ligne |
| Q4_K, 8 704 lignes × K 5 120 | 85,7 µs (9,8) | 79,1 µs (9,1) | ≈ 5,7 ns/ligne |

Le coût ne suit donc pas le nombre de vagues. Mon hypothèse était fausse.

## Différence de structure relevée dans le code

À nombre de threads égal, un thread flambeau traite 2 à 4 fois moins de poids par super-bloc
qu'un thread llama.cpp, et les mêmes octets sont chargés par plusieurs threads.

| Type | llama.cpp (`ggml/src/ggml-cuda/vecdotq.cuh`) | flambeau (`kernels-shared/include/mmvq_*.cuh`) |
|---|---|---|
| Q5_K | `vec_dot_q5_K_q8_1` (`:1023`), `VDR = 2`, `QR5_K = 2` : un thread charge 2 entiers de `qs` et 2 de `qh`, puis traite **les deux quartets**, soit 2 sous-blocs, 16 poids, 4 `dp4a`. 16 threads par super-bloc | `mmvq_q5_k_dp4a_row` : 32 threads par super-bloc ; un thread charge les mêmes 2 entiers de `qs` mais n'utilise **qu'un quartet** (`sub & 1`), soit 8 poids, 2 `dp4a` |
| Q4_K | `vec_dot_q4_K_q8_1`, `VDR = 2`, même schéma (16 poids par thread) | `mmvq_q4_k_dp4a_row` : 32 threads par super-bloc, 8 poids par thread |
| IQ4_XS | `vec_dot_iq4_xs_q8_1` (`:1420`), `VDR = 4` : un thread traite un sous-bloc entier, 32 poids, 8 `dp4a`. 8 threads par super-bloc | `mmvq_iq4_xs_dp4a_row` : 32 threads par super-bloc, 8 poids, 2 `dp4a` par thread |

Conséquences côté flambeau, par super-bloc : deux fois (Q4_K, Q5_K) ou quatre fois (IQ4_XS)
plus de threads, donc autant de fois le décodage des échelles, les chargements d'en-tête
(`d`, `dmin`, `scales`), les conversions entier → flottant et le travail de réduction. Pour
Q4_K et Q5_K, chaque entier de `qs` est chargé par deux threads.

Avec 128 threads, llama.cpp couvre 8 super-blocs par tour en Q4_K/Q5_K et 16 en IQ4_XS ;
flambeau en couvre 4. C'est cohérent avec le rapport mesuré de 1,5 à 1,7.

Les compteurs matériels vont dans le même sens (rocprofv3, `r10b/`, GPU à 51 °C) :

| Noyau | MemUnitBusy | MemUnitStalled | VALUBusy |
|---|---|---|---|
| flambeau `mmvq_q5_k_dp4a_q8_1` (256 threads), 5 120 et 8 704 lignes | **89 à 93 %** | 0,1 % | 64 à 67 % |
| llama.cpp `mul_mat_vec_q` Q5_K, 5 120 à 17 408 lignes | 66 à 74 % | 0,5 à 1 % | 59 à 71 % |
| llama.cpp Q4_K | 61 à 71 % | ≈ 1 % | 64 à 77 % |
| llama.cpp IQ4_XS | 61 à 69 % | ≈ 1 % | 49 à 65 % |

Le noyau flambeau sature les unités mémoire (≈ 90 %) tout en étant plus lent : il émet plus
de chargements par poids utile. C'est ce que prédit la lecture du code (entiers de `qs` chargés
deux fois, en-têtes chargés par 32 threads au lieu de 16 ou 8). Réduire les chargements par
poids est donc le bon levier ; ajouter des vagues ou réduire les VGPR ne l'est pas.

## Travail demandé

Nouvelles variantes une ligne, 128 threads, 1 ligne par bloc, calquées sur le découpage de
llama.cpp :

1. **Q5_K et Q4_K** : 16 threads par super-bloc. Chaque thread charge ses 2 entiers de `qs`
   (et de `qh` pour Q5_K) une seule fois, traite le quartet bas et le quartet haut (2
   sous-blocs), avec les 2 couples (échelle, min) et les 2 blocs Q8_1 correspondants.
   8 super-blocs par tour.
2. **IQ4_XS** : 8 threads par super-bloc. Chaque thread traite un sous-bloc entier (4 entiers
   de `qs`, 8 `dp4a`, une échelle). 16 super-blocs par tour.
3. Garder la correction par `s` du Q8_1 si elle reste exacte ; sinon reprendre le
   `dp4a(0x01010101, u)` de llama.cpp. Le choix le plus rapide gagne, à mesurer.
4. Nommage (règle 9) : la forme doit se lire dans le nom, par exemple
   `qmatmul_q5_k_mmvq_dp4a_t128_vdr2` et `qmatmul_iq4_xs_mmvq_dp4a_t128_vdr4`. Un certificat
   par variante (règle 2), lignes de dispatch gfx906 (règle 1). Les variantes t128 de R10a
   passent inactives, certificats conservés.
5. Hors périmètre pour l'instant : chemin à sortie F16 hors dispatch, variantes par lot,
   lm_head, Q6_K, Q3_K, IQ3_S, CUDA.

## Validation

- Certificats verts (sweeps lancés par le Vibe).
- Trace « math » sans MTP, analysée par `an_grid.py`. Cibles par appel : Q5_K 5 120 lignes
  ≤ 36 µs ; IQ4_XS 8 704 lignes ≤ 52 µs ; Q4_K 8 704 lignes ≤ 55 µs.
- Cible globale : sans MTP ≥ 26 t/s en moyenne sur les 5 invites (25,16 aujourd'hui).
- Contrôle MTP : MTP-4 ≥ 38 t/s, acceptation ≥ 58 %.
- Si le gain par appel est inférieur à 15 % : ne pas activer dans le dispatch, garder derrière
  `cfg(unverified)` avec une ligne de diagnostic, et me renvoyer les chiffres.
