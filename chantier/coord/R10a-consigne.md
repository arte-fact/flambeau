# R10a — MMVQ une ligne à 2 vagues (128 threads) sur gfx906

Rédigé par Fable (coordination), pour Opus. Objectif : l'écart sans MTP (21,7 → 25,9 t/s).

## Constat mesuré

Les MMVQ de décodage font environ 29 ms des 38 ms de calcul GPU par token et par GPU (trace
`r3r4/trace_r4`, après R4). Coût par ligne de matrice, à K = 5 120 (20 super-blocs), relevé
dans les traces rocprofv3 des deux moteurs sur ces mêmes MI50 :

| Noyau | Threads par ligne | Itérations par thread | ns par ligne |
|---|---|---|---|
| llama.cpp `mul_mat_vec_q` Q5_K / IQ4_XS, 10 240 et 17 408 lignes | 128 (2 vagues) | 5 | 5,7 à 6,4 |
| flambeau `mmvq_q5_k_dp4a_q8_1`, 5 120 et 8 704 lignes | 256 (4 vagues) | 2,5 | 10,3 à 11,0 |
| flambeau `mmvq_iq4_xs_dp4a_q8_1`, 8 704 lignes | 256 (4 vagues) | 2,5 | 8,8 |
| flambeau `mmvq_q4_k_r2_dp4a_q8_1`, 8 704 et 5 120 lignes | 32 (2 lignes par vague) | 20 | 13,8 à 14,8 |

- llama.cpp choisit `nwarps = 2`, `rows_per_block = 1` pour GCN à une colonne
  (`ggml/src/ggml-cuda/mmvq.cu:473-476` et `:600-610`).
- Le coût suit le nombre de vagues lancées par ligne, pas les octets lus : ≈ 2,6 à 3 ns par
  vague dans les deux moteurs. À 256 threads, flambeau lance deux fois plus de vagues, qui ne
  font que 2,5 itérations chacune ; au dernier tour, 4 des 8 groupes de threads sont inactifs
  (20 super-blocs pour 8 par tour).
- Sur les lignes longues (ffn_down, K = 8 704), les noyaux flambeau à 256 threads sont déjà
  corrects (IQ4_XS : 8,2 ns par ligne pour 34 super-blocs).

Hypothèse à tester : à 128 threads par ligne, les noyaux flambeau rejoignent ≈ 6 ns par ligne à
K = 5 120. Gain attendu sur Q5_K + IQ4_XS + Q4_K : ≈ 9 ms par token et par GPU. C'est une
hypothèse fondée sur les mesures ci-dessus, pas encore un résultat.

## Travail demandé

1. Variantes à 128 threads (4 super-blocs par tour, sinon identiques) des noyaux une ligne :
   - Q5_K : `mmvq_q5_k_dp4a_row` (`kernels-shared/include/mmvq_q5_k.cuh`, `THREADS`,
     `SB_PER_ITER`) ;
   - IQ4_XS : l'équivalent dans `mmvq_iq4_xs.cuh` ;
   - Q4_K : à partir de ta variante t256 de R5.
   Paramétrer le nombre de threads par un argument de gabarit plutôt que copier le corps.
2. Nommage (règle 9) : suffixe de forme explicite, par exemple
   `qmatmul_q5_k_mmvq_dp4a_t128`. Un `impl_id` et un certificat par variante (règle 2).
3. Dispatch (règle 1) : lignes dans `dispatch/hip/gfx906.toml`, pas de constante cachée. Si le
   dispatch sait filtrer par K, réserver t128 à K ≤ 5 120 et garder t256 au-delà ; sinon
   t128 partout dans un premier temps, et on tranche sur la mesure par forme.
4. Ne pas toucher aux variantes par lot (`batched_n5`) ni à la lm_head (64 threads,
   590 GB/s) dans cette étape. CUDA inchangé.

## Validation

- Certificats verts : `sweep --arch gfx906 --op qmatmul` sur les nouvelles variantes (lancé
  par le Vibe).
- Trace rocprofv3 « math », 130 tokens, sans MTP (script du Vibe `runB3.sh`), analysée avec
  `analyse-ecarts/an_grid.py` et `an_type.py`.
- Cibles par appel : Q5_K 5 120 lignes ≤ 36 µs (56 aujourd'hui) ; IQ4_XS 8 704 lignes ≤ 55 µs
  (77) ; Q4_K 8 704 lignes ≤ 60 µs (120).
- Cible globale : matmul ≤ 20 ms par GPU et par token (29 aujourd'hui) ; sans MTP ≥ 25 t/s en
  moyenne sur les 5 invites.
- Contrôle MTP : MTP-4 ≥ 33,6 t/s et acceptation ≥ 57,5 %.

Si la mesure ne confirme pas (gain < 15 % par appel), ne pas livrer : garder la variante
derrière `cfg(unverified)` avec une ligne de diagnostic, et me renvoyer les chiffres.
