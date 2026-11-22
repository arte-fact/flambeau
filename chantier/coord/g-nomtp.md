# g — écart sans MTP sur la RTX (40,9 contre 43,5 t/s, soit 24,45 contre 22,99 ms par token : +1,46 ms)

Méthode : traces r10s/r10s.sqlite (bin_R10s_CUDA, mtp 0) et r5/lc.sqlite (llama.cpp), requête « math ». Fenêtre de 149 tokens de décodage (têtes 40 à 189), bornée par les têtes Q6_K à 248 320 lignes. Sous nsys, on mesure 25,01 contre 24,07 ms par token.

1. **Temps GPU occupé : 22,77 contre 22,38 ms par token (+0,39).**
   - **MMVQ : 19,29 contre 19,07 (+0,22).**
     - La tête Q6_K fait +50 µs (1 225 contre 1 175).
     - Le Q8_0 gate_up fait +86 µs (48 × 7,0 contre 96 × 2,6 µs).
     - Les Q5_K et Q6_K à 5120 lignes font ≈ +110 µs.
     - Le reste est à la parité. llama.cpp fusionne gate et up en GLU (IQ4_XS à 17408 : 113 µs contre 2 × 56,4), ce qui n'apporte aucun gain de bande passante.
   - **Attention : 0,61 contre 0,22 ms (+0,39).**
     - `attention_decode_q8_kv_splitk_chunk` prend 35,6 µs × 16, quel que soit le nombre de chunks (4 à 8), avec des blocs de 64 threads, plus 2,5 µs de combine.
     - `flash_attn_ext_vec<256>` prend 13,6 µs, avec 24 blocs de 128 threads.
     - Notre noyau est donc limité par la latence, pas par les données (contexte d'environ 180 tokens).
   - **Petits noyaux : −0,22 en notre faveur.** `rmsnorm_q8_1_fused` et les quantize coûtent 1,06 ms, contre 1,47 ms pour rms_norm et quantize_q8_1 chez llama.cpp.
2. **GPU inactif : 2,24 contre 1,69 ms sous nsys.** Hors nsys, en soustrayant le temps occupé du temps de mur, on obtient **≈ 1,68 contre ≈ 0,61 ms (+1,07)**.
   - **Entre les noyaux d'un même token : 1,45 contre 0,11 ms.**
     - Chez nous, sans graphe, 1 459 trous de moins de 1 µs font 1,20 ms.
     - S'y ajoutent 62 trous de 3 à 10 µs, soit 0,20 ms, autour de copies : 49 DtoD de 120 Ko par token (état conv GDN, 69 µs) et 16 HtoD de 4 octets (positions d'attention).
     - llama.cpp rejoue un graphe : 1 906 trous qui totalisent 96 µs.
   - **À la frontière entre tokens :** 0,80 contre 1,62 ms sous nsys. Celui de llama.cpp est gonflé par `--cuda-graph-trace=node`. Hors nsys, il reste au plus ≈ 0,6 ms.
3. **Bilan de l'écart de 1,46 ms : ≈ 1,07 ms de trous de lancement et ≈ 0,39 ms de noyaux** (attention +0,39, MMVQ +0,22, petits noyaux −0,22).
4. **Leviers, par ordre recommandé (estimations, à mesurer) :**
   - **(a) Graphe CUDA du décodage qwen35** (plan r5-graphe-qwen35.md) :
     - KV append en `_dpos` et positions résidentes sur le GPU, ce qui supprime les 16 HtoD ;
     - état conv sans DtoD ;
     - capture du token.
     - **Gain : −1,0 à −1,3 ms par token**, soit 24,45 → ≈ 23,2 ms, ≈ 43 t/s. Le chemin MTP en profite aussi (vérification et brouillon). gfx906 reste hors graphe.
   - **(b) Attention de décodage sm_86** : un nouvel impl d'`attention_decode` q8_kv, avec des blocs de 128 à 256 threads et une réduction dans le bloc, sur le modèle de flash_attn_ext_vec. Ligne de dispatch sm_86 seulement, cert compris.
     - **Gain : ≈ −0,35 ms par token**, mécanique et indépendant de (a).
   - (a) + (b) donnent ≈ 22,8 ms, soit **≈ 44 t/s, contre 43,5**.
   - **Second ordre** : Q8_0 gate_up (−0,09), tête Q6_K batched à n = 1 (−0,05, r10m-pistes §3).
