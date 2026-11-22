# g5b — reprendre vec_dpos (boucle déroulée masquée, parallel_blocks sur 82 SM) ? Sans GPU.

**Base mesurée sur la RTX** (microbench de g5, trace r5 pour llama.cpp), attention de décodage qwen35 GQA 24/4, D 256, sur 16 couches et 22,25 ms de noyaux par token :

| Contexte | Split-K actuel (grille 32 du graphe) | Total par token | llama.cpp | Notre vec, n_par = 1 |
|---|---|---|---|---|
| 181 tokens | 32,7 µs | 0,52 ms (2,4 % des noyaux) | 13,6 µs (0,22 ms) | 39,5 µs |
| 1 k tokens | 39,1 µs | 0,63 ms | non mesuré | 149,7 µs (37,7 µs avec n_par = 4) |
| 4 k tokens | 142 µs | 2,3 ms | non mesuré | 110,5 µs avec n_par = 8 |
| 8 k tokens | 274 µs | 4,4 ms | non mesuré | 214,9 µs avec n_par = 8 |

1. **Pourquoi le vec actuel perd.** Les 32 tokens par warp sont traités en série : le `break` empêche le déroulage, et chaque token enchaîne une charge, un dp4a et une réduction. De plus, 24 blocs n'occupent que 24 des 82 SM. Le seul gain mesuré (×1,28 au-delà de 4 k, avec 8 parties) vient du parallélisme ajouté, pas du corps du noyau.

2. **Gain réaliste si la reprise atteint la forme de llama.cpp** (déroulage à la compilation, tokens masqués, et environ 82 à 164 blocs : n_par ≈ 4 à 7 pour 24 têtes) :
   - **À 181 tokens** : de 32,7 à environ 14 à 18 µs, soit −15 à −19 µs × 16 = **−0,24 à −0,30 ms par token** (−1,1 à −1,4 % des 22,25 ms) : de 44,4 à environ 45,0 t/s.
   - **À 1 k tokens** : le split-K ne monte qu'à 39 µs, car ses 768 blocs couvrent déjà les SM. Viser 18 à 22 µs donne **−0,27 à −0,34 ms par token** (environ −1,5 %).
   - **À 4 k à 8 k tokens** : le gain mesuré (−32 à −60 µs × 16, soit −0,5 à −1 ms) augmenterait avec un corps déroulé, à **−1,5 à −2,5 ms par token** (−6 à −10 %). C'est une estimation : elle suppose un corps 2× plus rapide que le vec mesuré à n_par = 8, sans mesure llama.cpp à ces contextes. C'est là qu'est l'enjeu.
   - Le MTP n'en profite pas directement : la vérification à m = 5 utilise un autre noyau.

3. **Coût et risque.**
   - **Coût** : une session, avec le même sweep, la même microbench et le même câblage (prérequis f722655e déjà prêt).
   - **Risque** : les textes changent par rapport à bin_G4, puisque l'ordre des sommes change ; cela redonne une nouvelle référence, à justifier par un contrôle de qualité.

4. **Recommandation : reprendre, avec un budget borné à une session et des seuils fixés d'avance** :
   - ≥ 1,5× contre le split-K à 181 tokens ;
   - ≥ 1,2× à 1 k tokens ;
   - ≥ 1,5× à 4 k tokens.

   Si les seuils sont atteints, on branche par la table pour sm_86 ; sinon, on arrête définitivement (cfg(unverified), comme aujourd'hui). La raison principale est le contexte long, où la différence se compte en millisecondes ; à 180 tokens, le gain ne fait que consolider l'avance sur llama.cpp (44,4 contre 43,5).
