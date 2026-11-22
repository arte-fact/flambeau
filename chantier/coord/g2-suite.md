# g2 — suite (trace g2/g2.sqlite, bin_G2_CUDA, mtp 0 ; sans GPU)

1. **Le lancement est sérialisé : aucun chevauchement entre tokens.** Médianes sur 198 tokens, en µs, à partir de la fin du dernier noyau du token N :
   - 0 → 209 : copie DtoH de **993 280 octets, soit toute la ligne de logits F32** (248 320 × 4). La mémoire hôte est paginable, donc l'appel est bloquant.
   - 326 : le `cuStreamSynchronize` rend la main.
   - 326 → 656 : **330 µs d'hôte**, pour l'échantillonnage CPU sur 248 k logits puis la mise en file.
   - 657 → 672 : HtoD de 4 + 8 octets (position, contexte).
   - 672 : `cuGraphLaunch`, qui dure 2 535 µs. Le premier noyau du token N+1 démarre au retour de l'appel (3 208).
   - À l'intérieur du graphe, il ne reste que **120 µs de trous pour 1 555 nœuds** : le graphe lui-même fonctionne.
   - En mtp 0, le chemin d'échantillonnage GPU de s1 n'est pas utilisé : il ne couvre que la vérification MTP.
2. **La part de nsys.** Sous nsys, la période est de 26,27 ms par token (médiane 25,68), contre **23,30 ms hors nsys** (1/42,91) : nsys ajoute ≈ 2,97 ms. Cela tient presque entièrement au mode `--cuda-graph-trace=node`, qui instrumente les 1 555 nœuds à chaque lancement : `cuGraphLaunch` passe à 2,6 ms. Hors nsys, l'inactivité réelle vaut 23,30 − 22,25 = **≈ 1,05 ms par token**. Elle se décompose, en estimation, en 0,12 de trous internes, ≈ 0,33 de copie des logits avec sa synchro, ≈ 0,33 d'hôte, ≈ 0,2 de captures amorties et un reste de lancement. Les 3,99 ms mesurées sous nsys n'ont donc pas de sens hors nsys.
3. **llama.cpp** fait la même chose dans le principe, sans chevauchement non plus : il recopie les logits, échantillonne sur le CPU, puis met à jour et lance le graphe. Sa copie va dans un tampon hôte épinglé et son échantillonneur glouton est léger : ≈ 0,61 ms d'inactivité par token, contre ≈ 1,05 chez nous. L'écart de 0,44 ms correspond à notre copie paginable de 1 Mo, à nos 330 µs d'hôte et à nos recaptures.
4. **Leviers classés par gain (estimations, à mesurer) :**
   - **(1) Échantillonnage glouton sur le GPU et lancement asynchrone en mtp 0 : −0,6 à −0,9 ms par token, soit ≈ 44,5 à 45 t/s.**
     - Le principe : l'argmax tourne dans le graphe, avec le noyau `sampler_argmax_f32` déjà certifié des deux arch. Il écrit le token directement dans `decode_ctx_i32[1]`, et seuls 4 octets reviennent, dans une mémoire épinglée et en asynchrone. On lance N+1 avant de lire le token N ; si N est un token d'arrêt, on jette le résultat de N+1.
     - Ce que ça supprime : la copie de 1 Mo, la synchro, et les 330 µs d'hôte, qui sortent du chemin critique.
     - Repli sur le chemin actuel dès qu'il y a une pénalité, un biais, des logprobs ou une température non nulle, comme pour s1.
     - Un premier pas sans pipeline, l'argmax sur le GPU avec seulement 4 octets relus, vaut déjà ≈ −0,5 ms.
   - **(2) Attention (b) : ≈ −0,35 ms par token.** Elle est à 35,5 µs × 16, contre 13,6 µs chez llama.cpp, à un contexte d'environ 200 tokens. C'est indépendant du lancement.
   - **(3) Grille figée à 32 chunks : ≤ −0,2 ms par token**, soit 46 ms d'instanciation sur 199 tokens, et seulement en dessous de 1 024 tokens de contexte. Au-delà, il y a déjà 32 chunks et aucune recapture.
5. **Recommandation : (1) d'abord.** C'est le plus gros gain, il ne crée aucun noyau et vise exactement l'écart de lancement qui reste contre llama.cpp (−1,4 %). Le gain attendu dépasse llama.cpp, à 43,5 t/s. Le MTP en profite peu, puisque sa vérification relit déjà les tokens sur le GPU. Ensuite (b), puis la grille figée si on vise les contextes courts.
