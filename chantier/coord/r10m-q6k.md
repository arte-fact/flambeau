# r10m — les appels Q6_K m = 1 de la trace MTP-4 (r10m/m.sqlite, après la 1re vérification, 44 cycles)

- **Ce sont tous des appels du brouillon (couche nextn, poids Q6_K), aucun de la cible :**
  - 173 têtes à vocabulaire complet (248 320 lignes), soit **3,93 par cycle**, à 1 225 µs : **4,82 ms par cycle** ;
  - le reste de la couche nextn, ≈ 4 par pas : gate et up 17408 (354), down et o 5120 (357), q 12288 (180), soit 1,49 ms par cycle.

  Hors préremplissage, le cycle fait 47,7 ms de mur pour 39,0 ms de noyaux, soit **8,6 ms de GPU inactif par cycle**.
- **llama.cpp fait pareil pour la tête** : `qwen35.cpp:662-668` calcule `build_lora_mm(model.output, …)` sur tout le vocabulaire, à chaque pas de brouillon (`ctx_dft`, m = 1). Le Q6_K m = 1 y coûte 1 175 µs (r5) contre 1 225 chez nous : la tête n'est donc pas un écart, à 4 % près.
- **La vraie différence est après la tête.**
  - llama.cpp échantillonne le brouillon **sur le GPU** : `draft.backend_sampling = true` par défaut (common.h:351), avec une chaîne top-k 10 attachée à `ctx_dft` (speculative.cpp:1876). Il ne remonte que les k candidats.
  - Chez nous, chaque pas fait une copie GPU→hôte de 970 Ko vers de la mémoire paginée, une `Stream::synchronize`, puis `sample()` sur l'hôte sur 248 320 logits (mtp.rs:226-247).
  - La trace montre **≈ 0,70 ms (médiane) de GPU inactif après chaque tête du brouillon, soit 3,4 ms par cycle**, et 3,0 ms après la tête de vérification (copie de 5 × 970 Ko puis 5 échantillonnages hôte).

**Leviers, du plus rentable au moins rentable (estimations, à mesurer) :**
1. **Échantillonnage du brouillon sur le GPU.**
   - Un noyau argmax, ou top-k avec softmax sur k, sur `logits_f32_dev` juste après la tête du brouillon. On ne copie plus que (token, q) et les k probabilités, au lieu de 970 Ko, et on ne trie plus 248 320 valeurs sur l'hôte.
   - En glouton (température 0), le texte doit rester identique. Avec une température non nulle, il faut que `mtp_sample_factory` passe par la même chaîne restreinte au top-k, comme llama.cpp.
   - Gain : la majeure partie des 3,4 ms par cycle.
2. **Même chose pour la vérification.** Argmax ou top-k par ligne sur le GPU pour les 5 lignes, et la décision acceptée/rejetée sur l'hôte ne porte plus que sur k candidats. Gain : de l'ordre des 3,0 ms par cycle.
3. **Ligne Q6_K m = 1 avec le trait batched à n = 1** (r10m-pistes §3 : −4 à −5 %). Gain ≈ 0,25 ms par cycle (4 têtes et la couche nextn), et ≈ 50 µs par token hors MTP.

**Total estimé** : 1 + 2 + 3 ≈ 5 à 6,5 ms sur 47,7 ms, soit environ 41 à 43 ms par cycle. À tokens par cycle constants (72 t/s × 47,7 ms ≈ 3,4), on serait vers **≈ 80 à 84 t/s** (estimation, objectif 78).

**Hors levier** : réduire la tête du brouillon (vocabulaire élagué) diminuerait les 4,8 ms, mais llama.cpp ne le fait pas et l'acceptation changerait. Pas prioritaire.
