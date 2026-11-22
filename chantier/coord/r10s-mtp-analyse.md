# r10s-mtp — lecture de la trace MTP-4 (Fable)
Requête « math », 200 tokens, 45 cycles, acceptation 87,6 %. Client 4,72 s = **105 ms par cycle** ; noyaux 4,26 s = 95 ms par cycle.
Comparaison : un token m = 1 sans MTP ≈ 27 ms de noyaux (r10s/D). llama.cpp MTP-4 à 78 t/s ≈ 4,4 tok/cycle → ≈ 56 ms par cycle.

Par cycle (somme / 45) :
- batched_n5 (vérification) : 2 121 ms → **47 ms** (IQ4_XS n5 131 µs contre 49 µs à m = 1 : ×2,7 ; une vérification bien faite, limitée par la bande passante, devrait coûter ≈ 1,2 × m = 1, soit ≈ 30 ms).
- MMVQ m = 1 hors tête : ≈ 1 520 ms → **34 ms**. Inexpliqué : IQ4_XS 17408 lignes à m = 1 = 4 087 appels, soit **91 par cycle** (le FFN en a 40 par forward complet) ; Q5_K 17408 : 42 par cycle ; Q4_K 17408 : 42 par cycle ; Q3_K 17408 : 39 par cycle. Ça ressemble à **plus d'un forward complet m = 1 de la cible par cycle**, en plus de la vérification.
- Têtes Q6_K 248320 lignes à m = 1 : 178 appels (≈ 4 par cycle) → 4,8 ms.

## Correction (Opus, r10s-mtp-chemin.md §7-9)
Les MMVQ m = 1 « inexpliqués » sont le **préremplissage** de l'invite (67 tokens), traité ligne par ligne sur sm_86 faute de lignes MMQ (1,31 s avant le 1er cycle). Mon calcul par cycle les incluait à tort. **Hors préremplissage : 75,1 ms par cycle** (llama.cpp ≈ 56). lp.py mesure hors TTFT. Vérification : 53 ms par cycle (row-tile 48,1 + boucles Q3_K/IQ4_NL 3,7 + Q8_0 1,3).
Ordre retenu : (1) batched R10s n = 2..8 sur sm_86 (estimation −22 à −26 ms/cycle) ; (2) préremplissage sm_86 en MMQ (TTFT) ; (3) brouillon, petits lancements, snapshot GDN.
