# r10m — pistes après le batched (lecture et cuobjdump seulement, sans GPU ; base 81632285)

1. **Q5_K, Q4_K et Q3_K à n = 5 (1,3 à 1,8 × m = 1)** : ce n'est pas un problème de registres au sens d'un débordement, mais de déquantification par colonne, plus l'occupation pour Q3_K.
   - **Registres** (`cuobjdump -res-usage`, n = 5) :
     - IQ4_XS 118, Q5_K 104, Q4_K 98, Q6_K 102, Q3_K 168, avec STACK et LOCAL à 0 partout ;
     - à m = 1, 40 registres pour tous.
   - **Occupation** : avec des blocs de 32 threads, le plafond de 16 blocs par SM limite tous les types à 16 warps par SM. Seul Q3_K (168 registres, 4 lignes par bloc) tombe à 12. Or IQ4_XS, au même plafond, atteint 1,11 : l'occupation n'explique donc pas Q4_K et Q5_K.
   - **Opérations par colonne et par ligne, pour 32 poids** :
     - IQ4_XS : 8 dp4a, 1 conversion int → float et 2 FMUL ;
     - Q4_K / Q5_K : 8 dp4a, 4 conversions et 4 FFMA (termes d et dmin, deux sous-blocs), soit environ deux fois plus d'instructions non-dp4a ;
     - Q3_K : 16 dp4a (4 plans, la correction −4 en dp4a), 4 conversions et 4 FFMA.

     Le coût par colonne n'est donc pas amorti par le chargement unique des poids : ces types sont limités par l'émission d'instructions, pas par la bande passante (Q4_K à n = 5 : 86 µs, pour un plancher de lecture d'environ 54 µs).
   - **llama.cpp** (`vec_dot_q4_K_q8_1_impl_vmmq` et `_q5_K_`) fait exactement les mêmes opérations par colonne, avec 64 threads × 2 lignes à ncols de 5 à 8. On peut donc s'attendre au même rapport, mais **c'est à mesurer** : aucune trace llama.cpp à n > 1 n'existe (`r5/lc.sqlite` vient d'un décodage sans MTP). Il faut un nsys de llama.cpp en MTP-4 et lire `mul_mat_vec_q<Q4_K, 5, …>`.
   - **Levier possible** : sortir de la boucle des colonnes ce qui ne dépend que de la ligne. Le terme dmin·Σu par sous-bloc, et Σu lui-même (dp4a contre 0x01010101), peuvent être précalculés une seule fois par colonne et par super-bloc, puis partagés par les R lignes. La somme d'en-tête Q8_1 essayée pour Q5_K n'a rien donné, donc le gain reste incertain. Pour Q3_K, R = 4 a déjà aidé, et la correction −4 peut aussi passer par Σu précalculé.

2. **Préremplissage MMQ sur sm_86** :
   - **Aujourd'hui** (batched par paquets de 8) : 67 tokens donnent 9 lancements par projection (8 × 8 + 3). À n = 8, environ 12 à 18 µs par token et par projection (IQ4_XS 93 µs / 8, Q5_K 139 µs / 8), contre 57 à 73 µs avant. Les 1 167 ms de MMVQ du préremplissage r10s/mtp tomberaient donc vers **≈ 250 à 300 ms** (estimation, à mesurer par le Vibe sur le TTFT).
   - **En MMQ**, au débit de préremplissage de llama.cpp sur la 3090, de l'ordre de 1 000 tokens/s pour ce 27B (non mesuré ici), on serait vers **≈ 60 à 80 ms pour 67 tokens**, et l'écart grandit avec l'invite : ≈ 2 s contre ≈ 0,5 s pour 512 tokens.
   - **Base existante** : les MMQ dp4a multi-backend (`qmatmul_<t>_mmq_lds_tiled`, `_mmq_wave64`) sont déjà certifiés sur sm_86, mais seulement en BENCH_REFERENCE. Plus le MMA Q4_0.
   - **Étapes** :
     - mesurer ces MMQ à m = 67 et 512 sur la RTX ;
     - couper la ligne batched en m = "2..T" et la ligne MMQ en m = ">=T+1" (T à fixer à la mesure, sans doute entre 16 et 32), descripteurs compris ;
     - gfx906 reste inchangé.

3. **Q6_K batched n = 1 comme ligne m = 1** :
   - **Mesure** (exp_cfg, 32 threads × 2 lignes) : 87,3 contre 90,6 µs à 17408×5120 et 89,2 contre 94,2 µs à 5120×17408, soit −4 à −5 %.
   - **Gain estimé** : la tête de 248 320 lignes, environ 1,2 ms, ferait ≈ −50 µs par token, et ≈ −200 µs par cycle MTP-4 avec ses 4 têtes de brouillon. Le reste du Q6_K (attn_k, attn_v, attn_output, ssm_out et le brouillon nextn) gagne aussi environ 4 %.
   - **Prérequis** : la ligne m = 1 sert aussi à `mmvq_f16_direct` (K et V Q6_K en F16 direct), et `mmvq_m1_recipe` exige un `RecipeKind::Mmvq`. Il faut donc :
     - une cellule Mmvq à part, `mmvq_q6_k_dp4a_vdr2` (entrées `_q8_1` et `_q8_1_f16` sur `mmvq_vdr_batched_row<Q6K, 1, 32, 2, OutT>`, recette à 32 threads et 2 lignes par bloc) ;
     - un Dtype de sweep une ligne pour son certificat (15 formes) ;
     - `mmvq_f16_direct_parity` sur la RTX ;
     - un texte glouton identique, ou un écart documenté.
   - **Priorité basse**, mais le changement est mécanique.
