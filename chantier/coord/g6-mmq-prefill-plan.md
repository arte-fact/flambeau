# g6 — préremplissage MMQ sur sm_86 (piste 2 de r10m-pistes ; plan, sans GPU ni code)

1. **Poids de qwen35 27B UD-Q4_K_M** (inspect-gguf, hors F32) : Q5_K ×131, IQ4_XS ×117, Q8_0 ×106, Q4_K ×104, Q6_K ×30, Q3_K ×7, IQ4_NL ×7, IQ3_S ×4. Aujourd'hui, sm_86.toml n'a pour chaque type que m = 1 (MMVQ) et m ≥ 2 (`…_batched`, par paquets de 8) : un préremplissage de 67 tokens fait 9 lancements par projection.

2. **MMQ déjà certifiés sur sm_86** (BENCH_REFERENCE_KERNELS_SM86, tous pass=true, formes 64, 256 et 4096) :

   | Type | Impl | Registres |
   |---|---|---|
   | Q5_K | `qmatmul_q5_k_mmq_wave64` | 80 |
   | IQ4_XS | `qmatmul_iq4_xs_mmq_wave64` | 56 |
   | Q8_0 | `qmatmul_q8_0_mmq_wave64_tile16` (et `_oracle`) | 40 |
   | Q4_K | `qmatmul_q4_k_mmq_lds_tiled` (4 warps, LDS) et `_wave64` | 84 / 72 |
   | Q6_K | `qmatmul_q6_k_mmq_wave64` | 56 |
   | Q3_K | `qmatmul_q3_k_mmq_wave64` | 56 |
   | IQ4_NL | `qmatmul_iq4_nl_mmq_wave64` | 48 |
   | IQ3_S | `qmatmul_iq3_s_mmq_wave64` | 56 |

   Côté activation, `quantize_f16_q8_1_mmq` est certifié aussi. Tensor cores : seul Q4_0 a un MMA (`mmq_q4_0_mma`).

3. **Types manquants pour qwen35 27B : aucun.** Les 8 types de poids ont un MMQ dp4a certifié. Ce qui manque :
   - les lignes de dispatch m ≥ T+1 (les impls passent de BENCH_REFERENCE à QMATMUL_SM86) ;
   - la vérification que le lanceur CUDA de qmatmul sait les dispatcher (activation Q8_1 MMQ ; voir `reads_mmq_activation`) ;
   - plus tard seulement, des MMA int8 pour les K-quants. Ce sont des noyaux neufs, à décider après la mesure : llama.cpp sur Ampere passe par MMA.

4. **Coupure par type.** `m = "2..T"` → `…_batched` actuel ; `m = ">=T+1"` → MMQ (`lds_tiled` pour Q4_K si la mesure le confirme, `tile16` pour Q8_0, `wave64` ailleurs). Descripteurs et sm_86.toml dans le même commit ; gfx906 inchangé.
   - **Trouver T** : microbench par type sur les formes de qwen35 (5120→17408 gate/up, 17408→5120 down, attention q/k/v/o et GDN, Q6_K 5120 lignes ; hors tête 248 320, qui reste m = 1), pour m ∈ {2, 4, 8, 12, 16, 24, 32, 48, 64}, en temps total batched contre MMQ.
   - T = le plus petit m à partir duquel le MMQ gagne sur toutes les formes du type, pris un cran au-dessus pour la marge. Je m'attends à 16 à 32 (r10m), mais c'est à mesurer.
   - **Point d'attention** : MTP-4 vérifie à m = 5 par la ligne batched ; il faut T ≥ 8 pour ne pas la toucher.

5. **Mesures** :
   - **Microbench du point 4** ;
   - **Bout en bout**, avec des invites de 67, 512 et 2 048 tokens et `--prefill-ubatch 512` :
     - TTFT et temps de préremplissage (nsys : MMVQ ou MMQ par projection), contre bin_G4 et contre llama.cpp `pp` à la même longueur ;
     - textes gloutons : identiques en décodage m = 1, mais le préremplissage change l'ordre des sommes, donc l'écart est à documenter (premier token, divergence tardive) ;
     - cert-check vert sur les deux arch.
   - **Estimation r10m, à confirmer** : pour 67 tokens, on passerait d'environ 250 à 300 ms en batched (contre 1 167 ms avant le batched) à environ 60 à 80 ms en MMQ ; l'écart grandit avec l'invite.
