# Matmul de vérification à petit m (2 à 16) : plan (Opus, rien de codé)

## Ce qui existe déjà
- **sm_86 n'a aucune MMQ routée pour les K-quants** (dispatch/cuda/sm_86.toml) :
  - Q3_K, Q4_K et Q5_K : `m = 1` → MMVQ t128 vdr2 ; `m >= 2` → `*_mmvq_dp4a_t128_vdr2_batched` (« 1 à 8 colonnes par lancement »), y compris en préremplissage ;
  - la seule MMQ dispatchée est `qmatmul_q4_0_mma`, pour `m >= 128`.
- **La MMQ LDS-tiled partagée est déjà compilée pour CUDA.** `kernels-shared/include/mmq_lds_tiled.cuh` utilise des lanes logiques et n'a pas de shuffle, donc il est valable en wave32 comme en wave64. Chaque cellule de `instantiations.toml` produit le `.cu` des deux backends.
  - Sur gfx906, ses variantes `*_lds_tiled_x8` (MMQ_X = 8) sont dispatchées à m = 3..32 pour Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS et IQ4_NL (dee02058).
  - Sur CUDA, il ne manque que l'id d'impl : `RecipeRow.ids` (ops/launch/qmatmul.rs:2380…) ne connaît que `…_gfx906`, et `backend-cuda/src/impls/qmatmul.rs` n'a pas de descripteur pour ces variantes.
  - Preuve que le header compile et passe sur sm_86 : le cert `certs/cuda/sm_86/qmatmul_q4_k_mmq_lds_tiled.json` existe (direct call, MMQ_X = 64).
- **Autres candidats certifiés sur sm_86, mais non dispatchés :** `qmatmul_{q3_k,q4_k,q5_k,q6_k}_mmq_wave64` (direct call).
- **llama.cpp HEAD, `mmq-few.cu` :**
  - MMA int8 `mma.sync` avec un anneau `cp.async`, 2 à 24 colonnes (3 groupes de 8), 16 lignes par bloc (32 sur les grandes matrices), 4 warps qui se partagent K par unités de 256 ;
  - NVIDIA Ampere et plus seulement ;
  - seuil de colonnes (réglé sur RTX 5090) : Q3_K et Q5_K ≥ 3, Q4_K, IQ4_NL et Q6_K ≥ 4, Q8_0 et IQ3_S ≥ 5 ; IQ4_XS reste en MMVQ jusqu'à 8.

## Cible mesurée (nsys V1, cycle MTP-4, m = 5, médiane par lancement en µs)
| dtype | lignes × k | flambeau MMVQ n5 | llama.cpp `mul_mat_q_few` | bande passante (fl / lc) |
|---|---|---|---|---|
| Q5_K | 17408 × 5120 | 96,9 | 77,7 | 633 / 789 Go/s |
| Q5_K | 10240 × 5120 | 59,6 | 48,7 | |
| Q5_K | 6144 × 5120 | 39,6 | 35,4 | |
| Q5_K | 5120 × (6144 ou 17408) | 38,6 | 32,9 | les deux k sont mêlés |
| Q5_K | 1024 × 5120 | 11,8 | 7,1 | |
| Q4_K | 17408 × 5120 | 89,1 | 64,1 | 562 / 782 Go/s |
| Q4_K | 10240 × 5120 | 54,7 | 40,2 | |
| Q4_K | 6144 × 5120 | 36,5 | 30,2 | |
| Q4_K | 12288 × 5120 | 64,2 | 47,6 | |
| Q3_K | 17408 × 5120 | 77,3 | 55,5 | 495 / 690 Go/s |
| Q3_K | 5120 × 17408 | 93,0 | 53,9 | |

Nombre de tenseurs par forme dans Qwen3.8-27B-UD-Q4_K_M (blk 0..63) :
- **Q5_K :** 5120×6144 ×44, 17408×5120 ×28, 5120×17408 ×22, 6144×5120 ×12, 10240×5120 ×12, 1024×5120 ×11, 12288×5120 ×2 ;
- **Q4_K :** 10240×5120 ×29, 17408×5120 ×28, 6144×5120 ×24, 5120×17408 ×10, 12288×5120 ×6, 5120×6144 ×5, 1024×5120 ×1 ;
- **Q3_K :** 17408×5120 ×6, 5120×17408 ×1.

Le plafond, c'est la bande passante : llama.cpp atteint environ 80 % des 936 Go/s à m = 5, flambeau 50 à 67 %.

## Étape 0 : harnais de microbench (codé : 265ea8ef sur v2)
- Nouvelle commande `flambeau qmatmul-bench --arch <a> --dtype <D> --shapes R:K,… --m 2,3,…,16 --impl <id>,… [--iters 200]`.
- Pour chaque (forme, m, impl), elle lance l'impl par sa recette de production (même stem, même entrée, même géométrie que `ops::launch::qmatmul`, choisie par id et non par le dispatch). Les poids sont des octets aléatoires valides et l'activation est dans la disposition que lit l'impl.
- Elle fait 3 lancements de chauffe et `iters` lancements sans synchro entre eux. Le temps vient du profileur (nsys `cuda_gpu_kern_sum` / rocprofv3 `--kernel-trace`), regroupé par (nom, grille) : pas de chrono hôte.
- La quantification d'activation de chaque impl est mesurée à part (`quantize_f16_q8_1` contre `…_mmq`). Après V2 (d), une seule tourne par projection.
- Il faut ajouter les ids CUDA des candidats (`RecipeRow.ids`) dans un `cfg(any(test, feature = "cuda_sweep"))` réservé au bench, sans descripteur : ils restent inatteignables par le dispatch, comme l'exige la règle 2 tant qu'il n'y a pas de cert.

### Candidats
| id | noyau | état |
|---|---|---|
| C0 | `*_mmvq_dp4a_t128_vdr2_batched` (actuel) | dispatché |
| C1 | `mmq_<D>_lds_tiled_x8` (MMQ_X = 8 ; m 9..16 = 2 tuiles en y) | compilé pour CUDA, id à ajouter |
| C2 | `mmq_<D>_lds_tiled_x16` (MMQ_X = 16) | cellule d'instanciation à ajouter, en-tête inchangé, gfx906 aussi |
| C3 | `mmq_<D>_lds_tiled` (MMQ_X = 64 ; Q4_K y64) | Q4_K certifié sm_86 |
| C4 | `qmatmul_<d>_mmq_wave64` | certifié sm_86, direct call |
| R | llama.cpp `mul_mat_q_few`, mêmes formes | référence |

## Commandes (à lancer quand le GPU me revient)
Communes : `B=$FE/target-o/release/flambeau`, `BC=$FE/target-cuda/release/flambeau` (après le build de V2 + étape 0, `-j 8`, un seul à la fois) ; `M=2,3,4,5,6,8,9,12,16`.

**sm_86** (RTX ≤ 50 °C, < 1 Go, un seul processus GPU) :
```bash
S="17408:5120,5120:17408,10240:5120,6144:5120,5120:6144,12288:5120,1024:5120"
for D in Q5_K Q4_K Q3_K; do
  CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=-1 nsys profile --trace=cuda -o $SP/mmqf/sm86_$D --force-overwrite true \
    $BC qmatmul-bench --arch sm_86 --dtype $D --shapes $S --m $M --impl C0,C1,C2,C3,C4
  nsys stats --report cuda_gpu_kern_sum --format csv $SP/mmqf/sm86_$D.nsys-rep > $SP/mmqf/sm86_$D.csv
done
```
**llama.cpp, référence sur les mêmes formes** : on construit seulement la cible `test-backend-ops` dans build-cuda-hd (`cmake --build build-cuda-hd --target test-backend-ops -j 8`), puis :
```bash
build-cuda-hd/bin/test-backend-ops perf -b CUDA0 -o MUL_MAT -p 'type_a=(q5_K|q4_K|q3_K),type_b=f32,m=4096,n=([2-9]|1[26]),k=14336'
```
- Les cas perf de HEAD couvrent Q4_K à 4096×14336 et 14336×4096 pour n = 1..9, 16, 17, 24, 25, 32 et 33, et tous les types à n ≤ 8.
- On mesure aussi C0..C4 à ces deux formes (`--shapes 4096:14336,14336:4096`) pour comparer à forme égale.
- Pour les formes du modèle, la référence reste le tableau nsys ci-dessus (m = 5).

**gfx906** (GPU 1 seulement, `wait_gpu1.sh` avant chaque lancement) :
```bash
for D in Q5_K Q4_K Q3_K; do
  HIP_VISIBLE_DEVICES=1 rocprofv3 --kernel-trace --stats -d $SP/mmqf/gfx906_$D -o $D -- \
    $B qmatmul-bench --arch gfx906 --dtype $D --shapes $S --m $M --impl C0,C1,C2,C3,C4
done
```
- C0 est le chemin de production tel quel (« dispatch », raccourcis compris) : sur gfx906, à m = 2..8, c'est le MMVQ row-tile batched, qui passe avant la ligne x8 de la table ; au-delà, c'est C1.
- On vérifie ainsi la frontière m = 2/3 et le coût de m = 9..16 avec deux tuiles x8 contre une tuile x16.
- **Tête de sortie Q4_K, à m = 1 (gfx906 seulement)** : `mmvq_q4_k_r4`, appelé en dur par `output_head` (coord/gemma-sm86-diag.md), contre la ligne Q4_K m = 1.
```bash
bash $SP/wait_gpu1.sh
HIP_VISIBLE_DEVICES=1 rocprofv3 --kernel-trace --stats -d $SP/mmqf/gfx906_lmhead -o lmhead -- \
  $B qmatmul-bench --arch gfx906 --dtype Q4_K --shapes 262144:3840,131072:5120 --m 1 --impl C0,C5
```
  - Si C5 gagne au-delà du bruit (3 répétitions), on en fait une vraie ligne gfx906 Q4_K m = 1, avec cert. Sinon, la branche est simplement supprimée.
  - Ce choix se fait dans le commit commun des raccourcis.

## Règle de décision
- **Coût de référence :** pour chaque m, on additionne le temps du noyau et de sa quantification sur les tenseurs de la vérification (pondérés par le nombre ci-dessus). On compare C0 au meilleur de C1 à C4.
- **Si un candidat existant bat C0** à m = 5 et à m = 9 (vérification DFlash2), avec un écart supérieur au bruit mesuré sur 3 répétitions, **c'est une simple ligne de dispatch**, sans nouveau noyau :
  - le descripteur CUDA ;
  - `RecipeRow.ids` ;
  - un sweep et un cert sm_86 (`sweep --arch sm_86 --op qmatmul --impl …`, sur toutes les formes du cert) ;
  - les bandes `m` dans sm_86.toml, aux frontières mesurées ;
  - all-dispatch et cert-check verts.
  - Pour gfx906, on ajoute la bande x16 seulement si C2 bat deux tuiles C1 à m = 9..16.
- **Sinon, on porte `mul_mat_q_few` :**
  - une impl sm_80+ dans `kernels-cuda/` (`mma.sync`, `cp.async` : intrinsèques CUDA, donc hors de `kernels-shared`, règle 5) ;
  - les helpers de fragments MMA de `qmatmul_q4_0_mma` sont réutilisés ;
  - Q5_K, Q4_K et Q3_K d'abord, puis Q6_K, IQ3_S, IQ4_NL et Q8_0 ;
  - jusqu'à 3 groupes de 8 colonnes (m ≤ 24, ce qui couvre DFlash2, 9 lignes) ;
  - quantification de l'activation au format du fragment : une impl de quantification de plus, avec son cert ;
  - un cert par dtype, puis une ligne de dispatch par dtype sur la bande mesurée.
  - Ce port tient sur plusieurs sessions. Rien n'est projeté tant que l'étape 0 n'a pas tranché.
- **gfx906 n'est pas concerné par le port**, qui est NVIDIA uniquement, comme chez llama.cpp. On n'y touche que les bandes x8/x16.
- **Raccourci row-tile batched** (`rowtile_batched_k_owns`, décision provisoire de Fable) : à m = 2..8, `qmatmul` choisit cette variante avant la table de dispatch, contre l'esprit des règles 1 et 2. Rien n'est changé avant le microbench.
  - Sur gfx906, à m = 3..8, C0 (le row-tile) est comparé à C1 (x8).
  - Le gagnant devient une vraie ligne de la table, avec son cert, et le raccourci est supprimé.
  - Les paths `mmvq_batched` / Q4_1 / Q8_0 en tête de `qmatmul` sont à inventorier de la même façon dans ce même commit.

## Suite (v3 6b69f618, commit des raccourcis mesurés)
- **Fait** :
  - gfx906 Q4_K : row-tile → ligne à m = 2..4, x8 à 5..8, x16 à 9..16, x8 à 17..32 ;
  - gfx906 Q5_K : row-tile → ligne à m = 2..7, x8 à 8..32 ;
  - gfx906 Q3_K : x8 à partir de m = 2 ;
  - tête de sortie : `qmatmul`, et `mmvq_q4_k_r4` sous `cfg(unverified)`.
- **TODO, non mesuré, raccourci conservé** : row-tile batched pour Q6_K, IQ4_XS, IQ3_S, IQ3_XXS et IQ2_S (bench C0 contre C1 à m = 2..8 avant de le supprimer) ; m = 7 en Q5_K ; m > 16 pour x16 en Q4_K ; chemins batched Q4_1, Q8_0 et Q4_0 en tête de `qmatmul`.
- **Liste du build du prochain créneau GPU** (gfx906, GPU 1, garde) :
  1. build `target-o` (hip_serve,hip_sweep) puis `target-cuda`, `-j 8`, l'un après l'autre ;
  2. `sweep --arch gfx906 --op qmatmul_mmq --dtype Q4_K_lds_tiled_y64_x16` → nouveau cert `certs/hip/gfx906/qmatmul_q4_K_mmq_lds_tiled_y64_x16_gfx906.json` ;
  3. re-sweep `Q3_K_lds_tiled_x8` (bande à partir de m = 2) et `Q4_K_lds_tiled_y64_x8` (forme m = 5) : les formes de sweep gagnent m = 2, 5 et 9 ;
  4. cert-check (gfx906 et sm_86), all-dispatch gfx906 ;
  5. tests sans GPU : `backend-hip` (bandes, toml, certs présents ; `every_descriptor_cert_path_exists_on_disk` reste rouge jusqu'au point 2), `ops` (`recipe_table_tests`), `cli` (`sweep_all`) ;
  6. `sweep --arch sm_86 --op quantize_f16_q8_1_few` → nouveau cert `certs/cuda/sm_86/quantize_f16_q8_1_few_sm_86.json` (F1, comparaison octet par octet à la référence hôte) ; `every_registered_cert_path_exists_on_disk` (backend-cuda) reste rouge jusque-là ;
  7. `sweep --arch sm_86 --op qmatmul_mmq_few_q4_k` → cert `certs/cuda/sm_86/qmatmul_q4_k_mmq_few_mma.json` (F2, enregistré comme référence de bench) ;
  8. F2 bench sm_86, sous nsys, 3 répétitions : `$BC qmatmul-bench --arch sm_86 --dtype Q4_K --shapes $S,4096:14336,14336:4096 --m 2,3,4,5,6,8,9,12,16,17,24 --impl C0,C6` ; bande Q4_K = les m où C6 bat C0 au-delà de la dispersion (règle de mmq-few-resultats.md) ;
  9. qmatmul-bench C0 après le changement : à m = 2..16, C0 doit valoir le meilleur des candidats de mmq-few-resultats.md.
