# g5 — attention de décodage sm_86, KV Q8 (plan, sans code)

Écart mesuré (coord/g-nomtp.md) : `attention_decode_q8_kv_splitk_chunk` prend 35,6 µs × 16 couches (+2,5 µs de combine), contre 13,6 µs pour `flash_attn_ext_vec<256>` chez llama.cpp. Gain visé : ≈ −0,35 ms par token.

1. **Diagnostic.** Notre corps (kernels-shared/include/attention_q8_kv.cuh:114-170) utilise des blocs de head_dim/4 = 64 threads par (tête, chunk) et parcourt les tokens un par un. Chaque token paie une réduction de bloc avec `__syncthreads` (attn_q8_step, :41) : le noyau est limité par la latence, à 2 warps par bloc.

2. **Forme de référence** (llama.cpp ffc87c3a7, ggml/src/ggml-cuda) :
   - fattn-vec.cuh:4-11 : 128 threads (4 warps) par bloc, un bloc par tête Q (24 blocs au total dans la trace r5).
   - fattn-vec.cuh:82-90 : sur CUDA, avec D = 256 et K/V Q8_0, nthreads_KQ = nthreads_V = 32. Un warp entier calcule le produit Q·K d'un token : réduction par `warp_reduce_sum`, sans `__syncthreads`.
   - fattn-vec.cuh:176-205 : Q est quantifié en q8_1 une fois, en mémoire partagée ; produit scalaire dp4a via `vec_dot_fattn_vec_KQ_q8_0` (fattn-common.cuh:333).
   - fattn-vec.cuh:273-279 : la boucle KV avance par tuiles de 128 tokens (un token par lane et par warp), entrelacées sur gridDim.y blocs parallèles.
   - fattn-vec.cuh:282-330 : scores en registres et softmax en ligne par warp ; seul un `syncwarp` est nécessaire.
   - fattn-vec.cuh:350-400 : accumulation de V avec `dequantize_V_q8_0` (fattn-common.cuh:639).
   - fattn-vec.cuh:437-470 : fusion des 4 warps en fin de bloc, via la mémoire partagée.
   - Si gridDim.y > 1 : fusion par `flash_attn_combine_results` (fattn-common.cuh:975), avec un nombre de blocs parallèles choisi par occupation (fattn-common.cuh:1198-1231).
   - La sélection de ce noyau se fait dans fattn.cu:702-742.

3. **Impl proposé** : `attention_decode_q8_kv_vec_dpos`, un noyau CUDA seulement (kernels-cuda/src/kernels/), qui reprend la forme du point 2 avec nos types (`flambeau_block_q8_0`, Q F16).
   - **Structure** : 128 threads, une tête Q par bloc, grille (n_heads_q, n_par).
   - **Position sur le GPU** : elle est lue dans `decode_ctx_i32[0]` (ou passée par l'hôte hors graphe, même noyau). n_par est gravé : les tuiles au-delà de la queue de KV sont vides, comme pour la grille figée de g4. On n'a donc ni recapture, ni témoin de chunk_size à suivre.
   - **Fusion** : si n_par > 1, le combine existant, qui ne dépend pas de la position (format m/s/o des partiels inchangé). Si n_par = 1, sortie directe, sans second lancement.
   - **Choix de n_par** : n_par = 1 jusqu'à ≈ 1 k tokens (24 blocs suffisent, puisque llama.cpp est à 13,6 µs avec gridDim.y = 1), puis 4 à 8 au-delà, à calibrer au sweep.

4. **Sélection (règle 13 et dispatch).** Ajouter une ligne `[[attention_decode]]` à dispatch/cuda/sm_86.toml : dtype "Q8_0", dtype_q "F16", shape m = "1", head_dim 256, impl "attention_decode_q8_kv_vec_dpos_sm_86", cert certs/cuda/sm_86/attention_decode_q8_kv_vec_dpos_sm_86.json. Plus l'entrée DIRECT_CALL_KERNELS_SM86, le stem dans CUDA_ONLY_STEMS et le sweep dans CUDA_ONLY_SWEEPS.
   - **Prérequis à vérifier** : aujourd'hui, l'attention de décodage CUDA est en appel direct, et aucune ligne attention de sm_86.toml n'est lue. standard_attn.rs doit donc choisir l'impl par la table (ou par un `Ops` qui la consulte), et non par un test d'arch.
   - gfx906 n'a pas de ligne Q8 et garde son chemin : .hsaco et chemin MI50 inchangés.

5. **Correction et graphe.** Le même noyau sert dans le graphe et hors graphe, donc les textes restent identiques avec et sans --no-decode-graph. Ils changent en revanche par rapport à bin_G3a, puisque l'ordre des sommes n'est plus le même : c'est une nouvelle référence. Il faut le dire et vérifier la qualité (sweep contre la référence f32, plus un contrôle de texte : divergence tardive seulement, comme pour m = 5 contre m = 1).

6. **Sweep** `sweep_attention_q8_kv_vec_dpos.rs` :
   - Cas : GQA 24/4/256 aux positions 0, 31, 32, 180, 1023, 1024, 4000 et 8191 ; n_par ∈ {1, gravé > besoin} ; plus GQA 16/2/256 et 4/1/128 pour la portabilité.
   - Référence f32 sur K/V déquantifiés, avec la tolérance du sweep Q8 _dpos existant (5e-2 · max(|ref|, √D)). Écart maximal relevé aussi contre le split-K actuel.
   - Microbench dans le même binaire, µs par appel contre le split-K actuel à 180, 1 k et 4 k tokens : il faut ≥ 2× à 180 avant de brancher (règle « pas de projection sans mesure »). Instantané PMC (occupation, registres) dans le cert.

7. **Ordre.** (i) Noyau, sweep et microbench seuls (commit 1, cert). (ii) Ligne de dispatch et câblage dans standard_attn, sous capture et hors capture (commit 2). (iii) Textes, cert-check, puis mesure du Vibe. Si la microbench est sous 1,5×, résultat nul : le noyau va sous `cfg(unverified)` avec son diagnostic.
