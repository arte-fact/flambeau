# (a) `token_embd` gardé en Q4_K sur le device : plan (lecture du code, base f5d29f30, sans build)

1. **Noyau** `embed_gather_q4_k_f16`, dans kernels-shared/include, sans intrinsèque et sans taille de wave :
   - entrées : `ids` (i32 sur le device), `n` lignes, `hidden`, poids Q4_K bruts ; sortie F16 [n, hidden] ;
   - un bloc par ligne, un thread par élément ; la ligne fait hidden/256 = 20 super-blocs de 144 o, soit 2 880 o ;
   - au décodage capturé, `ids = decode_ctx_i32 + 1`, comme aujourd'hui (embed.rs:53-60) ;
   - sinon, `ids` vient d'un nouveau tampon pool i32 [max_prefill_tokens], rempli par HtoD depuis une copie hôte stable (même schéma que decode_ctx_i32, scratch.rs:1081). Ce chemin remplace la boucle de memcpy par ligne (embed.rs:61-82) sur les deux backends.
2. **Égalité au bit près avec l'hôte.**
   - Calcul de l'hôte (quant/src/dequant.rs:800-826) : `d1 = d·sc`, puis `y = d1·q − m1` en f32 sans FMA (Rust ne contracte pas), puis `f16::from_f32`, arrondi au plus proche pair (loader/primitives.rs:67).
   - Le device contracte par défaut : nvcc a `-fmad=true`, et hipcc contracte aussi ; aucun build.rs ne le désactive.
   - Il faut donc deux primitives `mul_rn_f32` / `sub_rn_f32` par backend (`__fmul_rn` / `__fsub_rn` dans kernels-cuda et kernels-hip/arch_primitives, règle 5), puis `__float2half_rn`, et le même ordre que `get_scale_min_k4` (dequant.rs:702).
3. **Call sites : un seul**, `embed_local` (composites/embed.rs:10). Il dispatche sur le dtype de `EmbeddingWeights.token_embd`, qui devient un poids quant comme `lm_head` (globals.rs:61, `upload_quant_weight`). Il couvre :
   - le graphe de décodage (n = 1) ;
   - le prefill (engine.rs:768) ;
   - le MTP (composites/mtp.rs:112) ;
   - plus tard DFlash2 (bloc de 8, ligne masque 248 070).
4. **Sélection et loader.**
   - Pas de ligne de dispatch : c'est un appel direct choisi par le dtype du poids, comme `embed_gather_f16_dpos` (ops/registry.rs:249).
   - Le loader garde `upload_dequant_to_f16` (globals.rs:27) pour tout dtype sans noyau. On ne fait que Q4_K d'abord, plus le F16 existant.
   - gemma4 (post_scale et lm_head lié) n'est pas touché dans ce lot.
5. **Sweep et cert.**
   - On étend `bench/src/sweep_embed_gather_dpos.rs` : la sortie du device doit être égale **à 0 ULP** à `upload_dequant_to_f16`.
   - Couverture :
     - les 248 320 lignes du GGUF réel ;
     - des blocs synthétiques extrêmes : d/dmin sous-normaux et grands, q = 0 et 15 ;
     - n ∈ {1, 8, 9, 513}, avec les ids 0, vocab − 1 et 248 070.
   - Certs `embed_gather_q4_k_f16_{sm_86,gfx906}`. Les .hsaco existants ne changent pas, puisque c'est un nouveau module. On exige ensuite des textes bit-identiques en mtp 0 et mtp 4, sur les deux arch.
6. **gfx906 TP-2 : oui, l'embedding est répliqué.** En TP, `layer_range` vaut `None`, donc `owns_embed` est vrai sur chaque rang (qwen35-v2/src/loader.rs:75-90). En PP, seul le rang 0 le porte.
7. **Gain VRAM calculé (2 425 → 682 Mio)** :
   - sm_86 solo : −1 743 Mio ;
   - gfx906 TP-2 : −1 743 Mio par rang, −3 486 Mio au total ;
   - gfx906 PP : −1 743 Mio sur le rang 0.

   Ce sont des calculs, à confirmer par nvidia-smi / rocm-smi.
