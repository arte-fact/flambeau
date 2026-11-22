# VRAM RTX : 19,5 Go (flambeau) contre 16,8 Go (llama.cpp), lecture du code sans GPU (base f5d29f30)

Mesures (rtx/RESULTATS_*.md) : flambeau mtp 0 19 519 Mo et mtp 4 19 845 Mo, avec `--ctx-cap 8192` ; llama.cpp sans spec 16 783 Mo et draft-mtp 18 041 Mo, avec `-c 32768`. Poids du GGUF sur disque : 15 691 Mio (inspect-gguf), dont `token_embd` Q4_K 682 Mio, `output` Q6_K 995 Mio, et blk.64 (MTP) 335 Mio.

1. **Embedding en F16 sur le GPU : +2 425 Mio, l'essentiel de l'écart.**
   - `load_embedding` déquantifie `token_embd` (Q4_K) en F16 à l'upload (forward/src/loader/globals.rs:27, appelé par qwen35-v2/src/loader.rs:79-84) : 248 320 × 5 120 × 2 o = 2 425 Mio, contre 682 Mio en Q4_K.
   - llama.cpp garde la couche d'entrée sur l'hôte, donc 0 sur le GPU.
   - Le GPU en a besoin, parce que le graphe de décodage capturé lit sa ligne sur le device via `embed_gather_f16_dpos` (forward/src/core/composites/embed.rs:42-60), et que le MTP partage cet embedding.
2. **KV : flambeau en a moins.** Avec 16 couches d'attention en KV q8 (4 têtes KV × 256 × 2 × 34/32 o), cela fait 34 Ko par token. À 8 192 tokens, flambeau stocke 272 Mio ; llama.cpp, à 32 768, en stocke 1 088. À contexte égal, l'écart hors KV n'est donc pas de 2,7 Go mais d'environ 3,5 Go.
3. **Poids MTP chargés même avec `--mtp 0` : ≈ 394 Mio.**
   - La condition de chargement ne regarde que le GGUF (`nextn_layers > 0 && owns_lm_head`, qwen35-v2/src/loader.rs:247), pas `--mtp`.
   - S'y ajoute `eh_proj` uploadé en F16, soit 100 Mio contre 41 en Q6_K (loader.rs:297-320).
   - Les 326 Mo entre mtp 0 et mtp 4 sont donc des tampons d'exécution MTP, pas des poids.
   - Je n'ai pas vérifié si llama.cpp sans spec alloue blk.64.
4. **Logits : négligeables.** `logits_rows = max(slots, max_draft + 1)` × 248 320 × 4 o (core/scratch.rs:1236-1240), soit 1 Mio en mtp 0 et 4,7 Mio en mtp 4.
5. **Reste non attribué, ≈ 0,7 Go** (3,5 − 2,4 − 0,4) :
   - pool scratch à ubatch 513 : rien que le FFN pèse ≈ 100 Mio (gate et up en F32, 68 Mio ; gated F16, Q8_1 et down), scratch.rs:1225-1234 ;
   - état GDN, ≈ 144 Mio F32 par slot (48 couches × 48 têtes × 128²) ;
   - contexte CUDA, graphes, workspace.

   Ce n'est pas mesuré ; la somme de `allocs` et du pool, écrite au démarrage, le donnerait.
6. **Réductible sans perte :**
   - (a) **`token_embd` gardé en Q4_K sur le device**, avec une variante quantifiée de `embed_gather_*_dpos` : −1 743 Mio, et le graphe reste capturable.
     - Risque : un nouveau noyau et son cert, sur les deux arch. Il doit reproduire exactement l'arrondi de l'hôte (Q4_K → F32 → F16) pour que les textes restent bit-identiques.
     - Les autres utilisateurs (MTP, et plus tard le masque DFlash2) passeraient par le même gather.
   - (b) **Ne pas charger blk.64 quand `--mtp 0`** : −394 Mio, sans aucun risque de justesse. Le MTP reste activable au redémarrage.
   - (c) **`eh_proj` en Q6_K via qmatmul** au lieu de F16 : −59 Mio. Risque : l'écart numérique du chemin MTP, à re-certifier.
   - **Non recommandé :** garder l'embedding sur l'hôte comme llama.cpp. Cela réintroduit une copie hôte → device par token et casse le gather en position device du graphe, donc les gains de g3 et g4.
7. **Pour DFlash2 (+2,1 Go de drafter) :** (a) seul ramène flambeau à ≈ 17,8 Go en mtp 0, et (a) + (b) à ≈ 17,4 Go. Ce sont des chiffres calculés, à confirmer par nvidia-smi.
