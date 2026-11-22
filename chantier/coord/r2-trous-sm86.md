# Règle 2 : les deux trous sur sm_86 (lecture seule, base 292f65b8)

1. **`qmatmul_q4_0_mma` est atteignable, et sa ligne n'existe que côté Rust.**
   - Chemin : `ops/src/launch/qmatmul.rs:122` (le raccourci MMVQ ne vaut que si m < 32, ou k ou n non multiples de 128), puis `:314` `resolve_dispatch_recipe`, puis la ligne Rust `backend-cuda/src/impls/qmatmul.rs:162-171` (Q4_0, m ≥ 128), puis l'arme `RecipeKind::MmqMma` (`qmatmul.rs:268`).
   - Cas concret : tout préremplissage d'un modèle Q4_0 (gemma) avec ubatch ≥ 128 et k, n multiples de 128.
   - Côté TOML, `dispatch/cuda/sm_86.toml:113-119` dit Q4_0 m = "any" → `mmvq_dp4a` : la table Rust et le TOML divergent, et cert-check ne voit pas le MMA.
   - Le cert existe (pass=true, 12 formes), mais aucun sweep du dépôt ne le produit : il a été ajouté à la main par le commit 30c1a751, avec `emitted_at` à 00:00:00Z. Son `dtype_activation` vaut « Q8_1_MMQ », ce qu'une ligne TOML en Q8_1 refuserait.
2. **`attention_prefill_mma` est atteignable, sans dispatch et sans cert enregistré.**
   - Chemin : `forward/src/core/composites/standard_attn.rs:1091` `attn_prefill_f16`, puis `model-ops/src/ops/attn_prefill.rs:79`, puis `ops/src/launch/attention.rs:1527-1528`. Ce dernier prend le noyau dès que head_dim ∈ {128, 256, 512}, que n_q ≥ 16 et que le module est chargé.
   - C'est donc **tout préremplissage CUDA en KV F16**, qwen35 compris (D = 256). Le choix repose sur la présence du module, pas sur la table : c'est un test d'arch déguisé, au regard des règles 1 et 13.
   - L'impl est listé dans `UNCERTIFIED_CERTS` (`backend-cuda/src/impls/mod.rs:68`, « no sweep… yet »). Le fichier cert existe (13 formes), mais rien ne le produit.
3. **Proposition pour `qmatmul_q4_0_mma`** :
   - un sweep (MMA Q4_0 contre la référence F64, activation DS4) qui régénère le cert avec `dtype_activation` Q8_1 ;
   - dans sm_86.toml, la ligne Q4_0 coupée en `m = "1..127"` → `mmvq_dp4a` et `m = ">=128"` → `qmatmul_q4_0_mma`, alignée sur la table Rust ;
   - ensuite, cert-check la couvre.
4. **Proposition pour `attention_prefill_mma`** :
   - un sweep (d128, d256 et d512, n_q ∈ {16, 128, 300}, contre l'oracle F64 de `run_attention_flash_tile_sweep`) ;
   - l'enregistrement en DIRECT_CALL (et son retrait de `UNCERTIFIED_CERTS`) ;
   - le choix par une ligne `[[attention_prefill]]` sm_86, via `dispatch_attention_prefill` sur le modèle du `dispatch_attention_decode` de g5, au lieu de la présence du module.
5. **Pas de cfg(unverified)** : les deux sont des gains mesurés en préremplissage RTX (+18,9 % et +4,8 à +12 % d'après leurs commits), et les certs à la main passent. Il manque la reproductibilité et la couverture par cert-check, pas la justesse. Ordre proposé :
   - (a) les deux sweeps, sans GPU pour le code (≈ 1 session), puis un passage GPU de 2 à 3 minutes ;
   - (b) la ligne TOML Q4_0 ;
   - (c) la sélection `[[attention_prefill]]`.

   Chaque étape fait un commit. gfx906 reste inchangé.
