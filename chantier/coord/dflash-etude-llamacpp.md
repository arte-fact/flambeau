# DFlash / DFlash2 dans llama.cpp (ffc87c3a7) — notes d'étude (lecture de code, rien d'exécuté)

Matière première pour la note d'analyse DFlash (étape 5). Chemins relatifs à `~/llama.cpp`.
Relevé par un sous-agent en lecture seule ; les numéros de ligne sont à revérifier avant de
les citer dans une consigne.

## Ce que c'est
- Brouillon par diffusion de bloc : un petit modèle séparé émet un bloc entier en UNE passe,
  conditionné par des états cachés de la cible injectés dans son propre cache KV
  (`docs/speculative.md:57-61`, `src/models/dflash.cpp:248-250`).
- Entrée du brouillon : `[id_last, <mask> × n_max]` aux positions `pos0..pos0+n_max`, un seul
  `llama_process` (`common/speculative.cpp:1555-1565`). Au plus `block_size-1` tokens.
- DFlash2 = DFlash + convolution causale locale dans chaque couche + « sélecteur de candidats »
  bas rang sur le top-k de chaque position ; l'hôte parcourt ce treillis
  (`docs/speculative.md:88`, `common/speculative.cpp:1498`, `:1600-1648`).
- Variante détectée par la présence de `selector_hidden.weight` (`src/models/dflash.cpp:39-45`).
- Code : `src/models/dflash.cpp` (540 lignes), `src/models/dflash2.cpp` (224 lignes).

## GGUF `~/models/Qwen3.8-27B-DFlash2-Q8_0.gguf`
- 2,06 Go, 81 tenseurs, `general.architecture = "dflash"`, étiquette 1,9B.
- block_count 5 ; embedding_length 5120 ; feed_forward_length 17408 ; 32 têtes, 8 têtes KV,
  dim 128 ; block_size 8 ; conv_kernel_size 2, conv_group_size 16 ; selector_rank 256,
  selector_top_k 16 ; target_layers [6, 20, 34, 48, 62] ; attention non causale ;
  sliding_window 2048 ; rope sections [64,0,0,0], freq_base 1e7 ; mask_token_id 248070.
- Matmul en Q8_0, normes et bases de conv en F32.
- Globaux : `fc.weight` [5120, 25600], `enc.output_norm`, `output_norm`,
  `selector_hidden` [256, 5120], `selector_predecessor` et `selector_successor` [248320, 256].
- Par couche : attn_norm, attn_q [4096,5120], attn_k / attn_v [1024,5120], normes q/k [128],
  attn_output [5120,4096], ffn_norm, ffn_gate / ffn_up [17408,5120], ffn_down [5120,17408],
  attn_conv_base / ffn_conv_base [2,2,5120], attn_conv_proj / ffn_conv_proj [1280,5120].
- PAS de `token_embd` ni de `output.weight` : pris dans la cible (`dflash.cpp:224-246`).
  L'embedding du masque est la ligne 248070 du `token_embd` de la cible ; les logits passent
  par la lm_head de la cible (Q6_K, 1,04 Go).

## Entrées
- États cachés de la cible : l'ENTRÉE des couches 6/20/34/48/62 (flux résiduel avant
  `attn_norm`, `src/models/qwen35.cpp:164-168`), concaténés (25 600), puis `fc` et RMSNorm.
- Cache KV propre, fenêtre glissante par couche.
- Positions : celle du token cible, `{p,p,p,0}` ; M-RoPE à sections temporelles seulement.
- Pas d'embedding de bruit ni de pas de temps trouvé : un seul pas de débruitage.

## Graphe
A. Injection (une fois par token décodé par la cible, préremplissage et vérification compris ;
   `build_inject`, `dflash.cpp:285-342`) : `g = RMSNorm(fc · features)` ; par couche
   `K = rope(RMSNorm_k(Wk·g))`, `V = Wv·g`, écrits directement dans le KV de la couche. Ni
   attention ni FFN.
B. Brouillon (lot de 1 + n_max lignes, une fois par cycle ; `build_layers`, `dflash.cpp:356-465`) :
   embedding ; par couche : moitié attention (norme → conv_proj → conv côté 0 → Q/K/V, normes
   q/k, rope → attention NON causale sur cache injecté + bloc → conv côté 1 → résiduel), moitié
   FFN (même schéma, FFN SiLU) ; `output_norm` ; lm_head sur TOUTES les lignes ; sélecteur.
- Pas d'attention croisée ni de GDN.
- Conv (`dflash2.cpp:53-116`) : noyau dynamique par token, 320 groupes × 2 prises × 2 côtés
  = 1 280 coefficients ajoutés à `base` ; mélange causal à 2 prises dans le bloc.
- Sélecteur (`dflash2.cpp:120-209`) : top-16 des logits par position (score unaire = logit) ;
  `gate = selector_hidden · h` ; score de paire = `successor[cand]ᵀ (predecessor[prev] ⊙ gate)`
  + unaire ; la position 1 prend le token d'ancrage comme prédécesseur. Chaque ligne renvoie
  16 ids + 16×16 scores. L'hôte fait l'argmax ligne après ligne (`speculative.cpp:1600-1607`).

## Intégration serveur
- `-m cible.gguf -md DFlash2.gguf --spec-type draft-dflash2 --spec-draft-n-max 7`
  (`docs/speculative.md:96-97`). `-md` obligatoire.
- n_max par défaut 3, borné à block_size-1 = 7.
- Vérification : correspondance gloutonne par défaut ; rejet probabiliste en option.
- Côté cible : capture des entrées de 5 couches par token, de préférence dans un tampon GPU
  lu par le brouillon ; instantanés d'état récurrent `n_rs_seq = n_max`.
- Après chaque brouillon, le serveur retire du KV du brouillon les cellules du bloc de bruit ;
  le lot de vérification réinjecte les vraies caractéristiques.
- Différences avec draft-mtp : conditionnement par 5 couches intermédiaires au lieu du dernier
  état caché ; une passe par bloc au lieu d'un enchaînement ; `accept()` sans état.

## Cibles et contraintes
- Toute architecture qui expose les entrées de couches (qwen35, qwen35moe, qwen3next, gemma4…).
- `target_layers` obligatoire ; lm_head et embeddings de la cible obligatoires ; KV FP8 non
  géré pour cette architecture.

## Chiffres trouvés
- Planchers des tests nocturnes (Qwen3.8-27B-Q4_K_M + dflash2 Q8_0, n_max 7) : acceptation
  ≥ 0,518, longueur moyenne acceptée ≥ 4,54 (`tools/server/tests/unit/test_spec_methods.py:70-76`) ;
  planchers ≈ 3 points / 0,2 sous les valeurs mesurées.
- Aucun chiffre de débit dans le dépôt.

## Postes de coût pour flambeau
- Par brouillon : 5 couches × 8 lignes ; ≈ 354 Mo de Q8_0 lus par couche, soit ≈ 1,77 Go par
  brouillon (calcul du sous-agent à partir des tailles de tenseurs, à revérifier).
- lm_head de la cible (1,04 Go) sur les 8 lignes.
- Sélecteur sur GPU : top-16 sur 248k logits × 8 lignes, lectures de lignes dans
  `selector_successor` / `selector_predecessor`.
- Par token de la cible (préremplissage et vérification) : `fc` (25600 → 5120, 139 Mo) et
  5 × (Wk + Wv).
- Capture côté cible : 5 × 5120 flottants par token.

## ROADMAP flambeau (`doc/ROADMAP.md`)
- L250 : « B5 — Speculative decode (MTP, then DFlash): close the gap to llama.cpp ».
- L305-308 : « (4) DFlash2 drafter (dflash GGUF arch: 5 layers, block 8, target taps at layers
  6/20/34/48/62, selector rank 256 / top-k 16) — reuses this verify cycle; (5) sm_86 e2e. »
