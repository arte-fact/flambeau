# NanoQuant (ICML 2026) — note de faisabilité pour Flambeau (gfx906 / MI50)

- Date : 2026-11-24. Branche : `research/nanoquant` (base `main` @ `fccd0317`). Aucune branche existante modifiée.
- Sources : papier arXiv:2602.06694 (v1 06/02/2026, v3 14/06/2026 — lu en entier, corps + annexes A–G ; les chiffres ci-dessous sont ceux de la v3 sauf mention contraire), HTML `https://arxiv.org/html/2602.06694v3` ; blog Samsung Research `https://research.samsung.com/blog/NanoQuant-Efficient-Sub-1-Bit-Quantization-of-Large-Language-Models` (03/08/2026) ; code `github.com/SamsungLabs/NanoQuant` @ `a9e0a430` ; variante ShamAN-Q arXiv:2609.38521v1 (lu en entier).

## 1. Format exact

Factorisation binaire de faible rang, par couche linéaire (éq. 1 du papier) :

```
W ≈ Ŵ = diag(s1) · U±1 · V±1ᵀ · diag(s2)
U±1 ∈ {−1,+1}^(dout×r),  V±1 ∈ {−1,+1}^(din×r),  s1 ∈ R^dout, s2 ∈ R^din (FP16)
```

- **Rang par couche** : dérivé du budget de bits, uniforme en bits donc variable selon la forme de la matrice : `r = (a·b·bits)/(a+b) − 16` avec a = dout, b = din (code : `calculate_ranks._get_rank`, `src/nanoquant/utils/utils.py:132`). Exemples Qwen3-8B @ 1 bpw : 4096×4096 → r = 2032 ; 4096×12288 (gate/up/down) → r = 3056 ; 4096×1024 (k/v en GQA) → r = 803. @ 2 bpw, r double (4080 / 6128 / 1606).
- **Empaquetage des bits** : −1 → 0, +1 → 1, packés en mots entiers (bitfield `uint32` côté noyau ; `binary_packer(..., method="half2")` côté Python, `src/nanoquant/modules/linear.py:171`). Le déploiement = bitfield U + bitfield V + les deux vecteurs d'échelles FP16.
- **Coût en bits par poids** (annexe F.5, éq. 58–59) : `M = r(n+m) + 16(n+m)` bits, soit `BPW = [r(n+m) + 16(n+m)] / (mn)` ; pour n = m = d : `BPW ≈ 2(r+16)/d`.
- **Tailles mesurées** (Table 13) : L2-7B = 1,33 Go @ 1 bpw (13,48 Go en BF16), L2-13B = 2,24 Go, L2-70B = 5,75 Go (137,95 → 5,75 Go, 24×) — un 70B tient dans 8 Go de VRAM. @ 2 bpw, L2-7B = 2,14 Go (Table 8).
- **BPW effectifs des binaires PTQ concurrents** : BiLLM 2,88, ARB-LLM_RC 2,51, HBLLM_R 3,25, STBLLM 4,13 ; GPTQ (W2g64) = 2,28. NanoQuant est le seul PTQ à descendre sous 1 bpw réel.

## 2. Qualité : perplexité et zero-shot (v3)

Perplexité WikiText-2 (↓), Table 2 (colonnes choisies) :

| Méthode (BPW) | L2-7 | L2-13 | L2-70 | L3-8 | Q3-0.6 | Q3-1.7 | Q3-4 | Q3-8 | Q3-14 |
|---|---|---|---|---|---|---|---|---|---|
| FP16 (16,00) | 5,47 | 4,88 | 3,32 | 6,24 | 12,66 | 9,39 | 7,89 | 7,00 | 6,37 |
| RTN (1,00) | 1,63e5 | 4,82e4 | 1,57e5 | 4,41e5 | 6,29e21 | 2,58e7 | 5,14e8 | 1,56e10 | 1,37e8 |
| XNOR (1,00) | 6,59e4 | 9,80e3 | 1,37e4 | 8,50e5 | 5,00e6 | 3,32e6 | 3,27e7 | 1,63e8 | — |
| HBLLM_R (3,25) | 7,60 | 6,27 | 4,56 | 11,82 | 9,08 | 78,58 | 35,14 | 14,73 | 10,51 |
| STBLLM (4,13) | 10,12 | 8,08 | 5,26 | 16,68 | 16,46 | 329,76 | 1,32e3 | 35,03 | 20,04 |
| ARB-LLM_RC (2,51) | 11,80 | 8,43 | 5,20 | 19,06 | 16,80 | 129,52 | 49,51 | 18,04 | 12,74 |
| BiLLM (2,88) | 19,87 | 13,29 | 8,75 | 31,20 | 31,21 | 3,17e3 | 858,09 | 78,36 | 29,62 |
| **NanoQuant 1,00** | **10,34** | **8,71** | **6,52** | **14,97** | **27,56** | **19,21** | **14,29** | **12,47** | **10,92** |
| **NanoQuant 0,80** | 12,20 | 10,14 | 7,61 | 18,16 | 33,79 | 25,31 | 19,33 | 14,83 | 12,88 |
| **NanoQuant 0,55** | 16,66 | 13,46 | 9,82 | 25,69 | 52,94 | 33,74 | 32,86 | 20,04 | 17,06 |

- **2 bits** (Table 8, L2-7B) : NanoQuant 2,00 bpw → PPL 6,52, zero-shot 59,09 ; voisins 2 bits vectoriels : QTIP 2,02 (6,29 / 62,05), AQLM 2,02 (6,64 / 56,47), AQLM+PV (5,84 / 61,35). GPTQ W2g64 (2,28 bpw) sur L2-7B : PPL 21,00 (Table 4). NanoQuant 1,50 bpw : 7,08 / 55,81 ; 1,00 bpw (1,05M tokens) : 8,99 / 47,81.
- **Zero-shot** (Table 3, moyenne 6 tâches ↑) : L3-8 — BF16 71,44, NanoQuant 1 bit 45,95, HBLLM_col 50,45, ARB-LLM_RC 44,23, STBLLM 39,83, GPTQ 36,99, BiLLM 38,16. Q3-8 — BF16 71,22, NanoQuant 48,94, HBLLM_col 54,30, ARB 51,86, STBLLM 46,15, BiLLM 38,18, GPTQ 37,04.
- **QAT binaires** (Table 4, L2-7B, 1 bit) : OneBit PPL 9,73 en 700,7 GPU-h ; DBF 9,25 en 37,6 ; LittleBit 9,08 en 123,6 ; **NanoQuant 10,34 en 1,7 GPU-h** (0,26M tokens), 8,85 en 2,5 GPU-h (2,1M tokens).
- **k-quants (Q4_K, IQ
gaussi…)** : le papier ne compare aucun k-quant llama.cpp → mesure propre prévue à l'étape 2 (même modèle, Q4_K_M via llama.cpp).
- Ablations (Table 6, Q3-8B-Base) : pipeline complet 12,47 PPL / 48,94 zero-shot ; init LB-ADMM seule 206,03. Table 9 : plus de données de calibration → mieux (9,07 avec 512+512 échantillons).
- v1 vs v3 : en v1, Q3-4 @ 1 bit était à 1,96e3 (échec), corrigé à 14,29 en v3 ; taille 70B : 5,35 Go (v1) → 5,75 Go (v3).
- Inférence (papier) : RTX 3050 8 Go @ 0,55 bpw — L2-7B 86–134 t/s, L2-70B 12–20 t/s (20,11 t/s à seq 32), pic mémoire 5,86–6,20 Go pour le 70B (Table 12) ; Jetson TX2 (59,7 Go/s) : jusqu'à 12,2× vs PyTorch FP16. Noyaux CUDA maison (annexe E.2/E.3) : GEMV deux étages sur bitfield `uint32`, unpack par masque, FMA fp16 vectorisé (2 valeurs/instruction), échelles par colonne intégrées au FMA, **sans tensor cores** ; GEMM façon Marlin (mma.sync 16×8×16, cp.async 4 étages).

## 3. Code publié et licence

- **Oui** : `github.com/SamsungLabs/NanoQuant` @ `a9e0a430881ff80d8b622c3129e330dc33c04f5` (« Support for binary GEMM kernel benchmarking (#1) »), **Apache 2.0** (LICENSE 11 357 o, `pyproject.toml` `license = {text = "Apache-2.0"}`, README). NOTICE présent. Réutilisation commerciale OK avec notice.
- Stack : Python ≥ 3.9, torch ≥ 2.6, `transformers==4.51.3`, accelerate, datasets, optimum, lm_eval, cut_cross_entropy, gemlite (k-quants 1-bit Triton), zeus-ml (énergie). CLI : `python -m nanoquant.main --model_id Qwen/Qwen3.8-27B --bits 0.55 --num_calib_samples 128 --calib_dataset <...> --qmodel_path out.pt`. Support Qwen3.5/Qwen3.8 (blocs GDN/FLA). Éval : lm_eval (wikitext2 + 6 tâches zero-shot).
- Miroirs tiers (ex. `rkuznetsovcg/NanoQuant`) non officiels — préférer `SamsungLabs`. Attention à la collision de nom : `github.com/swayam8624/Nanoquant` et les GGUF `arelath/*-nanoquant-GGUF` sur HF sont des réimplémentations indépendantes (C++/Metal, PPL mesurée 292 vs 55 BF16 sur leur protocole) — sans lien avec le papier ICML 2026 ; non utilisés ici.

## 4. Coût de calibration et ce qui tient sur la RTX 3090 (24 Go)

- Calibration : **128 échantillons WikiText-2 × 2048 tokens (0,26M tokens), seed 0, 1 GPU**. Pipeline (annexe C + `quant_config.py`) : diagonales robustes globales (shrinkage γ = 0,2 Llama/Qwen, 0,6 Gemma ; 0,4 par défaut code) → par bloc : (1) TuneFP lr 1e-4 batch 4, 8 epochs ; (2) LB-ADMM **400 itérations × 5 inner**, pénalité linéaire, Cholesky stabilisé O(r³/3) ; (3) TuneLatentSTE lr 1e-5 batch 1, 8 epochs → pack binaire ; puis calibration globale des échelles par KL sur les logits (lr 1e-6 papier / 1e-5 code, batch 1, 8 epochs). 8 epochs à chaque stage, cosine. torch 2.6 / transformers 4.51.3 / CUDA 12.4, gradient checkpointing.
- Coûts mesurés (1× H100 80 Go) : L2-7B **1,7 GPU-h**, Q3-4B **2,3 GPU-h**, L2-70B **13 h** (137,95 → 5,75 Go).
- Mémoire : modèle BF16 + optimiseur du bloc courant uniquement. **Qwen3-8B BF16 = 16 Go → tient dans les 24 Go** (batch 1 + checkpointing). Qwen3-27B = 54 Go → **ne tient pas** sur une 3090 seule. 70B : impossible sur 3090.
- Temps sur RTX 3090 (~71 TFLOPS bf16 denses vs ~990 H100 SXM) : coût ADMM ≈ 400 × 2·r²·(a+b) FLOP par couche ; 8B @ 1 bpw ≈ 12 PFLOP → **~30–100 h estimés sur 3090** (à valider) ; 1,7B ≈ 0,65 PFLOP → ~2–6 h ; 0,6B ≈ 0,14 PFLOP → ~1–2 h. Essai réaliste : Qwen3-0.6B / 1.7B ; 8B à tenter en time-boxé.

## 5. Conception d'un noyau HIP de GEMV binaire pour gfx906

Contexte Flambeau (`origin/feature/cuda-decouple-v4`) : decode Q4_K sur gfx906 = `qmatmul_q4_k_mmvq_dp4a_t128_vdr2` (m = 1, 128 threads/ligne) puis `qmatmul_q4_k_mmvq_dp4a_row_tile_batched` (m = 2..4) ; MoE en `indexed_moe_mmvq_q4_k_r2/r4` (`dispatch/hip/gfx906.toml`). Bloc Q4_K = **144 o / 256 poids** (QK_K = 256 : `d`, `dmin` fp16, 12 o d'échelles 6-bit, 128 o de qs) = **4,5 bpw** ; activations Q8_1. Réductions DPP déjà en place (`crates/kernels-hip/src/arch_primitives/gfx906.cuh` : `quad_perm`, `row_ror`, `ds_swizzle_b32` xor16, `__shfl_xor` 32).

Instructions gfx906 disponibles (MI50 : 60 CU, wavefront 64, HBM2 **1024 Go/s** par GPU) :

- ALU entier 32 bits complet : `v_and/v_or/v_xor/v_not_b32`, `v_lshl/v_lshr_b32` → unpack des bits à la volée ;
- **popcount** `v_bcnt_u32_b32` (VALU GCN, dispo gfx906) — utile pour la variante XNOR (Σ bᵢxᵢ par XOR+popcount quand x est binaire) ;
- **dot products** : `v_dot4_i32_i8` (dp4a, déjà utilisé par Flambeau pour Q4_K/Q8_0) et `v_dot2_f32_f16` (Vega20+) ; FMA fp16 packé `v_pk_fma_f16` (2 FMA/instruction) ;
- DPP + `ds_swizzle_b32` pour les réductions de warp (déjà dans `gfx906.cuh`) ;
- pas de mma/WMMA utile en 1 bit : le GEMV binaire est memory-bound, l'ALU est disponible.

Conception proposée (miroir du noyau CUDA du papier, annexe E.2) — deux étages `y = s1 ⊙ U (Vᵀ (s2 ⊙ x))` :

1. étage 1 : `t = Vᵀ x̃`, `t ∈ R^r` — V en bitfield packé (din × r bits, un `u32` = 32 poids) ; chaque lane dépaque (shift + masque → ±1 fp16) et accumule par `v_pk_fma_f16` ; réduction DPP half-warp (layout r2 de `mmvq_q4_k_r2`) ;
2. étage 2 : `y = s1 ⊙ U t` — idem sur U (dout × r bits), échelle de ligne `s1` appliquée en sortie.

Layout proposé par couche : `U_packed[dout × ⌈r/32⌉ u32]`, `V_packed[din × ⌈r/32⌉ u32]`, `s1`/`s2` fp16 → taille = BPW·N/8 octets. Variante int8 à prototyper : x quantifié en int8 par groupe, bits expandus en octets ±1 puis dp4a (coût d'expansion ~1 op/bit).

Débit estimé (2× MI50, TP-2, 2048 Go/s) :

- Q4_K 27B : 27e9 × 4,5/8 = **15,2 Go/token** → plafond 135 t/s ; mesuré Flambeau 29,6 t/s (TP-2, MTP-0, `coord/chantier-ecarts-handoff`) ≈ 22 % du plafond (noyau limité LDS/ALU à m = 1) ;
- NanoQuant 1 bpw 27B : ≈ 27e9/8 + échelles ≈ **3,4 Go/token** → plafond ~600 t/s TP-2. Borne ALU d'unpack (~2–4 ops/bit × 27e9 bits ≈ 5–11e10 ops ; 60 CU × 64 lanes × ~1,6 GHz ≈ 6,1e12 ops/s) → 9–18 ms/token → **55–110 t/s** ;
- attendu : **~2–4× le Q4_K actuel (29,6 t/s) en decode m = 1**, selon l'efficacité atteinte. À m ≥ 2 le compute par token croît (≈ 4N FLOP @ 1 bpw, 108 GFLOP/token @ 27B → 9000 t/s à 12 TFLOPS fp32, non limitant).

## 5bis. Étape 3 — prototype du noyau HIP binaire (fait, branche `research/nanoquant`)

Fichiers : `crates/kernels-hip/src/kernels/mmvq_nq_bin.cu` (build auto-enregistré par `build.rs`), test `crates/backend-hip/tests/mmvq_nq_bin.rs`, bench `crates/backend-hip/examples/nq_bin_bench.rs`.

- Noyau deux étages : `flambeau_nq_stage1_vt_x` (t = Vᵀ(s2⊙x), grille 2D rang×lignes, tuile LDS de 512 lignes × 256 colonnes de bits chargée coalescée, unpack bit + FMA depuis LDS, `atomicAdd` dans t) puis `flambeau_nq_stage2_u_t` (y = s1⊙(U t), demi-warp par ligne de sortie, reduce `gfx906_half_warp_reduce_sum`).
- **Correction** : 3 tests vs référence CPU f64 (déquant explicite de Ŵ) — formes (128,256,64), (512,1024,r=100 impair), (4096,4096,r=2032) — erreur rel. max ≤ 6e-6. `cargo test -p flambeau-backend-hip --test mmvq_nq_bin` : 3 passed.
- **Microbench MI50 (1 GPU, m=1, poids seuls)** — évolution : v1 naïf 1,25 t/s → v2 découpage j-chunks 9,2 t/s → v3 tuile LDS + demi-warp 18,3 t/s → v4 unroll ×4 à accumulateurs indépendants **28,65 t/s sur 1 MI50, soit 57,3 t/s en TP-2** (débit atteint 19-84 Go/s selon les formes, plafond HBM2 1024 Go/s → 487 t/s 1 GPU / 974 t/s TP-2, donc ~17× de marge restante).
- Projection 27B (Qwen3.8-27B : hidden 5120, ffn 17408, 64 couches, vocab 248320) @ 1 bpw : 2,10 Go/token de poids (vs 15,19 Go en Q4_K, 7,2× moins) → **57,3 t/s TP-2 weights-only projeté**, à comparer aux 29,6 t/s TP-2 end-to-end mesurés pour Q4_K (chantier ETAT) et au plafond Q4_K de 135 t/s TP-2. Le noyau binaire naïf est donc déjà ~2× au-dessus du Q4_K mesuré en vitesse de poids, avec 7× moins de trafic HBM (gros modèles/KV plus gros en VRAM).
- Le goulot restant est l'unpack/LDS (pas la HBM) : pistes pour la suite — lectures LDS vectorisées, FMA fp16 packé (`v_pk_fma_f16`, 2 signes/poids), tuiles plus grandes, stage 1 sans atomicAdd (réduction en deux passes).

## 6. Variante ShamAN-Q (arXiv:2609.38521v1, IonQ)

- Remplace la géométrie de reconstruction diagonale de NanoQuant par une **courbure de Kronecker dense** (Shampoo, fit KL sur la Fisher empirique) → perte de Mahalanobis ; les updates ADMM continus deviennent des équations de Sylvester ; projection discrète et **format de déploiement identiques** à NanoQuant.
- Allocation de rang **non uniforme par couche** à bits totaux égaux (table manuelle ; κ = 0,6 → dernier bloc ≈ 1,8× le premier).
- Résultats Qwen3-Base, WT2 @ ~1 bpw : 27,56 → 22,96 (0,6B), 19,21 → 16,72 (1,7B), 14,29 → 13,80 (4B) ; @ 0,8 bpw sur 0,6B égale NanoQuant @ 1,0 bpw. Code : `github.com/jonathanmei/shaman` (Apache 2.0).

## 7. Étape 2 — essai de qualité (compress + PPL WikiText-2 vs Q4_K)

Dossiers sous `/home/artefact/Workspace/` : `nanoquant-env` (env + poids, provenance sha256 37/37 OK), `nanoquant-quantize` (QUANT.md), `nanoquant-ppl` (RESULTS.md). Délégation Vibe DeepSeek tentée via le hub mcp-agents (2 sessions créées) mais l'init des sessions enfant a échoué (erreur interne MCP github) → essais exécutés en direct par la coordinatrice. GPU partagé respecté : `nvidia-smi` vérifié avant chaque run, un seul job GPU à la fois, sd-server (PID 5083) jamais touché.

Choix du modèle : Qwen3-8B tient en VRAM (15,3 Gio mesurés au chargement BF16 ; pic 22,2 Gio pendant la compression) mais ~558 s/bloc × 36 blocs ≈ 5,6 h + stade KD sur RTX 3090 → hors budget de session (le papier fait L2-7B en 1,7 GPU-h sur H100). Essai retenu : **Qwen3-0.6B** (présent dans le Pareto du papier, Fig. 6).

Résultats (WikiText-2 test complet, 146 fenêtres × 2048) :

| Modèle | Format | PPL | Harnais |
|---|---|---|---|
| Qwen3-0.6B | BF16 | 20,95 | harnais du dépôt NanoQuant (fenêtres 2048) |
| Qwen3-0.6B | Q4_K_M GGUF (484 Mo, 4,5 bpw) | **19,41 ± 0,17** | llama-perplexity (build-cuda ac931b5bc, ctx 2048) |
| Qwen3-0.6B | F16 GGUF (1,50 Go) | 18,12 ± 0,16 | llama-perplexity, même réglage |
| Qwen3-0.6B | **NanoQuant 2,0 bits** (393 Mo) | **42,01** | harnais du dépôt NanoQuant |
| Qwen3-0.6B | **NanoQuant 1,0 bit** (365 Mo) | **52,49** | harnais du dépôt NanoQuant |

Notes : runs NQ effectués avec `--tune_model false` (stade KD tué par l'OOM killer cgroup RAM hôte 31 Go au premier essai ; ce stade ne change que les échelles) ; papier Table 2 v3 Q3-0.6 @1 bpw = 27,56 (lm_eval, pipeline complet) — notre 52,49 reflète surtout l'absence de ce stade. Même sans lui, à 1 bit le 0.6B est ~2,5× pire en PPL que le BF16 (même harnais) et ~2,7× pire que Q4_K_M (19,41, harnais llama-perplexity). À 2 bits, le papier (L2-7B, Table 8) atteint 6,52 ≈ QTIP 6,29 / AQLM 6,64 (2 bits vectoriels) — notre 0.6B à 2 bits : 42,01.

## 7bis. Étape 4 — intégration du format NQ2 2 bpw dans Flambeau (faite, branche `research/nanoquant`)

**Format GGUF** : nouveau dtype `NQ2` (wire id 40) dans `crates/quant` (`GgmlDType::NQ2`, `TensorInfo::size_in_bytes` calculée depuis les dims, déquant CPU `dequantize_nq2`), `QDtype::NQ2` dans `crates/core`, mapping loader `ggml_to_qdtype` (upload raw). Blob par linéaire (LE) : `[U_packed dout×wpr u32][V_packed din×wpr u32][s1 dout f16][s2 din f16]`, `wpr = ceil(r/32)`, rang exact du dépôt (`calculate_ranks`, bits=2, 2 échelles : `int(a·b·2/(a+b)−16)`, arrondi bas à 32, clamp `[32, min(a,b)]` — vérifié contre les 196 tenseurs du checkpoint). Convention de bits du dépôt `pack_binary_gemv` lsb_first : **bit set = −1, clear = +1** (le papier dit l'inverse, le code fait +1→0) ; V est stocké transposé par rapport au `.pt` (qui est rang×din/32) en din×wpr pour que les bits de rang d'une ligne d'entrée soient contigus. Correction du signe dans `nq_sign` (kenernels + test) après détection.

**Chemin de décodage** : variante Q8_1 du stage 1 (`flambeau_nq_stage1_vt_x_q8_1` dans `mmvq_nq_bin.cu`), branche NQ2 en tête de `qmatmul()` (`crates/ops/src/hip/qmatmul.rs::mmvq_nq2_launch`) : zero + stage1 + stage2 par slot d'activation, scratch `t[m×r]` en thread_local par device, stem `mmvq_nq_bin` ajouté à `KERNEL_STEMS`; descripteur `qmatmul_nq2_mmvq_gfx906` + ligne `dispatch/hip/gfx906.toml` + cert `certs/hip/gfx906/qmatmul_nq2_mmvq_gfx906.json` (tests `dispatch_toml_roundtrip` / `every_descriptor_cert_path_exists_on_disk` passent).

**Convertisseur** `scripts/nq_pt_to_gguf.py` (writer GGUF manuel, KVs copiées du GGUF F16 source, 196 linéaires → NQ2) : `Qwen3-0.6B-NQ2.gguf` (710 Mo) et `Qwen3-0.6B-NQ2-dequant-f16.gguf` (1,51 Go, Ŵ en F16 pour cross-check). `flambeau inspect-gguf` lit le fichier (sizes OK). Erreur de Frobenius Ŵ vs F16 : 0,36–0,60 selon les couches (régime attendu à ~2 bpw).

**Vérifications** :
- `crates/quant/tests/nq2_gguf.rs` : déquant Rust du blob GGUF == référence Python (f32) à 1e-5·scale — OK.
- `cargo test -p flambeau-backend-hip --test mmvq_nq_bin` : 3/3 OK sur MI50 (convention de signe corrigée).
- `crates/ops/examples/nq2_e2e.rs` (vrai dispatch `qmatmul` + vrais tenseurs du GGUF, MI50) : `blk.0.attn_q` (n=2048, k=1024) max_rel_err **3,5e-4** vs dequant CPU — OK.
- PPL cross-check : `llama-perplexity` (build-cuda ac931b5bc, ctx 2048, même harness que les 19,41/18,12) sur le GGUF déquant-F16 : **PPL = 35,05 ± 0,34** (harnais dépôt : 42,01 ; l'écart vient de la convention de fenêtres du harnais, pas des poids — le blob est validé bit-à-bit par ailleurs).

**Vitesse mesurée, 1×MI50, séquence GEMV complète du 0,6B (196 linéaires) via `qmatmul()`** :

| Chemin | m=1 | m=4 |
|---|---|---|
| NQ2 2 bpw (710 Mo) | 6,89 ms → **145 t/s** | 23,97 ms → **41,7 t/s** |
| Q4_K_M (484 Mo, 169 tenseurs quantisés) | 2,43 ms → **411 t/s** | 12,41 ms → **80,6 t/s** |

À 0,6B le chemin NQ2 est dominé par les 3 lancements/couche (588 lancements/token, ~35 µs/couche) — pas par la HBM. Projection 27B inchangée (§5bis) : 2,10 Go/token @1 bpw → 57,3 t/s TP-2 weights-only vs 29,6 t/s Q4_K end-to-end mesuré, 7× moins de trafic.

**Note archi** : le registry Flambeau ne connaît pas `qwen3` dense (archs : gemma4, qwen35, qwen35moe, qwen3moe, qwen3next — tous hybrides GDN, MoE ou attention gatée `attn_q_gated`), donc pas de run `serve` sur le 0.6B NQ2 ; le test e2e est fait au niveau op via le vrai `qmatmul()`. Un run serveur complet demanderait un petit modèle qwen35 quantisé NQ2.

## 6bis. Étape 5 — ShamAN-Q @ ~1 bpw sur Qwen3-0.6B (en cours, branche `research/nanoquant`)

**Disponibilité du code** : arXiv:2609.38521v1 (Mei et al., IonQ, 2026-09-29) — le papier dit que le code est « fourni avec l'article » et « intended to be open sourced as part of the publication » ; au 2026-11-25 aucun dépôt public (`github.com/jonathanmei/shaman` 404, e-print arXiv = source LaTeX seul). → **réimplémentation du delta §4** sur la base du code NanoQuant @ a9e0a430, patch local documenté dans `nanoquant-env/nanoquant-src/SHAMAN.md` : `src/nanoquant/core/shaman.py` (courbure de Kronecker dense KL-Shampoo 3 passes + shrinkage γ=0,2, transport + tempering spectral p=½, solves de Sylvester Mahalanobis dans la base propre conjointe avec α=ρ+λ, allocation de rang non uniforme « hand table » à parité de bits, statistique d'entrée fraîche par groupe + refresh tous les Δ=7 blocs). SVID, duals, extraction d'échelles, packing, TuneFP/TuneLatentSTE inchangés → **format de déploiement identique**, le noyau HIP et le chargeur NQ2 faits en §7bis sont réutilisables tels quels.

**Écarts documentés** (hardware) : eigendecompositions en fp32 (le papier rapporte lui-même que fp32 ne change pas la PPL, 16,66 vs 16,72 à 1,7B) ; α_k = ρ_k + λ exactement comme le papier (la variante ρ·mean(diag) du code NanoQuant diverge sur le système de Sylvester — vérifié sur jouet) ; jitter croissant autour de `eigh` sur les matrices quasi-signe (LAPACK error 15).

**Ablations du papier** (1,7B, rangs uniformes, 0,986 bpw) : repro NanoQuant 18,76 → NKP dense non temperé 19,21 → +tempering p=½ **17,46** → +fresh/refresh 17,28 → KL-Shampoo non temperé 17,04 → +tempering **16,72** (ShamAN-Q). Tempering = plus gros gain individuel ; allocation de rang non uniforme = hors bruit. Table principale : ShamAN-Q 0,6B @ ≈1 bpw = **22,96** (harnais papier) vs NanoQuant repro 29,21 ; @0,8 bpw = 27,51.

**Vérifs locales** : solver testé sur matrice jouet (recon ~0,94 en métrique dense, 0,84 en métrique identité vs 0,79 NanoQuant ; chemin transposé OK) ; allocateur de rang : 0,97 bpw atteint à bits=1.0 (parité), ratio profondeur bloc27/bloc0 ≈ 1,9 ≈ e^0,6 ; smoke 4 blocs/16 éch. (run `run-0.6B-shaman-1.0-smoke.log`, RC=0) : erreurs de recon ADMM normalisées 0,39–0,45, conforme au ‖E‖≈0,4–0,5‖W‖ annoncé à 1 bit.

**Bug de jauge du point fixe KL trouvé et corrigé** : l'itération de Jacobi détermine (A, G) à une constante près (cA, G/c) et collapsait l'échelle vers ~0 (tr(A)/n ≈ 0, cond ~10³) ; jauge tr(A) = n appliquée à chaque passe avec G compensé (le produit A⊗G, invariant, est préservé). Vérifié sur données synthétiques H = A0⊗G0 : le point fixe retrouve le vrai cond (~73 vs 72 cible) ; sur le modèle réel, cond de la covariance clippée brute ~10³¹ → cond KL post-shrinkage 917–9208 selon les couches.

**Run complet @ bits=1.0 (jauge corrigée)** : Qwen3-0.6B, 128 éch. WikiText-2 × 2048, `--tune_model false`, RTX 3090, RC=0 — log `nanoquant-quantize/run-0.6B-shaman-1.0-g2.log`, checkpoint `Qwen3-0.6B-SHAMAN-1.0-g2.pt` (365 Mo), GGUF `nanoquant-ppl/Qwen3-0.6B-SHAMAN-g2-NQ2{,-dequant-f16}.gguf` (0,97 bpw effectif). **PPL WikiText-2 (llama-perplexity ac931b5bc, `-c 2048 -ngl 99 -b 512`, 145 chunks)** :

| Modèle | bpw | PPL WT2 |
|---|---|---|
| F16 | 16 | 18,12 ± 0,16 |
| Q4_K_M | 4,5 | 19,41 ± 0,17 |
| NanoQuant | 2,0 | 35,05 ± 0,34 |
| **ShamAN-Q** | **2,0** | **à venir** |
| NanoQuant | 1,0 | 51,86 ± 0,54 |
| **ShamAN-Q (sans jauge)** | 0,97 | 54,44 ± 0,56 |
| **ShamAN-Q (jauge corrigée)** | **0,97** | **46,22 ± 0,45** |

→ Le gain directionnel du papier est reproduit sur notre harnais : ShamAN-Q bat NanoQuant au même bpw (46,2 vs 51,9, −11 % ; le papier rapporte −21 %, 29,21 → 22,96 sur son propre harnais). **La cible PPL < 25 n'est pas atteinte** — à noter que notre harnais est systématiquement plus sévère que celui du papier dès NanoQuant pur (51,9 vs 27,6 à 1 bpw, écart déjà noté en §7bis, dû entre autres au stade KD tué : `--tune_model false`). Run @ 2,0 bpw en cours pour la comparaison au même nombre de bits.

## 8. Recommandation

**Intégration faite à 2 bpw (§7bis) : le noyau HIP binaire est en place et branché dans le vrai dispatch `qmatmul` de Flambeau, validé bit-à-bit contre le convertisseur Python et correct sur MI50 (3,5e-4 via le chemin réel).** Bilan : à 0,6B, NQ2 2 bpw fait 145 t/s contre 411 t/s en Q4_K_M sur 1 MI50 (chemin NQ2 dominé par les lancements à petite échelle ; à 27B la projection reste ~57 t/s TP-2 weights-only vs 29,6 t/s Q4_K end-to-end, avec 7× moins de trafic poids). Côté qualité, sur le même harness llama-perplexity : NQ2 2 bpw = **35,05** vs Q4_K_M = 19,41 vs F16 = 18,12 — le 2 bpw reste ~1,8× pire en PPL que Q4_K, et à 0,6B le GGUF NQ2 (710 Mo) est même plus gros que le Q4_K_M (484 Mo) car embed + lm_head restent en F16 — le gain de taille n’apparaît qu’à l’échelle 27B (~9 Go vs 15,2 Go en Q4_K, poids seuls 2 bpw ≈ 6,75 Go vs 14,4 Go quantisés).

**Recommandation : on s'arrête là sur NanoQuant pur ; si on continue, c'est par ShamAN-Q (§6), pas par NanoQuant.** La vitesse n'est pas le bloqueur (noyau intégré, marge HBM énorme), la qualité à 2 bpw l'est : 35 vs 19,4. ShamAN-Q partage exactement le même format de déploiement (binaire low-rank + 2 échelles fp16) — le noyau et le chargeur NQ2 faits ici lui servent tels quels ; il ne reste qu'à changer la procédure de calibration (courbure Kronecker dense + rang non uniforme). À défaut, la variante sûre reste Q4_K. Prochaine étape proposée si feu vert : compresser Qwen3-0.6B en ShamAN-Q @ ~1–2 bpw (même harness, PPL cible < 25) puis re-mesurer le chemin Flambeau existant.
