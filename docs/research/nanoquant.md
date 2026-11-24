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
- Miroirs tiers (ex. `rkuznetsovcg/NanoQuant`) non officiels — préférer `SamsungLabs`.

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

## 6. Variante ShamAN-Q (arXiv:2609.38521v1, IonQ)

- Remplace la géométrie de reconstruction diagonale de NanoQuant par une **courbure de Kronecker dense** (Shampoo, fit KL sur la Fisher empirique) → perte de Mahalanobis ; les updates ADMM continus deviennent des équations de Sylvester ; projection discrète et **format de déploiement identiques** à NanoQuant.
- Allocation de rang **non uniforme par couche** à bits totaux égaux (table manuelle ; κ = 0,6 → dernier bloc ≈ 1,8× le premier).
- Résultats Qwen3-Base, WT2 @ ~1 bpw : 27,56 → 22,96 (0,6B), 19,21 → 16,72 (1,7B), 14,29 → 13,80 (4B) ; @ 0,8 bpw sur 0,6B égale NanoQuant @ 1,0 bpw. Code : `github.com/jonathanmei/shaman` (Apache 2.0).

## 7. Étape 2 — essai de qualité (compress + PPL WikiText-2 vs Q4_K)

En cours — résultats à ajouter ici (dossiers délégués : `nanoquant-env`, `nanoquant-quantize`, `nanoquant-ppl` sous `/home/artefact/Workspace/`).

## 8. Recommandation

À compléter après l'étape 2.
