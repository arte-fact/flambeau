# qwen35-v2 `real_gguf_cuda_parity` (Qwen3.6-27B-UD-IQ2_XXS) : échec antérieur à V1

**Constat sur V1 (838f3db3).** Le test échoue sur l'assert « séquences gloutonnes HIP == CUDA » (real_gguf_cuda_parity.rs:135). CUDA et HIP partagent les 3 premiers tokens, puis divergent au 4e (CUDA 8340, HIP 1817). Les autres asserts passent :
- CUDA et llama.cpp ont 2 tokens communs, le minimum exigé ;
- l'argmax du préremplissage est identique ;
- la cosine des logits de préremplissage est ≥ 0,95.

**Le test n'emprunte aucun chemin modifié par V1.**
- `token_embd` est en Q3_K : la table reste en F16 et n'utilise pas le gather Q4_K de (a).
- Le GGUF n'a pas de `blk.64`, donc pas de tête MTP : mtp0-noload n'a aucun effet.
- Le test décode sans spéculation : l'index GDN reste à 0 et la matérialisation ne fait rien.

**Preuve.** Le même test construit à f5d29f30 (serie-7) échoue avec des séquences identiques.

**Bisect.** Restreint aux chemins kernels-*, ops, backend-*, dispatch et model-ops, de d84426c8 (passe, 10/10) à f5d29f30, soit 7 builds testés. Il a fallu ajouter dans le script le 7e argument de `scratch_config`, que le test ne passait pas sur une partie de la plage.

| commit | HIP == CUDA (sur 10) | cosine du préremplissage HIP/CUDA | HIP généré |
|---|---|---|---|
| d84426c8 | 10 | 0,998417 | 248068, 198, 8160, 579, … |
| c9b1176e (dernier bon) | 10 | 0,998417 | idem |
| **dee02058 (premier mauvais)** | 2 | 0,997644 | 248068, 198, 90700, 1817, … |
| 9512db81 | 2 | 0,997651 | idem |
| f5d29f30 / V1 | 3 | — | 248068, 198, 90700, 1817, … (CUDA : …, 90700, 8340, …) |

- Sur toute la plage c9b1176e..9512db81, la sortie CUDA ne change pas. C'est HIP qui bouge.
- **Premier commit fautif : dee02058**, « perf(hip): small-batch LDS tile (m=3..32) + Q8_0 LDS-tiled from m=32 ».
- **Mécanisme, déduit du message de commit et non vérifié noyau par noyau.** Le préremplissage de 15 tokens du test fait m = 15 sur gfx906. Avant, il passait par la boucle MMVQ ligne par ligne. Depuis dee02058, il passe par la MMQ LDS-tiled x8, qui quantifie les activations et accumule dans un autre ordre.
  - Les logits de préremplissage HIP s'écartent légèrement de CUDA : cosine 0,998417 → 0,997644.
  - Avec des logits 2 bits très plats, le glouton bascule sur une quasi-égalité. Le message du test l'anticipe déjà pour llama.cpp.
  - Les impls x8 sont certifiées dans leur tolérance : ce n'est pas un défaut de justesse démontré.
- **À trancher, hors V1.** Soit l'assert d'égalité exacte des séquences HIP/CUDA devient « préfixe commun ≥ N », comme pour llama.cpp. Soit le préremplissage HIP à 3 ≤ m ≤ 32 doit reproduire CUDA au bit près. Je n'ai rien corrigé.
