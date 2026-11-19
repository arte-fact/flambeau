# Écarts flambeau ↔ llama.cpp — Qwen3.8-27B-UD-Q4_K_M, 2× MI50, KV q8_0

Base : flambeau `e073c460` (feature/cuda-decouple), binaire release du 19/11 10:13 ;
llama.cpp `ffc87c3a7` (`~/llama.cpp/build-hip`). Mêmes GPU (2× gfx906), même GGUF,
glouton, 5 invites `lp.py`, 400 tokens max.

Configuration de référence :
- flambeau : `--mesh-mode tp --tp-size 2 --kv q8 --prefill-ubatch 512` ;
- llama.cpp : `-sm layer -ts 1/1 -ctk q8_0 -ctv q8_0 -fa on`, avec draft-mtp
  `--spec-draft-n-max 4`.

| | flambeau | llama.cpp | écart |
|---|---|---|---|
| Sans MTP (mesure Vibe) | 20,6 t/s | 25,9 t/s | −20 % |
| MTP-4 (mesure Vibe) | 26,5 t/s | 33,1 t/s | −20 % |
| Acceptation | 49,3 % | 58,8 % | −9,5 pts |

Scripts, traces et données brutes : `~/Workspace/flambeau-exp/analyse-ecarts/`.
- Traces rocprofv3 : `traces/`.
- Analyse : `an_kern.py` (temps par kernel et par token), `an_grid.py` (par forme de
  grille), `an_type.py` (temps matmul par type de quantification).
- Lancement des serveurs : `flserve.sh` / `lcserve.sh`.

---

## 1. Priorité 1 — bug de répartition des têtes GDN en TP-2

En TP-2, flambeau calcule un modèle **faux** pour Qwen3.5/3.8-27B : un tiers des têtes V de chaque couche GDN lisent la mauvaise tête Q/K. Ce bug dégrade la qualité du modèle en TP et fausse l'état caché fourni à la tête MTP, d'où la plus grande part de l'écart d'acceptation (49,3 % contre 58,8 %). Il passe avant tout travail de performance : toute mesure TP-2 sur ce modèle porte sur un calcul incorrect.

### 1.1 Preuves

Preuve 1 — lecture du code :
- `gdn_tp_mode_for` (`forward/src/loader/gdn_layer.rs:46-56`) choisit `FullShard`
  parce que 24 % 16 ≠ 0.
- Sa propre documentation (`:24-25`) indique « FullShard pairs with
  `rep_inner_layout = true` (qwen3-Next) ». Or qwen35 charge `rep_inner_layout: false`
  (`qwen35-v2/src/loader.rs:144`).
- Avec `false`, le noyau calcule `h_kv = h_idx % H_kv`
  (`kernels-shared/include/gdn_state_step_alphabeta.cuh:34,125`), comme llama.cpp
  (`gated_delta_net.cu:52`, `iq1 = h_idx % neqk1`).
- `pack_gdn_qkv_slab` (`forward/src/loader/gdn_shard.rs:76-89`) découpe ainsi :
  rang r ← têtes K `[8r, 8r+8)` et têtes V `[24r, 24r+24)`.
- Au rang 0, la tête V globale j ∈ 8..15 utilise la tête K j−8 au lieu de j. Au rang 1,
  les têtes j ∈ 32..39 utilisent la tête K 8+(j−32) au lieu de j−32.
- Bilan : **16 têtes V sur 48 (un tiers) sont fausses, dans les 48 couches GDN.**

La preuve 2 isole l'effet du TP : le token t0 sort du préremplissage, il ne dépend donc
ni du MTP ni du décalage de position (§4.8). Le mode PP de flambeau, qui n'a pas de
découpage GDN, colle à llama.cpp ; le mode TP s'en écarte d'un ordre de grandeur de plus
que le bruit de quantification du KV.

Preuve 2 — log-probabilités du premier token t0. Écart maximal sur le top-5 commun, en
nats (`cmp_lp.py`) :

| Invite | flambeau TP q8 / llama.cpp q8 | flambeau TP f16 / llama.cpp f16 | llama.cpp q8 / f16 | **flambeau PP f16 / llama.cpp f16** |
|---|---|---|---|---|
| code | 0,79 | 0,38 | 0,20 | 0,56 |
| math | 1,51 | 1,42 | 0,10 | **0,19** |
| chat | 1,56 | 1,53 | 0,06 | **0,13** |
| story | 0,51 | 0,33 | 0,08 | 0,36 |
| list | 1,22 | 2,68 | 0,10 | 0,53 |

Exemple « math », t0 : log p(« To ») vaut −0,85 chez flambeau TP contre −2,16 chez
llama.cpp.

Preuve 3 — longueur du préfixe commun avec le texte glouton de llama.cpp, en
caractères :

| Invite | code | math | chat | story | list |
|---|---|---|---|---|---|
| flambeau TP | 68 | 67 | 48 | 4 | 9 |
| flambeau PP | 522 | 67 | 307 | 215 | 200 |

Preuve 4 — pourquoi ce bug fait chuter l'acceptation MTP. Le reste du pipeline MTP est
identique dans les deux moteurs, qui utilisent :
- l'état caché pris **après** `output_norm` (llama.cpp `qwen35.cpp:213-217` ;
  flambeau `mtp.rs:73-83`) ;
- la concaténation `[enorm(emb), hnorm(h)]` puis `eh_proj` ;
- un bloc d'attention complète ;
- `shared_head_norm` puis la lm_head partagée ;
- des brouillons 2 à 4 enchaînés sur l'état caché de sortie MTP (après norme) ;
- la paire (emb(x_p), h_{p−1}) ;
- une acceptation gloutonne par correspondance exacte, avec token bonus ;
- un retour arrière de l'état GDN exact (instantané plus rejeu chez flambeau,
  instantané par token chez llama.cpp) ;
- un KV MTP en q8_0.

Le GGUF ne contient ni `nextn.embed_tokens` ni `nextn.shared_head_head`, donc aucune
divergence de ce côté.

Ce qui diffère, c'est **l'état caché lui-même**, produit par les couches GDN faussées.
La tête MTP, entraînée sur les états cachés du vrai modèle, propose donc moins souvent
le token que le modèle cible (faussé lui aussi) retient.

Gain estimé :
- l'acceptation devrait remonter vers celle de llama.cpp (58,8 %), soit 3,3 à 3,37
  tokens par cycle au lieu de 2,97, **+11 à 13 % en MTP : 26,5 → ≈ 29,5 t/s (+3 t/s)** ;
- la qualité du modèle en TP est corrigée.

Cette estimation n'a pas pu être mesurée directement : flambeau refuse le MTP en mode
PP (`token_embd has 0 F16 elems`).

Difficulté : faible à moyenne.

### 1.2 Pourquoi les tests ne l'ont pas vu

- `qwen35-v2/tests/session_tp2_27b.rs` vérifie seulement que les logits sont finis et non constants.
- `qwen35-v2/tests/session_tp_parity.rs` compare TP et mono-GPU sur le 9B, dont la géométrie (têtes V par rang multiples de 16) passe par `KReplicated`, un mode correct. Le chemin `FullShard` + `rep_inner_layout = false` n'a jamais été validé.

### 1.3 Correctif recommandé (pour flambeau@threadripper)

Fichiers :
- `crates/forward/src/loader/gdn_layer.rs` (`gdn_tp_mode_for`, `load_gdn_layer`) ;
- `crates/forward/src/loader/gdn_shard.rs` (`pack_gdn_qkv_slab`,
  `upload_gdn_fused_qkv_quant`, `upload_f32_array_sharded`) ;
- les appels `upload_col_sharded_quant` / `upload_row_sharded_quant` pour `attn_gate`,
  `ssm_alpha`, `ssm_beta`, `ssm_out`, `ssm_dt.bias`, `ssm_a`, `ssm_conv1d`.

Approche recommandée : « FullShard permuté », sans surcoût de lecture.
- Quand `rep_inner_layout == false`, le rang r prend les têtes K `[r·nk/R, (r+1)·nk/R)`
  et les têtes V `{ j : (j % nk) ∈ [r·nk/R, (r+1)·nk/R) }`, dans l'ordre croissant de j.
- Pour le 27B :
  - rang 0 : têtes V 0..7, 16..23, 32..39 ;
  - rang 1 : têtes V 8..15, 24..31, 40..47.
- Le noyau garde `h_local % H_kv_local`. Avec cet ordre, la tête V locale h retrouve sa
  bonne tête K locale.
- Appliquer **la même permutation des têtes V** partout :
  - aux lignes V du bloc QKV fusionné ;
  - aux canaux V de `ssm_conv1d` ;
  - aux lignes de `attn_gate` (z) ;
  - à `ssm_alpha`, `ssm_beta`, `ssm_dt.bias` et `ssm_a` ;
  - aux **colonnes** de `ssm_out` (sa dimension d'entrée `d_inner`, découpée par têtes V).

  `ssm_norm` est partagé entre têtes : rien à faire.
- Alternative moins bonne : `KReplicated` avec un décalage de tête V passé au noyau,
  `h_kv = (h_local + v_off) % H_kv`. Elle réplique Q et K et coûte environ 346 Mo lus
  en plus par token et par GPU.

Validation :
1. Nouveau test `qwen35-v2/tests/session_tp2_27b_parity.rs`. Il compare, sur le
   Q4_K_M, la `Session` TP-2 à la `Session` PP-2 sur les 5 invites de `lp.py`.
   Critères : écart maximal de log-probabilité ≤ 0,3 nat sur le top-5 de t0, et préfixe
   glouton commun ≥ 32 tokens. Il doit **échouer avant** le correctif (aujourd'hui :
   jusqu'à 2,7 nats à t0, divergence dès le 1er-2e token sur story et list).
2. `cmp_lp.py` sur les 5 invites : l'écart t0 « flambeau TP f16 / llama.cpp f16 » doit
   descendre au niveau de « flambeau PP » (≤ 0,6 nat, ≤ 0,2 sur math et chat).
3. Banc `bench_step4d.sh` (MTP-4 avec `mtp.cycle=debug`), GPU à moins de 60 °C au
   départ.

Mesure attendue : acceptation ≥ 55 %, plus de 3,25 tokens par cycle, MTP ≥ 29 t/s à
chaud.

Contrôle d'architecture (CLAUDE.md) : le correctif reste dans le chargeur générique (`crates/forward/src/loader`) et se déclenche sur `rep_inner_layout` et la géométrie, pas sur le nom de l'architecture (règles 12 et 13). Aucun flag d'environnement (règle 1). Aucun noyau modifié, donc pas de nouveau certificat de noyau ; le test de parité TP/PP du 27B tient lieu de preuve de correction.

---

## 2. Résumé

Quatre constats expliquent l'essentiel des deux écarts.

1. **Bogue de correction en TP-2 (GDN).**
   - En TP-2, le 27B (48 têtes V, 16 têtes K) bascule en `GdnTpMode::FullShard`. Ce
     mode découpe les têtes V et K en blocs contigus.
   - Or qwen35 associe la tête V `j` à la tête K `j % 16` (`rep_inner_layout = false`).
   - Résultat : sur chaque rang, 8 des 24 têtes V lisent la mauvaise tête Q/K, dans les
     48 couches GDN.
   - Les logits TP s'écartent de llama.cpp de 1,2 à 2,7 nats dès le préremplissage.
     Le mode PP de flambeau, lui, colle à llama.cpp (0,13 à 0,56 nat).
   - La tête MTP reçoit donc des états cachés « faux » : c'est la cause principale de
     l'écart d'acceptation.
2. **Matmul quantifiés deux fois moins efficaces par octet.**
   - Les MMVQ de flambeau lisent les poids à environ 300 GB/s par GPU (Q4_K : 194 GB/s).
     llama.cpp atteint environ 490 GB/s sur le même silicium et les mêmes poids.
   - Le TP-2 divise par deux les octets lus par GPU, mais ce gain est presque entièrement
     perdu : 29,1 ms de matmul par GPU et par token pour la moitié des poids, contre
     31,7 ms chez llama.cpp pour la totalité.
3. **Cycle MTP chargé en tête de sortie (lm_head) et en allers-retours vers l'hôte.**
   - La lm_head est répliquée sur les deux GPU (vocabulaire complet, 1,04 Go en Q6_K).
   - Elle est appelée environ 7,7 fois par cycle : 4 brouillons, plus 1 à 5 lignes de
     vérification traitées une par une.
   - Chaque appel déclenche une copie des logits vers l'hôte et un échantillonnage sur
     l'hôte. Total : environ 13 ms de lm_head et environ 8 ms de bulles par cycle de
     91 ms.
4. **La thermique fausse les mesures.**
   - Les MI50 atteignent 97 à 102 °C de jonction et la fréquence oscille entre 925 et
     1725 MHz.
   - Sur 10 requêtes enchaînées, flambeau passe de 22,6 à environ 17,5 t/s (−22 %),
     llama.cpp de 26,3 à 23,3 t/s (−11 %).
   - Les 20,6 / 25,9 de référence mélangent donc des états thermiques différents. À
     froid, l'écart de base est de 22,6 contre 26,3 t/s.

S'y ajoutent :
- un décalage d'une position au premier token décodé : la ligne KV `L` n'est jamais
  écrite, d'où une fuite d'état d'une requête à l'autre (prouvée) ;
- une attention de décodage sous-occupée (12 workgroups, 230 µs par appel) ;
- un rmsnorm+quantification fusionné mono-workgroup (27 µs × 65 appels) ;
- le MTP désactivé sur les 24 premiers tokens et sur tout le chemin streamé (SSE).

---

## 3. Méthode et précautions

- **Profilage.**
  - flambeau : `rocprofv3 --kernel-trace` sur `flambeau serve`, une requête « math »
    de 130 tokens, sans MTP, GPU à 57-61 °C au départ. Une trace `--runtime-trace` de
    400 tokens sert pour les appels d'API HIP.
  - llama.cpp : `llama-server` et `llama-bench` plantent sous rocprofv3 dans
    `ggml_cuda_graph_evaluate_and_capture`. La trace est donc prise avec
    `GGML_CUDA_DISABLE_GRAPHS=1 llama-bench -p 0 -n 128` (décodage seul), GPU à 51 °C.
  - Les graphes HIP valent +4,6 % à llama.cpp (26,64 t/s avec, 25,46 sans).
- **Effet du profileur.** Il gonfle le temps hôte et les trous entre kernels :
  - flambeau : 18,3 t/s tracé contre 22,6 non tracé ;
  - llama.cpp : 23,1 contre 25,5.

  Les **durées de kernels** sont fiables ; les temps d'inactivité tracés ne le sont
  pas. L'inactivité réelle est estimée par la formule : temps réel par token − somme
  des durées de kernels.
- **Thermique.** Tous les chiffres « à froid » sont pris avec une jonction ≤ 61 °C au
  départ. Voir §4.6.
- **Tokens par fenêtre.** Comptés par les kernels à fréquence connue : 48 pas GDN et
  16 attentions par token.

---

## 4. Débit de base (sans MTP)

### 4.1 Bilan par token (à froid)

| | flambeau TP-2 (par GPU, en parallèle) | llama.cpp `-sm layer` (2 GPU en série) |
|---|---|---|
| Temps réel par token, non tracé | **44,2 ms** (22,6 t/s) | **37,5 ms** (26,6 t/s, `llama-bench tg128`) |
| Temps GPU (somme des kernels) | 40,4 ms par GPU | 20,0 + 20,9 = 40,9 ms (série) |
| Inactivité estimée | ≈ 4 ms | ≈ 0 (graphes HIP) |
| Lancements de kernels par token | 1 824 **par GPU** | ≈ 960 par GPU (≈ 30 par couche) |
| Matmul quantifiés (MMVQ) | **29,1 ms** pour la moitié des poids | **31,7 ms** pour la totalité |
| Attention de décodage | **3,6 ms** (16 × 225 µs) | 0,66 ms (16 × 41 µs) |
| RMSNorm (+ quantification Q8_1) | 1,8 ms (65 × 27,4 µs, 1 workgroup) | 1,8 ms (rms_norm 6 µs + quantize 4,5 µs) |
| GDN (pas d'état, conv1d, normes, etc.) | ≈ 1,2 ms | ≈ 1,1 ms |
| Communications TP (copies, all-reduce, purge L2) | 2,5 ms | ≈ 0 (une copie d'état caché) |
| Copies GPU→hôte | 2 par token (1 Mo de logits, 0,14 ms, sur **chaque** rang) | 1 |
| Petits kernels élémentaires | ≈ 3,4 ms | ≈ 2,5 ms |

Lecture : flambeau tient chaque GPU occupé environ 40 ms par token pour lire 8,2 Go de
poids (moitié des 15,4 Go, plus la lm_head répliquée). llama.cpp met environ 40 ms
GPU, en série, pour en lire 16,4. **Le parallélisme TP n'apporte rien parce que chaque
octet coûte deux fois plus cher.**

### 4.2 Cause B1 — efficacité des MMVQ (le levier dominant)

Preuve : temps MMVQ par type, sur l'ensemble des deux GPU, comparé aux octets du GGUF
(fichiers `traces/*`, script `an_type.py`).

| Type | Octets du modèle | llama.cpp ms/token | GB/s | flambeau ms/token (2 GPU cumulés) | GB/s |
|---|---|---|---|---|---|
| Q5_K | 4,83 Go | 9,66 | **501** | 15,88 | **304** |
| IQ4_XS | 4,76 Go | 9,64 | **494** | 15,38 | **310** |
| Q4_K | 3,49 Go | 7,21 | **484** | 17,98 | **194** |
| Q6_K (hors lm_head) | 0,43 Go | 0,83 | 519 | 1,11 | 388 |
| Q6_K lm_head | 1,04 Go | 1,53 (une fois) | 680 | 3,54 (répliquée) | 590 |
| Q3_K | 0,27 Go | 0,76 | 354 | 1,64 | 163 |
| IQ3_S | 0,15 Go | 0,71 | 214 | 1,03 | 149 |

Exemples de formes, par appel, GPU 1 (`an_grid.py`) :
- `flambeau_mmvq_q4_k_r2_dp4a_q8_1`, 8 704 lignes × 5 120 (moitié de ffn_gate/up) :
  120 µs, soit 209 GB/s.
- `flambeau_mmvq_q5_k_dp4a_q8_1`, 5 120 lignes (moitié de attn_qkv) : 56 µs, soit
  321 GB/s.
- `flambeau_mmvq_iq4_xs_dp4a_q8_1`, 8 704 lignes : 77 µs, soit 309 GB/s.

Les noyaux 256 threads (Q5_K, IQ4_XS) traitent **une ligne par workgroup de 4 vagues**
sur K = 5 120, soit 20 super-blocs pour 256 threads. Le noyau Q4_K est la variante
r2 à 64 threads (`dispatch/hip/gfx906.toml:63-68`, impl `qmatmul_q4_k_mmvq_dp4a`), le
pire de tous. Rappel CLAUDE.md : « Q4_K/Q5_K MMVQ wc (64-thread) loses to cuda1
(256-thread) ».

Gain estimé :
- amener tous les types à environ 470 GB/s ramène les matmul de 29,1 à environ 18 ms
  par GPU : **−11 ms par token**, soit 44,2 → 33 ms, **≈ 30 t/s (+7 t/s)** ;
- le seul Q4_K à 310 GB/s (niveau des autres noyaux flambeau) : −3,4 ms par token,
  **+1,9 t/s**.

Base de mesure : MMVQ de llama.cpp sur ce même silicium.

Difficulté :
- Q4_K : faible à moyenne (une ligne de dispatch, plus éventuellement une variante
  256 threads) ;
- tous types : élevée, sur plusieurs sessions.

### 4.3 Cause B2 — attention de décodage sous-occupée

Preuve :
- `flambeau_attention_decode_q8_kv` : grille de 768 threads, soit **12 workgroups**
  (12 têtes Q par rang) pour 60 CU. Il prend 225 à 283 µs par appel, 16 appels par
  token, soit **3,6 à 3,8 ms par GPU**.
- La variante split-K prend 54 + 6 µs (fenêtre de 400 tokens), mais elle n'est
  utilisée qu'au-delà de 256 positions de KV : `standard_attn.rs:1422`, `:770`,
  `:936`, condition `> 256`.
- llama.cpp `flash_attn_ext_vec` : 41 µs.

Gain estimé : −2,6 ms par token, **+1,4 t/s** en base.

Difficulté : faible.

### 4.4 Cause B3 — rmsnorm+quantification fusionné, mono-workgroup

Preuve :
- `flambeau_rmsnorm_q8_1_fused` : grille de 256 threads, 27,4 µs par appel, 65 appels
  par token, soit 1,8 ms.
- Le lancement se fait avec `LaunchCfg::one_d(shape.m, 256)` (`ops/src/launch/norm.rs`,
  `rmsnorm_quant_q8_1`), donc un seul workgroup quand m = 1.

Gain estimé : −1,3 ms, **+0,7 t/s**.

Difficulté : faible.

### 4.5 Cause B4 — lm_head répliquée, logits copiés vers l'hôte sur chaque rang

Preuve :
- La lm_head en Q6_K, grille de 15 892 480 threads (vocabulaire complet de 248 320),
  prend 1,76 ms par token **sur chaque GPU**. `load_lm_head` (`forward/src/loader/globals.rs`)
  ne la découpe pas.
- Chaque rang copie ensuite 1 Mo de logits vers l'hôte (2 copies par token), puis
  échantillonne sur l'hôte.

Gain estimé :
- découpe du vocabulaire en deux : −0,9 ms ;
- argmax sur GPU en glouton : −0,3 à 0,5 ms de bulle ;
- total **+0,6 t/s** en base. L'effet est bien plus fort en MTP (§5.3).

Difficulté : moyenne.

### 4.6 Cause B5 — thermique et méthode de mesure

Preuve :
- **flambeau sans MTP**, ordre inversé
  (`list,story,chat,math,code,code,code,list,list`) : 22,6 → 22,6 → 21,6 → 19,7 →
  17,5 → 17,4 → 17,2 → 17,1 → 17,0 t/s.
  - L'invite « list » donne 22,6 t/s en première position et 17,0 en dernière.
  - Le ralentissement dépend donc de l'âge du serveur, pas de l'invite.
  - Il plafonne autour de 17,5 à 18,5 t/s, avec une jonction de 97 à 100 °C et une
    fréquence sclk entre 1282 et 1725 MHz.
- **llama.cpp**, 10 requêtes enchaînées : 26,3 → 26,2 → 26,2 → 25,8 → 25,3 → 24,7 →
  24,1 → 23,6 → 23,3 → 23,3 t/s, jonction 99 à 102 °C (`smi_lc_hot.txt`).
- Pourquoi flambeau perd plus :
  - en TP, les deux GPU travaillent en permanence ; en découpe par couches, chacun ne
    travaille que la moitié du temps ;
  - ses MMVQ, loin de la bande passante, dépendent davantage de la fréquence sclk.
  Corriger B1 réduit donc aussi la sensibilité thermique.

Gain : aucun sur le code, mais indispensable pour juger les autres leviers.

Difficulté : nulle.

### 4.7 Cause B6 — communications TP, copies, synchronisations, lancements

Preuve, trace `--runtime-trace` (appels d'API HIP sur les deux threads hôtes, par
token) :
- 3 277 `hipModuleLaunchKernel` ;
- 256 `hipMemcpyPeerAsync` (2 par all-reduce, 128 all-reduce par token) ;
- 132 `hipMemcpyAsync` ;
- 512 `hipEventRecord` et 512 `hipStreamWaitEvent` ;
- 2 `hipStreamSynchronize`.

Côté GPU, par GPU et par token :
- 193 kernels `__amd_rocclr_copyBuffer` (5 µs chacun, ≈ 1,0 ms) ;
- `p2p_allreduce_residual_rmsnorm_tp2` : 64 × 14,6 µs ;
- `p2p_allreduce_residual_tp2` : 64 × 2,8 µs ;
- `p2p_l2_flush` : 128 × 3 µs.

Communications : environ 2,5 ms par GPU, plus environ 4 ms d'inactivité, dues aux
attentes croisées et aux trous entre 1 824 lancements.

Gain estimé : −2 ms, **+1 t/s**.

Difficulté : moyenne à élevée. La capture de graphes HIP est nulle sur gfx906 d'après
CLAUDE.md, donc **ne pas la proposer**. Il faut réduire le nombre de kernels et de
copies.

### 4.8 Cause B7 (correction) — décalage d'une position au premier token décodé

Preuve, côté code :
- Le préremplissage écrit les positions `0..L-1` (`forward/src/runtime/mod.rs:382-400`).
- La boucle de décodage commence à `step = 1` avec
  `pos = prompt_ids.len() + step + mtp_offset` (`server/src/routes/decode_loop.rs:1204-1211`).
  Le premier token généré est donc envoyé en **L+1**.
- Même formule aux lignes `:621`, `:826` et `:1727`.
- La ligne KV `L` de chaque couche d'attention complète, et celle de la couche MTP, ne
  sont jamais écrites. L'attention lit pourtant `pos+1` lignes, donc une ligne
  périmée, héritée de la requête précédente.
- Effet supplémentaire en MTP : l'état caché en attente vaut `L-1`, donc à `L+1` la
  ligne MTP est calculée avec un état caché **nul** (`MtpLead::Zero`,
  `qwen35-v2/src/model.rs:51-62`).

Preuve expérimentale :
- La sortie gloutonne de « list » **change selon la requête précédente** : identique
  après « list » ou « math », divergente au caractère 356 après « code ».
- Sur un serveur neuf, « list » produit 243 morceaux SSE ; plus tard, 233.

Gain : 0 t/s. C'est de la correction, avec un effet de qualité et d'acceptation non
chiffrable sans correctif.

Difficulté : faible.

---

## 5. Acceptation MTP et débit MTP

### 5.1 Décomposition de l'écart MTP-4 (26,5 contre 33,1)

À partir de `fl_b4d.log` (une ligne `mtp.cycle` par rang, dédoublonnée) et de
`lc_mtp2.log` :

| Invite | flambeau acc. | llama.cpp acc. | flambeau tokens/cycle | llama.cpp tokens/cycle |
|---|---|---|---|---|
| code | 89,7 % | 92,4 % | 4,59 | 4,70 |
| math | 66,3 % | 75,5 % | 3,65 | 4,02 |
| chat | 38,1 % | 53,2 % | 2,52 | 3,13 |
| story | 35,1 % | 40,8 % | 2,41 | 2,63 |
| list | 42,9 % | 43,9 % | 2,72 | 2,76 |
| **total** | **49,3 %** | **58,8 %** | **2,97** | **3,37** |

Distribution des acceptations chez flambeau (0 à 4 sur 4) : 178 / 220 / 144 / 114 / 218
cycles. Probabilité conditionnelle de passer au rang suivant : 0,80, puis 0,68, 0,70
et 0,66.

Coût d'un cycle (4 brouillons + vérification de 5 tokens) :
- flambeau : **91 à 103 ms** à froid (`fl_b4d`), soit 2,05 tokens simples ; 103 à
  132 ms à chaud (`fl_b4`) ;
- llama.cpp : 97 à 113 ms (`eval_time / (accepted / (mean_len − 1))`), soit 2,6 à
  2,9 tokens simples.

**Le cycle flambeau n'est pas plus cher que celui de llama.cpp. L'écart MTP se
décompose ainsi :**
1. moins de tokens par cycle : 2,97 contre 3,37, soit −12 % ;
2. une base plus lente, dont la vérification paie toute l'inefficacité des MMVQ
   (≈ 43 ms de matmul par cycle et par GPU) ;
3. 19 à 23 tokens par requête en décodage simple, à cause de la fenêtre d'arrêt (§5.5) ;
4. l'état thermique (b4 tourne juste après b0, à chaud).

### 5.2 Cause M1 — bug de répartition des têtes GDN en TP-2

Voir §1 (preuve complète et correctif). C'est la cause principale de l'écart d'acceptation.

### 5.3 Cause M2 — lm_head ligne par ligne et allers-retours vers l'hôte dans le cycle

Preuve, trace MTP-4 (`traces/fl_tp2_mtp4_*`, 75 cycles, 300 tokens) :
- **603 appels de lm_head par GPU**, soit 23 tokens simples plus 75 cycles × 7,72 (4
  brouillons + en moyenne 3,72 lignes de vérification). Chaque appel prend 1,67 ms, soit
  **12,9 ms par cycle et par GPU**.
- Autant de copies de 1 Mo vers l'hôte (602 par rang).
- 812 trous de plus de 200 µs sur le GPU 1, soit 601 ms au total, c'est-à-dire
  **≈ 8 ms par cycle** d'attente de l'échantillonnage hôte.
- La vérification appelle `head_row` ligne par ligne (`qwen35-v2/src/model.rs:131-150`,
  appels `:241` et `:253`). Chaque ligne relit 1,04 Go de poids.

llama.cpp, lui, échantillonne les brouillons sur le GPU (`backend_sampling=true`,
`common/speculative.cpp:1873`). Il calcule les logits de vérification des 5 positions
en un seul passage par lot, sans relire la lm_head ligne par ligne.

Gain estimé :
- une lm_head par lot de 5 lignes en vérification, plus un argmax sur GPU pour la
  vérification et les brouillons : environ −10 ms par cycle, 91 → 81 ms,
  **+12 %, ≈ +3 t/s en MTP** ;
- avec en plus la découpe du vocabulaire (B4) : −4 ms par cycle supplémentaires,
  **+4 %**.

Difficulté : moyenne.

### 5.4 Cause M3 — décalage de position : état caché nul et ligne périmée dans le KV MTP

Voir §4.8. Les lignes `L` (périmée) et `L+1` (état caché nul) du KV MTP sont lues par
chaque brouillon de la requête.

Gain : non chiffrable sans correctif. C'est probablement un effet secondaire
comparé à M1.

Difficulté : faible, car c'est le même correctif que B7.

### 5.5 Cause M4 — MTP coupé sur les 24 premiers tokens, et sur tout le chemin streamé

Preuve :
- `in_stop_window` (`decode_loop.rs:1214-1215`) force le décodage simple tant que
  `generated.len() < min_response_tokens`. Les traces montrent 19 à 23 tokens simples
  par requête (`plain_steps`), soit environ 7,7 % des tokens, à 44 ms au lieu de
  ≈ 30 ms en MTP.
- `mtp_step` n'est appelé que dans `run_completion_blocking_ids` (`decode_loop.rs:1198-1260`).
  Les chemins streamés (`run_completion_streaming_scheduler_pp_blocking` `:716`,
  `run_completion_blocking_streaming` `:1471`) n'ont pas de MTP.
- Mesure : une requête `stream:true` avec `--mtp 4` donne 18,2 t/s et **zéro** cycle
  MTP.

Gain estimé :
- fenêtre d'arrêt : **+3 % en MTP (+0,8 t/s)** sur le banc ;
- streaming : nul sur le banc `lp.py`, qui est non streamé, mais tout client SSE en
  profiterait.

Difficulté : faible à moyenne.

---

## 6. Recommandations classées par rapport gain / effort

Base de calcul :
- sans MTP : 22,6 t/s à froid (44,2 ms par token) ;
- avec MTP : 26,5 t/s, mesure Vibe, à chaud.

Gains en t/s, à vérifier une fois corrigés.

| # | Levier | Gain estimé | Effort | Ratio |
|---|---|---|---|---|
| R1 | Corriger le découpage GDN en TP pour qwen35 (M1) | MTP +3 t/s, acceptation ≈ 57 %, qualité TP | faible-moyen | ★★★★★ |
| R2 | Protocole thermique du banc (B5) | fiabilité des mesures | nul | ★★★★★ |
| R3 | Corriger le décalage de position (B7 / M3) | correction, plus un peu d'acceptation | faible | ★★★★ |
| R4 | Split-K de l'attention de décodage pour tout contexte (B2) | base +1,4 t/s | faible | ★★★★ |
| R5 | MMVQ Q4_K : variante 256 threads ou portage llama.cpp (B1, partiel) | base +1,9 t/s, MTP ≈ +1 | faible-moyen | ★★★★ |
| R6 | Vérification MTP : lm_head par lot et argmax sur GPU (M2) | MTP +3 t/s | moyen | ★★★★ |
| R7 | rmsnorm_q8_1_fused multi-workgroup (B3) | base +0,7 t/s | faible | ★★★ |
| R8 | MTP dans la fenêtre d'arrêt, et sur le chemin streamé (M4) | MTP +0,8 t/s (plus SSE) | faible-moyen | ★★★ |
| R9 | lm_head découpée par vocabulaire, argmax combiné (B4) | base +0,6, MTP +1 t/s | moyen | ★★ |
| R10 | MMVQ Q5_K / IQ4_XS / Q6_K / Q3_K / IQ3_S à ≈ 470 GB/s (B1) | base +5 t/s, MTP +4 t/s | élevé, plusieurs sessions | ★★ |
| R11 | All-reduce : moins de copies, d'événements et de lancements (B6) | base +1 t/s | moyen-élevé | ★ |

Projection, à valider levier par levier :
- sans MTP, R4 + R5 + R7 + R9 + R10 + R11 ramènent le token d'environ 44 à environ
  26 ms, **≈ 35-38 t/s, au-dessus des 26 t/s de llama.cpp** ;
- avec MTP, R1 + R6 + R8 donnent **≈ 35 t/s** avant même les leviers de base.

### R1 — Découpage GDN en TP pour qwen35

Voir §1.3.

### R2 — Protocole thermique du banc

Fichiers : `~/Workspace/flambeau-exp/runfl.sh`, `bench_lc.sh`, `bench_step4*.sh`.

Approche :
- avant chaque serveur, attendre une jonction ≤ 60 °C sur les deux GPU
  (`rocm-smi --showtemp`) ;
- alterner flambeau et llama.cpp invite par invite, ou recharger le serveur entre les
  passes ;
- consigner la température et la fréquence sclk avant et après chaque requête ;
- donner deux chiffres : « à froid » (première requête) et « régime chaud »
  (10e requête).

Validation : relancer `bench_step4.sh` et `bench_lc.sh` en suivant ce protocole.

Mesure attendue : l'écart à froid sans MTP est d'environ 22,6 contre 26,3 t/s (−14 %),
et non −20 %.

### R3 — Décalage de position au décodage

Fichier : `crates/server/src/routes/decode_loop.rs`, aux quatre sites `:621`, `:826`,
`:1211` et `:1727` (`prompt_ids.len() + step [+ mtp_offset]`).

Approche :
- utiliser `prompt_ids.len() + step - 1`, puisque le token issu du préremplissage est à
  la position L ;
- ou faire démarrer la boucle à `step = 0` en ajustant `max_tokens` ;
- vérifier `mtp_offset`, le chemin mixte (`dispatch_mixed_with_pending`) et la
  restauration depuis le cache de préfixes ;
- après correctif, l'état caché en attente `(slot, L-1)` correspond à la première
  position décodée, et le repli `MtpLead::Zero` disparaît.

Validation :
1. Rejouer `lp_order.py` : `list,list`, puis `code`, puis `list`. La sortie de « list »
   doit être identique quelle que soit la requête précédente. Aujourd'hui, elle diverge
   au caractère 356.
2. Ajouter un test de session : générer N tokens par `forward_one_token` aux positions
   L, L+1, …, puis vérifier que les logits égalent ceux d'un préremplissage de
   prompt+générés.

Mesure attendue : sorties déterministes d'une requête à l'autre, acceptation MTP
égale ou meilleure.

### R4 — Split-K de l'attention de décodage pour tout contexte

Fichier : `crates/forward/src/core/composites/standard_attn.rs`, conditions
`use_splitk` aux lignes `:770`, `:936` et `:1422` (`max_dec_kv > 256`, `n_chunks > 1`).
Taille de bloc fournie par `flambeau_model_ops::splitk_chunk_size`.

Approche :
- abaisser le seuil à environ 32 positions et forcer `n_chunks` ≥ 4 ;
- autre option : une ligne de dispatch par plage de contexte dans
  `dispatch/hip/gfx906.toml` (règle 1 : pas de constante cachée si le dispatch peut
  porter le choix) ;
- étendre le certificat split-K aux courts contextes :
  `cargo run -p bench -- sweep --arch gfx906 --op attention` avec des formes
  `n_kv ∈ {33, 64, 128, 256}`.

Validation : sweep vert, puis rocprofv3 sur la requête « math » de 130 tokens.

Mesure attendue :
- `attention_decode*` ≤ 1 ms par GPU et par token, contre 3,6 ;
- 44,2 → ≈ 41,6 ms, **≈ 24 t/s à froid**.

### R5 — MMVQ Q4_K

Fichiers :
- `dispatch/hip/gfx906.toml:63-68` (impl `qmatmul_q4_k_mmvq_dp4a`, r2, 64 threads) ;
- noyaux `flambeau_mmvq_q4_k_r2_dp4a_q8_1*` dans `crates/kernels-hip`.

Approche :
1. Tester par microbanc (`bench sweep --op qmatmul`, avec compteurs PMC) les formes
   réelles du TP-2 :
   - 8 704 × 5 120 (gate/up attn, moitié) ;
   - 5 120 × 8 704 (down) ;
   - 3 072 × 5 120 ;
   - 6 144 × 5 120.
2. Comparer trois variantes :
   - le r2 à 64 threads actuel ;
   - une variante 256 threads une ligne, sur le modèle de `mmvq_q5_k_dp4a_q8_1` ;
   - le schéma de llama.cpp `mul_mat_vec_q<GGML_TYPE_Q4_K, 1>`, avec ses
     `calc_nwarps` et `calc_rows_per_block` pour GCN.
3. Garder la meilleure dans le dispatch.

Validation :
- certificat vert ;
- l'appel 8 704 lignes doit passer de 120 µs à ≤ 75 µs ;
- `an_type.py` : Q4_K ≥ 310 GB/s.

Mesure attendue : −3,4 ms par token, **≈ 24,5 t/s à froid** (cumulable avec R4).

### R6 — Cycle MTP : vérification par lot et argmax sur GPU

Fichiers :
- `crates/models/qwen35-v2/src/model.rs` (`head_row`, boucle d'acceptation `:236-262`,
  boucle des brouillons) ;
- `crates/forward/src/core/composites/mtp.rs` (`mtp.rs:212-228`, lm_head des
  brouillons) ;
- `crates/forward/src/engine.rs` (`output_head`).

Approche :
1. **Vérification par lot.**
   - Un seul `output_head` sur les 5 lignes de `resid`. Le noyau
     `mmvq_q6_k_q8_1_batched_n*` existe déjà sous forme n2 ; ajouter n5 et le
     certifier.
   - Puis un argmax sur GPU, ligne par ligne, quand l'échantillonnage est glouton : une
     seule copie de 5 × u32 vers l'hôte.
   - L'acceptation par correspondance se fait sur l'hôte, à partir de ces 5 entiers.
2. **Brouillons.**
   - Argmax sur GPU, avec le token réinjecté directement dans l'embedding suivant, sans
     copie vers l'hôte.
   - Les ajustements de tokens d'arrêt (`stop_adjust`) s'appliquent comme biais de
     logits sur le GPU.
3. Garder le chemin hôte actuel pour température > 0 (rejet probabiliste).

Validation :
- mêmes sorties gloutonnes que le chemin actuel sur les 5 invites (`lp.py`, comparaison
  octet par octet du texte) ;
- trace MTP : lm_head par cycle ≤ 5,2 appels, contre 7,72 ; trous > 200 µs par cycle
  ≤ 2, contre ≈ 11.

Mesure attendue : cycle à froid 91 → ≈ 81 ms, **MTP +12 %**.

### R7 — rmsnorm_q8_1_fused

Fichier : `crates/ops/src/launch/norm.rs` (`rmsnorm_quant_q8_1`, `LaunchCfg::one_d(m, 256)`)
et le noyau correspondant.

Approche, au choix :
- 1 024 threads par ligne ;
- une variante à deux passes multi-workgroups quand m est petit (somme des carrés par
  workgroup, réduction, puis normalisation et quantification par blocs de 32) ;
- vérifier si le noyau fusionné `p2p_allreduce_residual_rmsnorm_q8_1` peut absorber
  ces appels.

Validation : sweep de norme ; 27,4 → ≤ 8 µs par appel.

Mesure attendue : −1,3 ms par token, +0,7 t/s.

### R8 — MTP dans la fenêtre d'arrêt et en streaming

Fichier : `crates/server/src/routes/decode_loop.rs` :
- `in_stop_window` `:1214` ;
- `factory` `:1229-1255` ;
- `run_completion_blocking_streaming` `:1471` ;
- `run_completion_streaming_scheduler_pp_blocking` `:716`.

Approche :
- dans la fenêtre, passer `f32::NEG_INFINITY` à tous les identifiants d'arrêt dans
  `stop_adjust`. C'est le masque déjà utilisé hors MTP. Le cycle reste cohérent, car
  brouillons et vérification voient le même masque ;
- factoriser le pas MTP pour que les chemins streamés l'appellent, en émettant les
  tokens d'un cycle un par un en SSE.

Mesure attendue : `plain_steps` tombe à environ 0 par requête, +3 % en MTP ; en
streaming, le MTP devient actif (≥ 26 t/s au lieu de 18).

### R9 — lm_head découpée par vocabulaire

Fichiers : `crates/forward/src/loader/globals.rs` (`load_lm_head`) et la tête de sortie
TP dans `engine.rs`.

Approche :
- chaque rang charge la moitié des lignes du vocabulaire ;
- en glouton, chaque rang calcule son argmax local, puis on échange une paire
  (valeur, indice) en P2P ou par l'hôte ;
- sinon, rassembler les logits.

Gain : −0,9 ms par token en base ; −0,83 ms × 5 appels par cycle en MTP.

### R10 — Efficacité générale des MMVQ (structurel, plusieurs sessions)

Point de départ : `an_type.py` et `an_grid.py` sur les traces fournies.

Approche :
1. Mesurer d'abord les compteurs PMC (`VALUBusy`, `MemBusy`, `MemStall`, vagues par
   SIMD) sur `mmvq_q5_k_dp4a_q8_1` 5 120 × 5 120 et `mmvq_iq4_xs_dp4a_q8_1` 8 704 × 5 120.
   C'est la règle de mesure n° 6 de CLAUDE.md.
2. Hypothèse à confirmer : avec une ligne par workgroup de 256 threads et 20
   super-blocs par ligne, le parallélisme mémoire par CU est insuffisant (noyau limité
   par la latence, pas par la bande passante). Cela expliquerait aussi la sensibilité à
   sclk observée en B5.
3. Si c'est confirmé, passer au schéma llama.cpp : plusieurs lignes par workgroup
   (`rows_per_block`) et `nwarps` réglés pour gfx906, ou plus de lignes par vague.

Cible : ≥ 450 GB/s par type, mesuré sur les formes TP-2.

Validation : certificat de chaque type, puis banc complet.

### R11 — All-reduce TP

Approche :
- remplacer la paire `hipMemcpyPeerAsync` + événement + noyau par un noyau qui lit
  directement le partiel du pair par BAR P2P (`backend-hip/src/bar_p2p.rs`) avec un
  drapeau de synchronisation ;
- fusionner `p2p_l2_flush` dans le noyau d'all-reduce ;
- au passage, réduire les 132 `hipMemcpyAsync` par token (≈ 4 par couche GDN).

Mesure attendue :
- par token et par GPU : `copyBuffer` 193 → < 70, `hipStreamWaitEvent` 256 → 0 ;
- −2 ms par token.

---

## 7. Limites de l'analyse

- Le gain d'acceptation de R1 est **estimé** : le mode PP de flambeau ne gère pas le
  MTP. Les preuves sont la fidélité des logits (§1.1) et l'identité du reste du
  pipeline MTP.
- La trace llama.cpp est prise sans graphes HIP et via `llama-bench`. Les durées de
  kernels sont comparables ; l'inactivité de llama.cpp en production est plus faible
  que dans la trace.
- Les sorties gloutonnes des deux moteurs diffèrent (à cause du bogue R1), donc
  l'acceptation par invite ne compare pas exactement les mêmes textes.
- Les gains en t/s se cumulent en millisecondes par token, pas en pourcentages. Chaque
  levier doit être remesuré seul, selon le protocole R2.
