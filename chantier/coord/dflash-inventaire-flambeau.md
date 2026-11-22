# Étape 6 — inventaire flambeau pour le drafter DFlash2 (lecture seule, base ee32a49c / 1ad6047e ; sans GPU)

Cible : Qwen3.8-27B-UD-Q4_K_M (16,46 Go). Drafter : `~/models/Qwen3.8-27B-DFlash2-Q8_0.gguf` (2,06 Go, arch `dflash`). Côté llama.cpp, voir coord/dflash-etude-llamacpp.md.

1. **Prises de hiddens aux couches 6/20/34/48/62 : elles n'existent pas.**
   - Le MTP ne capture qu'**une ligne** de l'état **final après `output_norm`** : `capture_pending` (qwen35-v2/src/model.rs:69-92), via `capture_hidden_from` (forward/src/engine.rs:1087). Il le fait après la tête (model.rs:50-62) et à la fin du cycle (model.rs:278).
   - DFlash2 veut l'**entrée** (le résiduel avant `attn_norm`) de 5 couches, et pour **toutes les lignes** de chaque forward (préremplissage, décodage, vérification).
   - À ajouter : une prise dans la boucle `for li in layers` de `forward` (model.rs:22) et de `verify_layers` (model.rs:96-106), qui copie `resid` vers un tampon [n, 5 × 5120] F16 du pool. C'est une copie DtoD par couche visée, ou un mini-noyau capturable sous graphe.
   - En TP-2 (MI50), le résiduel est complet sur chaque rang, donc on peut capturer sur le rang 0. En PP, les couches 6 à 62 sont réparties : il faudra un pair-à-pair.
2. **Vérification à bloc 8 (ancre + 7 brouillons, n_max ≤ block_size − 1) : m = 8, ou 9 si n_max = 8.**
   - Le pool la dimensionne déjà par `--mtp N` : `logits_rows = max(slots, max_draft + 1)` (core/scratch.rs:1236-1239), `mtp_verify_rows()` (scratch.rs:1134), et les lignes GDN de spéculation (scratch.rs:1140). Le plafond est `ARGMAX_MAX_ROWS = 16` (scratch.rs:145), donc 8 ou 9 lignes tiennent.
   - **qmatmul à m = 8 ou 9 sur sm_86** :
     - le row-tile prend m ∈ 2..=8 (ops/src/launch/qmatmul.rs:460) ;
     - à m = 9, `mmvq_batched` découpe en 8 + 1 pour les 8 types de qwen35, IQ3_S compris depuis 1ad6047e ;
     - la table n'est consultée au vrai m que depuis 092d0563 (branche g6, non fusionnée). Avant, elle l'était à min(m, 8), ce qui donne le même résultat aujourd'hui, puisque les lignes batched couvrent m ≥ 2.
   - **Sur gfx906** : il n'y a pas de ligne batched. Q8_0 passe en row-tile à m 2..=4, et les autres en MMQ lds_tiled ou x8 dès m ≥ 3, certifiés à m = 8 (77cf1d6a). m = 9 est donc couvert, mais **non mesuré**.
   - **Réutilisation du cycle MTP** (`mtp_cycle`, model.rs:146-290) :
     - se réutilisent tels quels : `verify_layers` (model.rs:212), `output_head_argmax_rows` (model.rs:221), l'acceptation gloutonne (model.rs:235-240), et `gdn_snapshot` / `gdn_replay_rows` (model.rs:210, 273) ;
     - restent propres au MTP : `ctx.mtp_draft` (model.rs:181), `mtp_stage_hidden`, `capture_pending` et `mtp_catchup` (model.rs:277-286).
     - Il faudra donc un `dflash_cycle` sur le même squelette, ou extraire « vérifier + accepter + rejouer GDN » en fonction commune. Avec MTP, cela fait 2 sites (règle 14).
3. **Ops manquantes ou à adapter.**
   - **Attention non causale du drafter** (cache injecté + bloc) : pas de nouveau noyau a priori. Les noyaux de préremplissage bornent `limit = q_offset + q_idx + 1` puis le plafonnent à `n_k_tokens` (kernels-shared/include/attention_flash_tile_q8.cuh:39-40 ; attention_flash_tile.cuh:42-44).
     - Passer `q_offset ≥ n_k − 1` rend donc chaque ligne non causale.
     - La fenêtre de 2048 doit alors passer par le glissement du pointeur KV (comme `in_band` au décodage), avec `window_size = 0`, puisque le calcul du début de fenêtre suppose un vrai `q_offset`.
     - À vérifier pour attention_prefill_mma (kernels-cuda/src/kernels/attention_prefill_mma.cu:49, `last_q`), et à certifier par un sweep dédié.
   - **Conv causale DFlash2** (noyau dynamique, 2 prises, 320 groupes, base F32 + proj Q8_0) : `causal_conv1d_f32` (ops/src/ops_trait.rs:796) a des poids statiques par canal, un état et un noyau de 4 ; il n'est pas réutilisable.
     - `conv_proj` (5120 → 1280) passe en qmatmul Q8_0 existant.
     - Le mélange à 2 prises dans le bloc est un **nouveau petit noyau** élément par élément.
   - **Sélecteur top-k 16** :
     - top-16 sur 248 320 logits × 8 lignes : `topk_softmax_f32` (ops_trait.rs:675, k ≤ 4096, sampler_topk_softmax.cuh:14) convient, par ligne. Une log-softmax ne décale le score unaire que d'une constante par ligne, donc l'argmax par ligne est inchangé.
     - Le score de paire `successor[cand]ᵀ (predecessor[prev] ⊙ gate)` (16 × 16 × 256 par ligne, plus `selector_hidden` 256 × 5120 en qmatmul) est un **nouveau noyau**.
     - Le parcours du treillis se fait sur l'hôte, comme llama.cpp.
   - **RoPE** : sections [64, 0, 0, 0], freq_base 1e7, positions `{p, p, p, 0}`. La rope partielle NeoX de qwen35 (lue dans `rope.dimension_count`, qwen35-v2/src/config.rs:70-71) est à vérifier pour 64 dimensions tournées sur 128.
   - **Injection** (`fc` 25 600 → 5120, Q8_0, puis RMSNorm, puis K/V par couche) : qmatmul, rmsnorm, rope et `kv_append` existants. C'est une composition, sans nouveau noyau.
4. **Partage de `token_embd` et de la sortie avec la cible : le modèle existe déjà.**
   - `mtp_draft` reçoit `&model.lm_head` et `&model.embedding` de la cible (model.rs:181-185). Le drafter fait de même : masque = ligne 248 070 de `token_embd` (Q4_K) ; logits via la lm_head Q6_K de la cible sur ses 8 lignes, au même m que la vérification.
   - **Chargement** : il faut un second GGUF (`-md`) avec son propre loader, une arch `dflash` dans un crate modèle (règle 4), et un `--draft-model` au serveur.
5. **Mémoire.**
   - **Poids** : drafter 2,06 Go, plus son KV (5 couches × K+V 1024 × fenêtre 2048 × F16 ≈ 42 Mo par slot), le tampon de prises (5 × 5120 × F16 × ubatch 512 ≈ 26 Mo) et 8 lignes de logits F32 (≈ 8 Mo, déjà couvertes par `max_draft + 1`).
   - **RTX 24 Go** : cible 16,46 Go + KV q8 8 k + pool, puis + ≈ 2,15 Go. L'occupation actuelle du serveur n'est **pas mesurée** ici ; à relever par nvidia-smi pendant un serve avant de conclure.
   - **MI50 TP-2 (2 × 32 Go)** : la cible fait ≈ 8,2 Go par rang. Le drafter répliqué (+ 2,1 Go par rang) ou sur le seul rang 0 tient largement.
   - Pas d'estimation de gain : aucune base mesurée chez nous (llama.cpp ne publie que des planchers d'acceptation : ≥ 0,518, longueur moyenne acceptée ≥ 4,54).
