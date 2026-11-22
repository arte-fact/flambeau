# r5 — plan : graphe CUDA de décodage pour qwen35 (base a45374a0, lecture seule, rien codé)

But : que le décodage n = 1 de qwen35 sur sm_86 passe par le graphe capturé (`SoloSession::forward_decode_graph`) au lieu de 1554 lancements individuels par token (coord/r5-chemin-cuda.md). Aucun gain n'est annoncé ici : la trace nsys dira quelle part du temps se passe entre les noyaux.

## 1. Append KV de qwen35 vers les variantes `_dpos` (standard_attn.rs, branche `prefill_shape`, l.565-718)

État actuel : quatre branches choisies sur `attn_v_unit_norm_w` (présence d'un poids, pas un nom d'architecture) et `is_q8`. Les deux branches sans V unit-norm (celles de qwen35) refusent la capture.

**Q8 : fusionner les deux branches Q8.** C'est moins de code et aucun nouveau noyau.
```
} else if is_q8 {
    if weights.attn_v_unit_norm_w.is_some() {
        v_unit_norm_per_head_f16(&mut v_for_norm, …)?;   // inchangé (gemma4)
    }
    if state.capture.is_some() {
        kv_append_f16_to_q8_dpos(…, state.pool.decode_ctx_i32)?;
    } else {
        kv_append_f16_to_q8(…)?;
    }
}
```
- La suite d'appels de gemma4 Q8 reste identique, et celle de qwen35 hors capture aussi (`kv_append_f16_to_q8`, deux `quantize_f16_q8_0`).
- On supprime le `mark_uncapturable("kv_append: Q8 append not captured")` et environ 25 lignes de doublon.
- `kv_append_f16_to_q8_dpos` existe déjà (model-ops/ops/kv_append_f16_to_q8.rs:140, noyau CUDA `quantize_f16_q8_0_dpos.cu`).

**F16 : un noyau CUDA de copie `kv_append_f16_dpos`** (lignes K et V à la ligne `decode_ctx_i32[0]`, ring_depth géré comme `kv_append_v_unit_norm_f16_dpos`).
- Aujourd'hui l'append F16 est un `cuMemcpyDtoDAsync` dont la destination dépend de la position. Le plan de rebind ne sait réécrire que la source d'une copie (`RebindValue::MemcpySrc`), pas sa destination ; d'où un noyau.
- Placement : `kernels-cuda/src/kernels/kv_append_f16_dpos.cu`, comme les cinq `_dpos` existants. La règle 5 est respectée : l'intrinsèque de lecture de la position reste côté CUDA.
- Plomberie :
  - une méthode `Ops::kv_append_f16_dpos`, implémentée sur les deux backends, comme `kv_append_v_unit_norm_f16_dpos` (même forme de trait) ;
  - un `else` symétrique de la branche Q8 dans standard_attn.rs.
- Priorité : seconde. Le run mesuré est en `--kv q8` ; on peut livrer Q8 seul d'abord.

**Seuil d'entrée** : `DECODE_GRAPH_MIN_KV = 257` (solo.rs:25) est périmé.
- Le split-K s'enclenche dès n_kv > `SPLITK_MIN_TOKENS_KV` = 32 (launch/attention.rs:1075), en Q8 comme en F16.
- À remplacer par `SPLITK_MIN_TOKENS_KV + 1`, importé et non recopié. Sinon les ≈ 250 premiers tokens de chaque requête resteraient hors graphe.

**Règle 13** : aucun test d'architecture ajouté. Le choix reste piloté par la présence de `attn_v_unit_norm_w`, par le type de cache (`KvLayout`) et par `state.capture`.

## 2. Autres obstacles à la capture

| obstacle | état | action |
|---|---|---|
| HtoD des positions (16 par token, 4 o) | Déjà hors capture : sous capture, standard_attn ne copie pas (`if state.capture.is_none()`, l.302) ; le pilote charge `position_i32` et `decode_ctx_i32` (4 o + 8 o) avant chaque replay, sur le même flux. | Rien d'obligatoire. Option : buffers hôtes épinglés (§2b). |
| Embedding | Déjà géré : sous capture, `embed_gather_f16_dpos` (certifié). | Rien. |
| Historique de convolution GDN (DtoD de 122 880 o ×48) | Adresses fixes par slot (`conv_history`, `scratch.conv_input`) : capturé comme nœud memcpy. Aucun argument dépendant de la position dans `forward_decode_with_ar_hook`, aucune synchronisation. | Rien. À vérifier par une capture qui réussit. |
| Logits | Sous capture, `output_head` sort avant la copie ; le pilote fait la copie GPU→hôte hors graphe vers `last_logits` (`Vec<f32>` paginé, 993 280 o), puis synchronise. | Option : copie vers un buffer épinglé (§2b). |
| Attention split-K | Variante `_dpos` Q8 déjà branchée sous capture (standard_attn.rs:787). | Rien pour la capture ; voir §2c pour la réutilisation. |
| Une passe (n_kv ≤ 32) | Non capturable par construction. | Couvert par le nouveau seuil (§1). |

**2b. Mémoire épinglée** (facultatif, à décider sur la trace) :
- quoi : `position_stage_host`, `decode_ctx_host` et un buffer de logits de `vocab` f32 alloués en mémoire épinglée ;
- comment : une méthode `Device::alloc_host_pinned` / `free_host_pinned`, avec une vraie implémentation sur les deux backends (`cuMemHostAlloc`, `hipHostMalloc` ; pas de défaut vide, règles 3 et 12) ;
- effet attendu : la copie de 970 Ko devient un vrai DMA asynchrone, et les deux petites copies HtoD ne passent plus par la copie de staging en mémoire paginée ;
- à faire seulement si `cuda_api_sum` montre un coût de `cuMemcpyDtoHAsync` / `cuMemcpyHtoDAsync` significatif.

**2c. Re-captures** :
- `reusable_at` (capture.rs:262) exige le même `(n_chunks, chunk_size)`. Avec MAX_SPLITK_CHUNKS = 32, chunk_size vaut 32 de n_kv 33 à 1024, donc n_chunks change tous les 32 tokens. Cela donne une re-capture et une instanciation d'un graphe d'environ 1550 nœuds tous les 32 tokens ; au-delà de 1024, environ tous les 128.
- Levier disponible sans nouveau noyau : le noyau `_dpos` Q8 accepte une grille plus large que nécessaire (blocs vides, testés par le cas `bake_n_chunks: Some(MAX_SPLITK_CHUNKS)` du sweep).
- Mise en œuvre : graver n_chunks = 32 tant que chunk_size reste 32 (n_kv ≤ 1024), et adapter `AttnGridWitness` / `reusable_at` à « même chunk_size, n_chunks ≤ gravé ». Un seul graphe couvrirait alors les 1024 premières positions.
- Coût : 24 × 32 blocs d'attention par couche, dont des blocs vides tant que n_kv est petit.
- À trancher sur la trace (coût de `cuGraphInstantiate` contre le coût des blocs vides). Pas dans le premier commit.

## 3. Certificats (règle 2)
Les `_dpos` sont lancés par nom (direct-call, catalogue `DIRECT_CALL_KERNELS_SM86`, réciprocité testée par backend-cuda `impls::tests`). Aucune ligne de dispatch n'est ajoutée.

- `quantize_f16_q8_0_dpos_sm_86` : certifié, mais ses formes s'arrêtent à kv_width 256/512/2048. **Ajouter la forme qwen35 27B** (kv_width 1024 = 4 × 256, positions 33, 300, 1641 ; un cas en anneau existe déjà) et régénérer le certificat.
- `attention_decode_q8_kv_splitk_dpos_sm_86` : certifié pour head_dim 256, mais en GQA 16/2 et 4/1. **Ajouter 24/4/256** (qwen35 27B, GQA 6) :
  - position 33 (premier split-K) ;
  - position 300 ;
  - position 1641 ;
  - un cas avec grille gravée à 32 si §2c est retenu.
- `embed_gather_f16_dpos_sm_86` : ajouter hidden 5120 si absent.
- Nouveau `kv_append_f16_dpos_sm_86` (si F16 est fait) : sweep au bit près contre la copie DtoD de référence (formes kv_width 1024 et 2048, avec et sans anneau), certificat, entrée DC, test de réciprocité.
- Contrôle de bout en bout, comme eo :
  - textes identiques au bit près entre `--no-decode-graph` et le graphe actif sur le 27B (invites lp.py, KV Q8 puis F16) ;
  - `parity_O*` vert ;
  - cert-check des deux architectures (gfx906 inchangé, sm_86 avec les certificats régénérés).
- Mesure : trace nsys du même scénario que r5-B ; le nombre de lancements par token doit passer de 1554 à 1 `cuGraphLaunch`.

## 4. Effet sur gfx906
- `Backend::supports_decode_graph()` vaut `false` pour HIP (backend/src/lib.rs:52, défaut ; seul cuda.rs:21 renvoie `true`). Donc `forward_decode_graph` est inatteignable et `state.capture` vaut toujours `None` sur les MI50.
- Les seules modifications du chemin sans capture sont :
  - la fusion des deux branches Q8 (même suite d'appels, même ordre) ;
  - le seuil `DECODE_GRAPH_MIN_KV`, lu seulement par le pilote du graphe.
- Les noyaux `_dpos` (existants et nouveaux) sont dans `kernels-cuda`, donc non compilés pour HIP : rien de nouveau à certifier sur gfx906.
- Le choix « graphe ou non » reste une capacité du backend, avec une vraie valeur des deux côtés, et non une variable d'environnement (règle 1). Le graphe HIP, nul ou négatif sur gfx906 selon CLAUDE.md, n'est pas activé.
- Contrôle : bin gfx906 avant/après sur les invites eo (88/88, textes identiques, débit au bruit près).

## Découpage proposé (un commit par point)
1. Q8 : fusion des branches et seuil `SPLITK_MIN_TOKENS_KV + 1`. Avec les formes 24/4/256 et kv_width 1024 ajoutées aux deux sweeps `_dpos`, et les deux certificats sm_86 régénérés.
2. F16 : noyau `kv_append_f16_dpos`, méthode `Ops`, sweep, certificat, entrée DC.
3. Selon la trace : mémoire épinglée (§2b), puis grille gravée à 32 chunks (§2c).
