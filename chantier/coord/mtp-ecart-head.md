# MTP-4 RTX : llama.cpp ffc87c3a7 (≈ 38 ms par cycle) contre flambeau f5d29f30 (≈ 42 ms), lecture de code seulement

Sources : llama.cpp = `common/speculative.cpp` (abrégé spec), `src/`, `ggml/src/ggml-cuda/ggml-cuda.cu` (abrégé ggml-cuda) ; flambeau = qwen35-v2/src/model.rs (`mtp_cycle`, :146-290), forward/src/engine.rs et forward/src/core/composites/mtp.rs. Rien n'est mesuré ici.

## 1. Graphes CUDA : chez eux sur tout le cycle, chez nous sur aucune étape
- **llama.cpp.** Les graphes CUDA ne sont pas limités à n = 1. Le cache est indexé par graphe (`graph_key`, ggml-cuda:5476) : un graphe est capturé après 2 appels de même forme (ggml-cuda:5487-5512), puis relancé par `cudaGraphLaunch` (ggml-cuda:3392-3404). Seul `MUL_MAT_ID` avec synchronisation le désactive (ggml-cuda:3236-3247), et il n'y en a pas dans un modèle dense.
- La vérification (n = k + 1), le rattrapage MTP (`process`, spec:2055) et chaque pas de brouillon (spec:2175) sont donc rejoués en graphe.
- **flambeau.** Le graphe ne couvre que le décodage à un token hors MTP (runtime/solo.rs:124-145). Dans `mtp_cycle`, toutes les étapes sont lancées noyau par noyau, hors graphe :
  - `mtp_draft` (model.rs:181 → mtp.rs:213-288) ;
  - `verify_layers` (model.rs:212), dont `embed` marque d'ailleurs n > 1 non capturable (embed.rs:47-52) ;
  - la tête (model.rs:221) ;
  - le rejeu GDN (model.rs:273) ;
  - `mtp_catchup` (model.rs:279).
- Je n'ai pas compté les lancements par cycle. Un profil nsys d'un cycle de chaque côté donnerait la part de ce poste.

## 2. Retour arrière de l'état GDN : un index chez eux, une copie, une synchro et un rejeu chez nous
- **llama.cpp.** Avec `n_rs_seq = draft.n_max` (common.h:417-427), l'op `ggml_gated_delta_net` écrit elle-même les K = n_max + 1 derniers états par token pendant la vérification (models/delta-net-base.cpp:98-140). La conv fait de même (llama-graph.cpp:4187-4215). Un rejet ne fait que changer un index, `set_rs_idx` (llama-memory-recurrent.cpp:198-215) : pas de copie avant la vérification, pas de passe de rejeu.
- **flambeau.**
  - Avant la vérification, `gdn_snapshot` fait 96 memcpy D2D : état et historique de conv sur 48 couches GDN, environ 144 Mio. Il les fait suivre d'un **`Stream::synchronize`** (engine.rs:1191-1232, appelé model.rs:210).
  - Cette synchro bloque l'hôte avant les lancements, non capturés, de la vérification.
  - Sur rejet, il y a un noyau `gdn_state_step` et un memcpy par couche GDN (engine.rs:1321-1395, model.rs:268-275).
- **Octets** : l'écart n'est pas en leur faveur. Ils écrivent 5 états par couche à chaque vérification (≈ 720 Mio) ; nous copions ≈ 144 Mio, plus ≈ 144 Mio de rejeu en cas de rejet. Ce qui diffère, c'est donc la synchro et les lancements supplémentaires, pas la bande passante.

## 3. Synchronisations hôte par cycle (greedy, k = 4)
- **flambeau.**
  - 4 pour le brouillon : un token de 4 o chacun (mtp.rs:246-260) ;
  - 1 pour `gdn_snapshot` (engine.rs:1229) ;
  - 1 pour les argmax de vérification (mtp.rs:374) ;
  - 1 pour `logits_row_to_host`, soit 1 ligne de 248 320 × 4 o ≈ 1 Mio (engine.rs:1174-1188, model.rs:240) ;
  - plus celles de `capture_pending` et `mtp_catchup`, que je n'ai pas lues.
- **llama.cpp.**
  - 4 pour le brouillon : échantillonnage top-10 sur le backend, activé par défaut (common.h:351, spec:1872-1886) ;
  - 1 pour les logits de vérification : k + 1 lignes, ≈ 5 Mio (llama-context.cpp:2433) ;
  - 1 ou 2 lectures de ligne h de 20 Ko (spec:2095, 2293).
- Le nombre de synchros est proche, et ils copient plus de logits. La seule synchro qu'ils n'ont pas est celle du snapshot GDN (point 2).

## 4. Postes équivalents (rien à gagner a priori)
- La tête de brouillon passe par la lm_head complète de la cible, en Q6_K, à m = 1 (qwen35.cpp:656-661 ; mtp.rs:226-238).
- La vérification fait une seule passe de lm_head sur k + 1 lignes, des deux côtés (model.rs:221).
- Le rattrapage MTP fait une passe sur le lot vérifié chez eux (spec:2007-2062, lignes h restées sur le device), contre les lignes acceptées seulement chez nous (model.rs:279-286). C'est la même structure.
- `p_min` vaut 0 par défaut (common.h:349). Leurs 1 510 brouillons pour 383 cycles font ≈ 3,94 brouillons par cycle, contre 4 chez nous : c'est la même charge de brouillon.
- Embedding : chez eux, `ggml_get_rows` lit directement le Q4_K, sur la cible comme dans le MTP (qwen35.cpp:541-543). C'est le point (a) de VRAM, sans effet sur le temps de cycle.

## 5. Pour attribuer les ≈ 4 ms
Il faut un profil nsys d'un cycle stable, des deux côtés. Il dirait si les 4 ms viennent du temps hôte entre noyaux (point 1, plus la synchro du point 2) ou de noyaux plus lents à m = 5, la vérification GDN passant chez nous par le chemin préremplissage (composites/gdn.rs:102-130). Ce sont les deux hypothèses que je n'ai pas pu séparer en lisant le code.
