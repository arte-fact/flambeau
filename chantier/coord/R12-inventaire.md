# R12 — inventaire des deux boucles historiques (HEAD 4d6f9509)

A = `run_completion_blocking_ids` (:951-1479, non streamé). B = `run_completion_blocking_streaming` (:1485-1900, streamé).

| Étape | A | B | Nature |
|---|---|---|---|
| Entrée | `prompt_ids` | `String`, tokenise lui-même | interface |
| Aiguillage ordonnanceur, prise de slot, prefill par blocs | identique | identique | commun |
| Capture du cache de préfixes | avant le 1er échantillonnage | après émission du 1er token (copie des logits non masqués, hors TTFT) | moment seulement |
| Masque d'arrêt du 1er token, échantillonneur, `ReasoningCloser`, `stop_bias`, `min_response_tokens` | identique | identique | commun |
| Mode JSON (masque, `feed_slice`, arrêt sur JSON complet) | oui | **absent** | réel |
| Logprobs | oui | **absent** | réel |
| MTP (`mtp_step`, fenêtre d'arrêt de R8) | oui (:1198-1374) | **absent** | réel, objet de R12 |
| `force_mask` | `generated.len()` (R8) | `step` | identique sans MTP |
| Détection d'arrêt par chaîne | fenêtre de 16+ tokens re-décodés, tous les 4 pas ou après `min_response` ; marqueurs de raisonnement + **`chat_stop_markers()` de l'arch** + séquences utilisateur | fenêtre de 64+ octets du texte émis, à chaque pas ; marqueurs + séquences utilisateur, **sans `chat_stop_markers()`** | réel (B ne s'arrête pas sur les marqueurs d'arch) |
| Émission | accumule ; `finalise()` à la fin : retire les tokens d'arrêt, sépare le raisonnement, **tronque à la 1re séquence d'arrêt, `trim_end()`** | détokenisation incrémentale (gestion de U+FFFD), `emit(delta)` ; le token d'arrêt n'est pas émis ; client parti → `finish_reason "stop"` ; ni troncature ni `trim_end` | émission + post-traitement |
| Sortie | `CompletionOutput` (texte, logprobs, compteurs, raison) | `(raison, prompt_tokens, completion_tokens)` | interface |
| Profilage `FLAMBEAU_PROFILE_DECODE` / `FLAMBEAU_HOST_PROFILE` | absent | présent (env, ~130 lignes) | outillage |

Défauts relevés dans A (chemin MTP), à corriger au passage :
1. Après un cycle, `continue` saute la détection d'arrêt par chaîne : séquences utilisateur et marqueurs ignorés pour les tokens MTP.
2. Après un cycle, A applique quand même masque et `sampler.sample` sur `logits_buf`, puis jette le résultat. Coût hôte, et l'état RNG avance pour rien quand température > 0.
3. Le rythme « tous les 4 pas » de la détection compte des cycles, pas des tokens.

## Découpage proposé

- `decode_tokens(state, slot, prompt_ids, params, relax, sink: &mut dyn TokenSink) -> DecodeOutcome { generated, finish_reason, logprobs, mtp_cycles }` : prise de slot, prefill, 1er token, boucle pas simple ou cycle MTP, JSON, logprobs, et une seule détection d'arrêt appliquée **token par token**, y compris dans un cycle.
- `trait TokenSink { fn token(&mut self, tok: u32, generated: &[u32]) -> Result<bool>; }` (false = arrêter). Le cœur passe les tokens d'un cycle un par un et s'arrête au milieu dès que l'un des cas l'exige : token d'arrêt, `max_tokens`, chaîne d'arrêt, JSON complet, ou `false` du récepteur.
- Récepteurs : `Accumulate` (no-op ; A appelle ensuite `finalise`) et `SseSink` (la détokenisation incrémentale de B, déplacée telle quelle).
- A et B deviennent deux enveloppes de quelques lignes. B garde son interface publique (`messages.rs:652` et `stream_completion_sse` l'appellent).
- Ordonnanceur hors périmètre ; ses deux boucles émettent aussi token par token et pourraient prendre le même `TokenSink` plus tard (pas de MTP chez elles aujourd'hui).
- Volume attendu : A+B passent de ~950 à ~500-550 lignes, soit **−350 à −450**, dont ~130 lignes de profilage par variable d'environnement si on les retire (voir désaccord 3).

## Risques et désaccords avec la consigne

1. **Égalité octet pour octet streamé/non streamé : déjà fausse à HEAD, indépendamment du MTP.** `finalise()` tronque à la séquence d'arrêt et fait `trim_end()` ; le flux SSE ne le fait pas. Un texte qui finit par « \n » diffère donc dès `--mtp 0`. Proposition : critère « égal après `trim_end` et hors séquence d'arrêt utilisateur », ou retenir en flux les octets en attente (blancs finaux, préfixe possible d'une séquence d'arrêt) pour obtenir l'égalité stricte. Ce second choix change le comportement SSE actuel : à trancher.
2. Unifier la détection d'arrêt change B (il s'arrêtera aussi sur `chat_stop_markers()`) et A (vérification à chaque token au lieu de tous les 4 pas, donc arrêt parfois plus tôt). Le texte non streamé peut donc bouger dans ces cas d'arrêt, alors que la consigne exige « inchangé par rapport à HEAD ». Sur les 5 invites `lp.py` sans séquence d'arrêt, c'est sans effet attendu ; à confirmer par la mesure.
3. Profilage par variable d'environnement (règle 1) dans B : je propose de le retirer plutôt que de le généraliser au cœur ; sinon il reste dans le récepteur SSE seulement.
4. JSON et logprobs en streaming : le cœur les rend possibles en SSE, mais je propose de ne pas les activer dans cette étape (hors périmètre, interface SSE à définir).
