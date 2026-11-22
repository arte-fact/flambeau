# R12 — MTP sur le chemin streamé (SSE)

Rédigé par Fable (coordination), pour Opus. État de référence : HEAD `4d6f9509`.

## Constat

- `crates/server/src/routes/decode_loop.rs` (1 900 lignes) contient quatre boucles de
  décodage :
  - `run_completion_scheduler_pp_blocking` (`:525`) et
    `run_completion_streaming_scheduler_pp_blocking` (`:716`), chemin ordonnanceur ;
  - `run_completion_blocking_ids` (`:951`), chemin historique non streamé, **le seul à appeler
    `mtp_step`** (`:1198-1274`) ;
  - `run_completion_blocking_streaming` (`:1485`), chemin historique streamé, sans MTP.
- Avec `--mtp N`, `scheduler_can_engage` (`:326`) renvoie `false` : les requêtes vont sur les
  chemins historiques. Une requête `stream: true` décode donc sans MTP.
- Mesure du 19/11 (binaire `e073c460`, `--mtp 4`, `stream: true`) : 18,2 t/s et aucun cycle
  MTP, contre 26,5 t/s en non streamé à l'époque. Aujourd'hui l'écart attendu est d'environ
  28 contre 38 t/s.

## Contrainte d'architecture

Règle 14 (CLAUDE.md) : quatre boucles de plusieurs centaines de lignes existent déjà. Copier
le bloc MTP (≈ 80 lignes, plus la gestion de la fenêtre d'arrêt de R8) dans une autre boucle
est exclu.

## Travail demandé

1. **D'abord un inventaire, sans coder** : tableau côte à côte des deux boucles historiques
   (`:951` et `:1485`) : état porté, étapes par token (masque d'arrêt, pénalités,
   échantillonnage, fermeture du raisonnement, JSON, logprobs, détection d'arrêt, émission).
   Indique ce qui est identique, ce qui diffère réellement, et ce qui ne diffère que par
   l'émission (accumuler contre envoyer un fragment SSE). Envoie-moi ce tableau et ta
   proposition de découpage avant d'écrire le code.
2. **Découpage visé** (à confirmer par ton inventaire) : une seule boucle historique qui
   produit les tokens, pas simple ou cycle MTP, et les remet à un récepteur ; deux récepteurs,
   l'un qui accumule (non streamé), l'autre qui émet les fragments SSE. Le cycle MTP rend
   plusieurs tokens d'un coup : le récepteur SSE les émet un par un, dans l'ordre, et
   l'arrêt (token d'arrêt, `max_tokens`, séquence d'arrêt, client déconnecté) doit pouvoir
   tomber au milieu d'un cycle sans émettre les tokens suivants.
3. Les deux boucles de l'ordonnanceur ne sont pas dans le périmètre de cette étape ; ne les
   modifie pas, mais dis dans l'inventaire si le même récepteur pourrait les servir plus tard.
4. Pas de nouveau flag ni de variable d'environnement (règle 1). Pas de branche par
   architecture dans ce fichier partagé (règle 13).

## Validation

- Égalité stricte : pour les 5 invites de `lp.py` en glouton, le texte streamé (concaténation
  des `delta.content`) est identique octet pour octet au texte non streamé, avec `--mtp 0` et
  avec `--mtp 4`. Le texte non streamé lui-même ne change pas par rapport à HEAD `4d6f9509`.
- `finish_reason` et `usage.completion_tokens` identiques entre streamé et non streamé.
- Débit streamé avec `--mtp 4` : ≥ 95 % du débit non streamé (38,3 t/s à ce jour), mesuré par
  le Vibe avec `analyse-ecarts/lp_order.py` (qui streame).
- Débit non streamé inchangé avec et sans MTP (28,15 et 38,28 t/s à ± 3 %).
- Un cas d'arrêt au milieu d'un cycle : `max_tokens` choisi pour tomber dans un cycle (par
  exemple 30, 31, 32, 33), même texte en streamé et non streamé.
