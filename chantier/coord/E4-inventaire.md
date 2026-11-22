# E4 — inventaire nettoyage / factorisation (HEAD 9512db81, lecture seule)

Classement bénéfice/risque : ★★★ = gros gain, risque faible … ★ = à discuter. Vérif. « std » = check --all-targets (3 configs) + tests GPU masqués + cert-check 2 arch + --all-dispatch 2 arch.

## (e) ★★★ Chemin de décodage à sortie F16 hors dispatch — `ops/launch/qmatmul.rs:1345-1525` (~180 l.)
- `mmvq_f16_direct` : table codée en dur (stem, entry, threads, rows) pour 19 dtypes ; garde dupliquée dans `forward/src/ctx.rs:395` (`supports_decode_to_f16`).
- Utilisé à m=1 par Q/K/V/O de l'attention (`standard_attn.rs:189,243,268,1496`) et ffn_down (`dense_ffn.rs:207`) : **ces projections n'utilisent pas les noyaux R10b** (Q4_K 256 threads, IQ4_XS 256 threads ; Q5_K exclu par la garde, son entrée r2 reste).
- Proposition : descripteur de dispatch avec variante « dst F16 » (règle 1) ; variantes `_f16` des noyaux t128 de R10b. Risque moyen (noyaux à écrire). Vérif. : std + `mmvq_f16_direct_parity` + mesure Vibe (levier de perf probable, non mesuré).
- Test `ops/tests/mmvq_f16_direct_parity` (échoue depuis R10b : Q4_K, 1 ligne, ≈ 6e-5) : il compare au bit près `mmvq` (F32 puis cast, noyau choisi par le dispatch) et `mmvq_f16_direct` (autre noyau), ce qui n'a de sens que si les deux côtés lancent le même noyau. Après (e) : (1) chaque variante `_f16` est comparée au bit près à **sa propre** variante F32 suivie du cast (même corps de noyau, seule l'écriture finale change) ; (2) la comparaison au flottant de référence se fait avec la tolérance du certificat de la ligne de dispatch ; (3) on garde la vérification de saturation à ±65 504.

## (i) ★★★ Sweeps historiques à arguments construits à la main — dérive de signature
- Corrigés (1654273b) : decode F16, prefill F16 + tuile flash, prefill Q8, decode Q8, gdn_alpha_beta.
- Non exercés par --all-dispatch, donc non vérifiés : `sweep_moe` (3 280 l., 174 push : tile8 ×18, q2_k/q3_k/q5_k, gate_up), `sweep_q4_0_q5_0` (52 push : q4_1, warpcoop64), `sweep_mmq` (dtypes hors dispatch), `sweep_quantize_q8_1_mmq`, `sweep_gdn_step_alphabeta` (36 push), `sweep_attention_prefill_q8_kv` (masqué par le sweep générique de même nom).
- Prévention : (1) les sweeps appellent les lanceurs de `ops/launch` (précédent : `sweep_attention_splitk`, `…q8_kv_splitk`, 0 push ; bench dépend déjà de flambeau-ops) ; (2) garde générique : `kernels-hip/build.rs` lit déjà `instantiations.toml` → émettre le nombre de paramètres par entrée, `KernelArgs` vérifie le compte au lancement (debug + sweep), noyaux `.cu` manuels inclus par analyse de la signature `extern "C"`.
- Commit 1 : garde de compte (+~80 l.) ; lancer `sweep` sur toutes les ops SIMPLE_SWEEPS (Vibe), corriger ce qui casse. Commit 2+ : portage vers les lanceurs, fichier par fichier.

## (b) ★★★ Boucles de l'ordonnanceur — `server/src/routes/decode_loop.rs:526-707` (bloquant) et `:717-~905` (streaming), ~370 l.
- Doublons de `decode_tokens` (R12) : masque d'arrêt, `stop_bias`, `min_response_tokens`, fenêtres d'arrêt différentes (16 tokens vs 64 octets), détokenisation propre (`push_and_emit`).
- Défauts connus : séquence d'arrêt émise en flux (pas de rétention), `completion_tokens` décalé de 1 entre les deux.
- Plan : extraire du cœur de `decode_tokens` une source de pas (`trait StepSource { fn next_logits(..) }` : forward direct, ordonnanceur, cycle MTP) ; les deux boucles deviennent `decode_tokens(.., source = Scheduler, sink = Discard|Forward)` → même `Emitter`/`TextTracker` (rétention), même compte de tokens. ~−250 l.
- Risque moyen (chemin batching). Vérif. : std + tests token_emit + Vibe : `lp.py` 5 invites ×{stream, non-stream} ×{1, 4 clients}, textes identiques à R12, `completion_tokens` égaux.

## (h) ★★ Doublons > 100 lignes (règle 14)
1. `Recipe::from_impl_id` (`ops/launch/qmatmul.rs:1817`, 108 bras ~650 l.) ∥ `KernelDescriptor` (`backend-hip/src/impls/qmatmul.rs`, 695 l.) : deux tables par impl_id. Fusion : stem/entry/threads/rows dans le descripteur. Risque moyen ; vérif. : std + test « chaque impl dispatché a une recette ».
2. Sweeps HIP historiques ∥ `sweep_pointwise_cb` (mêmes noyaux, impl sans arch) : f32_pointwise 912, cast 137, rope 245, softmax 218, swiglu 153, l2 148, split_q_gate 176 (~2 000 l.). Faire écrire `{stem}_{arch}` au générique puis supprimer les historiques. Vérif. : --all-dispatch identique.
3. `cli/main.rs` `sweep()` / `sweep_cuda()` : 13 blocs `if let Some(x) = …_dtype(dtype)` copiés deux fois (~340 l.) → une table (dtype → fn générique `<B>`). Risque faible ; vérif. : --all-dispatch.
4. (b) ci-dessus.

## (a) ★★ Variantes rendues inactives pendant le chantier
- gfx906 : `qmatmul_q4_k_mmvq_dp4a_t256` (R5, remplacé par R10b) : descripteur m_range (MAX,MAX) `impls/qmatmul.rs:36`, recette `ops/launch/qmatmul.rs:1987`, Dtype `Q4KDp4a` du sweep, certificat. Noyau `mmvq_q4_k_dp4a` **toujours vivant** (chemin F16, cf. e) → supprimer le descripteur, la recette et le certificat, garder le noyau jusqu'à (e).
- `qmatmul_q4_k_mmvq_dp4a` (r2), `qmatmul_iq4_xs_mmvq_dp4a`, `qmatmul_q5_k_mmvq_dp4a` : inactifs sur gfx906 mais **dispatchés sur sm_86** → garder ; retirer leurs descripteurs gfx906 « référence » (m_range MAX).
- Aucun cas cfg(unverified) : rien de nul, seulement remplacé. Commentaires périmés : `impls/qmatmul.rs:22,46,60`.
- Vérif. : std ; `every_cert_on_disk_is_registered` impose de retirer certificat et descripteur ensemble.

## (d) ★★ Variables d'environnement
- Choix de variante ou de comportement lus en dur (règle 1) : `FLAMBEAU_MOE_TILE8_DECODE` (`model-ops/src/moe_experts.rs:1319`, variante de noyau → ligne de dispatch), `FLAMBEAU_DECODE_GRAPH` (`forward/src/runtime/solo.rs:56`, graphe on/off → option CLI ou retrait), `FLAMBEAU_GEMMA4_BAIL_AFTER_LAYER` (`gemma4-v2/src/model.rs:80`, débogage qui tronque le modèle → retirer).
- Diagnostic seulement, via `dev_flag` : TRACE_BATCH, DUMP_PROMPT, DUMP_RAW_REQ, DEBUG_TOOL_RAW, LOAD_TRACE, GRAPH_TRACE → acceptables, à regrouper dans une option `--debug`.
- Exposées par la CLI (clap `env=`) : INFLIGHT_SLOTS, PREFILL_UBATCH, MTP, MAX_QUEUE_DEPTH, DECODE_BATCH_WINDOW_US, CTX_CAP, NO_GPU_SAMPLER, NO_BATCHED_DECODE, PREFIX_CACHE, PREFIX_CACHE_MAX_GB, KV, DEFAULT_SYSTEM, EMBEDDING_MAX_TOKENS → configuration, pas variante de noyau ; conformes.
- Notes du dispatch qui citent `FLAMBEAU_VARIANT=baseline` (lu nulle part) → périmées.

## (c) ★ Chemins historiques (avertissements du compilateur + recherche d'appelants)
- Compilateur : seulement `upload_q4_k` inutilisé (`forward/tests/common/mod.rs:72`) et `ncclComm_t` (style).
- `pub fn` sans aucun appelant (19) : backend-cuda nccl ×3, backend-hip `with_aux_stream`, `hipDeviceDisablePeerAccess` ; forward `table_slice_ptr`, `new_peer_buffer`, `per_layer_bytes` ; model-ops `embed_token_host`, `alloc_u64`, `forget_allocs` ; quant `metadata_u64`, `advise_drop_tensor` ; runtime `from_counts`, `rng_mut`, `sample_from_distribution` ; server-core `is_host`/`is_argmax`/`is_keep_on_device`.
- Pas de pile v1 restante (reste seulement le journal `forward_stack = "v2"`, `serve.rs:142`). Les 95 fichiers noyaux sont tous référencés à l'exécution. `crates/certs/parity/` : 3 JSON orphelins hors crate.
- Risque faible ; vérif. : check 3 configs.

## (g) ★ Test gemma4 en échec — `quant/tests/chat_template_gemma4_smoke.rs:151`
- Le gabarit du GGUF 26B-A4B ajoute `<|channel>thought\n<channel|>` après `<|turn>model\n` (raisonnement désactivé) ; l'assertion date du gabarit E4B.
- Proposition : comparer au rendu de llama.cpp du même GGUF (oracle, comme `chat_template_parity`) plutôt qu'à une chaîne figée ; à défaut, accepter les deux fins. Vérif. : test vert sur 26B et 31B.

## (f) ★ Commentaires narratifs — 379 lignes candidates (borne haute ; les renvois de format GGUF/llama.cpp légitimes restent)
- Par crate : quant 59, kernels-hip 55, kernels-shared 40, bench 39, forward 37, ops 37, server 32, backend-hip 26, model-ops 13, qwen35-v2 11, gemma4-v2 11, runtime 8, autres ≤ 4. Plus 11 notes dans `dispatch/hip/gfx906.toml`.
- 5 pires : `bench/src/sweep_mmvq.rs` 19, `forward/src/core/scratch.rs` 15, `server/src/api.rs` 12, `quant/src/tokenizer.rs` 10, `bench/src/sweep_mmq.rs` 10.

## Découpage en commits proposé (ordre)
1. (i) garde de compte d'arguments + correction des sweeps qu'elle révèle. 2. (h3) table dtype de la CLI. 3. (a) retrait des variantes inactives. 4. (d) trois variables codées en dur. 5. (c) code mort. 6. (g) test gemma4. 7. (b) ordonnanceur via `decode_tokens` (mesure Vibe). 8. (h2) sweeps historiques → générique. 9. (h1) recette dans le descripteur. 10. (e) F16 dans le dispatch + noyaux R10b `_f16` (mesure Vibe). 11. (f) commentaires, par crate.
