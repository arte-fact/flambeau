# V1 : séquence de validation au feu vert (Opus ; préparée, rien d'exécuté)

Branche `mtp-gdn-kstates` e6d1f1ea dans ~/Workspace/flambeau-impl-o. Elle contient, sur f5d29f30 (serie-7) :
- mtp0-noload 979ec082 ;
- (a) cbcb01f1, 83210609, 0353c9fb ;
- D4 c193cafc ;
- GDN 54aeb95d, e6d1f1ea.

`FE=~/Workspace/flambeau-exp`, `SP=<scratchpad>`, `B=$FE/target-o/release/flambeau`, `BC=$FE/target-cuda/release/flambeau`. Un seul build lourd à la fois, `-j 8`.

Gardes GPU :
- MI50 : GPU 1 seulement, `HIP_VISIBLE_DEVICES=1`, avec `wait_gpu1.sh` (jonction ≤ 50 °C, VRAM < 5 %) avant chaque lancement ;
- RTX : < 1 Go utilisé et ≤ 50 °C ;
- tout processus GPU arrêté avant chaque rapport. sd-server et llama-monitor ne sont pas touchés.

## 0. Avant le build (sans GPU)
1. `sha256sum $B` doit être égal à bin_S7 (b2636898…). Si c'est le cas, on refait `$SP/hsaco_before/` à partir de `$(ls -td $FE/target-o/release/build/*kernels-hip*/out | head -1)/*.hsaco`. C'est la référence gfx906 de serie-7, soit 286 fichiers attendus.
2. On ajoute les 4 entrées `DirectCallKernel` : `embed_gather_q4_k_f16` et `gdn_state_step_alphabeta_f32_kstates`, dans `backend-hip/src/impls/direct_call.rs` (certs/hip/gfx906/…) et dans `backend-cuda/src/impls/direct_call.rs` (certs/cuda/sm_86/…). Elles ne sont pas commitées tant que les JSON n'existent pas, puisque les tests du catalogue exigent le fichier.

## 1. Builds (l'un après l'autre)
```bash
cd ~/Workspace/flambeau-impl-o
CARGO_TARGET_DIR=$FE/target-o    nice cargo build -j 8 --release -p flambeau-cli --features hip_serve,hip_sweep
CARGO_TARGET_DIR=$FE/target-cuda nice cargo build -j 8 --release -p flambeau-cli --features hip_serve,hip_sweep,cuda_serve,cuda_sweep
```
Contrôle hsaco : on compare `$SP/hsaco_before/*.hsaco` au nouveau `out/` avec `cmp`. Attendu :
- 0 fichier différent sur 286 ;
- 2 fichiers en plus : `embed_gather_q4_k_f16.hsaco` et `gdn_state_step_alphabeta_f32_kstates.hsaco`.

`gdn_state_step_alphabeta_f32{,_batched_slots}` doivent rester identiques, puisque leur en-tête a été refactoré mais que l'ISA a déjà été vérifiée identique.

## 2. Tests
1. **Sans GPU** : `WT=~/Workspace/flambeau-impl-o $SP/std2.sh v1` (workspace, `--all-features`, GPU masqués). On fait `comm -13 $SP/failed_o2.txt $SP/failed_v1.txt`, ce qui doit être vide : les échecs attendus sont seulement ceux qui demandent un GPU. Il faut aussi que `qwen35-v2 mtp_head_loads_only_when_requested` passe.
2. **GPU CUDA** (`CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=-1`, `--release --features hip,cuda`, `-- --test-threads=1`) :
   - forward : `embed_q4k`, `mtp_draft_chain` et `mtp_gdn_kstates` (filtre `_cuda`) ;
   - `parity_snapshot_dense_cuda` ;
   - bench : `FLAMBEAU_TEST_Q4K_GGUF=~/models/Qwen3.8-27B-UD-Q4_K_M.gguf --test embed_gather_q4k_real_gguf cuda_` ;
   - ops : `cuda_registry_smoke`, qui doit lister 2 stems chargés de plus et 0 nouveau « skipped ».
3. **GPU HIP**, après `wait_gpu1.sh` (`HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=`) :
   - les mêmes trois tests forward (filtre `_hip`) ;
   - `synth_gdn`, `synth_gdn_mixed`, `parity_snapshot_gdn` et `snapshot_restore_parity` (GDN et snapshot touchés), ainsi que les autres `synth_*` / `parity_*` ;
   - bench `embed_gather_q4k_real_gguf hip_`.
4. **Croisé** (les deux GPU visibles) : `parity_moe_q4k_cross_backend`, plus `qwen35-v2 real_gguf_cuda_parity` si son GGUF est présent.

## 3. Sweeps et certs
1. **Nouveaux ops**, écrits dans le dépôt (`--out .`) :
   - `$B sweep --arch gfx906 --op embed_gather_q4_k_f16` et `--op gdn_state_step_alphabeta_f32_kstates` ;
   - `$BC sweep --arch sm_86 --op …`, mêmes ops.

   Attendu : `pass=true` partout. Avec 0 ULP de tolérance, `max_rel_err` (le nombre d'éléments différents) vaut 0 sur chaque forme.
2. **Contrôle du refactor GDN**, sorties dans `$SP/v1_sw/` (les certs commités ne sont pas réécrits) : `--op gdn_state_step_alphabeta_f32` et `--op gdn_state_step_alphabeta_f32_batched_slots` sur les deux arch, plus `gdn_state_step_alphabeta` sur gfx906. On compare les `results` aux JSON commités, avec les mêmes `max_rel_err`.
3. **all-dispatch** : `$B sweep --arch gfx906 --all-dispatch --out $SP/v1_ad_gfx906`, puis `$BC sweep --arch sm_86 --all-dispatch --out $SP/v1_ad_sm86`. Attendu : verdict vert pour chaque impl, dispatch inchangé.
4. **cert-check** : `$BC cert-check` (gfx906) et `$BC cert-check --backend cuda --arch sm_86`. Attendu : **86/0** et **33/0**, puisque les deux nouveaux noyaux sont des appels directs et n'ajoutent pas de ligne de dispatch.
5. **Fichiers de cert** : gfx906 passe de 273 à **275**, sm_86 de 217 à **219**. Les tests `impls::tests` des deux backends passent avec les 4 nouvelles entrées.
6. **Commit unique** `certs: embed_gather_q4_k_f16 + gdn_state_step_alphabeta_f32_kstates (gfx906, sm_86)`, contenant les 4 JSON et les 2 fichiers `direct_call.rs`.

## 4. Binaires
- `cp $B $FE/impl/bin_V1` et `cp $BC $FE/impl/bin_V1_CUDA`.
- `$FE/impl/bin_V1.txt` contient :
  - la branche et le hash du commit des certs ;
  - la liste des commits depuis f5d29f30 ;
  - les features (target-o : hip_serve,hip_sweep ; target-cuda : les 4) ;
  - le sha256 des deux binaires ;
  - le hsaco (0 différent sur 286, plus 2) ;
  - cert-check 86/0 et 33/0 ;
  - les fichiers de cert, 275 et 219 ;
  - le résultat des tests : nombre de tests passés, plus les nouveaux échecs, qui doivent être 0.
- Le Vibe mesure ensuite : nvidia-smi en mtp 0 et mtp 4 (avant = bin_S7_CUDA, après = bin_V1_CUDA), les débits, les textes (Qwen3.8 sur les deux arch, gemma-4-12b-it-Q4_0 et ornith-1.0-35b-Q4_K_M).

## 5. nsys d'un cycle MTP-4 (Opus, après les mesures du Vibe, RTX)
- **flambeau** :
  - `nsys launch --trace=cuda,osrt --session-new=v1fl $FE/impl/bin_V1_CUDA serve …` avec la même ligne que rtx/RESULTATS (`--ctx-cap 8192 --kv q8 --mtp 4`) ;
  - une requête de chauffe, puis `nsys start --session=v1fl -o $SP/nsys_v1/fl_mtp4` ;
  - une requête `lp_timings` (400 tokens, glouton), puis `nsys stop`.
- **llama.cpp** : même procédure avec `build-cuda-hd/bin/llama-server`, la ligne D2 de coord/dflash-base-commandes.md (`--spec-type draft-mtp --spec-draft-n-max 4`).
- **Analyse** :
  - `nsys stats` (cuda_gpu_kern_sum, cuda_api_sum) et `$SP/ktimes.py` ;
  - un cycle est découpé entre deux synchros de flux consécutives, sur 50 cycles en régime établi ;
  - pour chaque cycle, on attribue le temps GPU actif, les trous côté hôte entre noyaux (lancements), les synchros, et la copie GDN : elle doit être absente en V1, on la mesure aussi sur bin_S7_CUDA pour la référence.
- **Rendu** dans coord/mtp-ecart-nsys.md, puis l'ordre des corrections restantes (V5).
