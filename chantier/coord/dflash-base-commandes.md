# DFlash — base mesurée llama.cpp : commandes pour le Vibe (RTX et MI50, 3 modes)

## 0. Constat sur les binaires (vérifié sans compiler, sans GPU)
- Les binaires construits datent du 10/11 : `build-cuda` (sm_86, `CMAKE_CUDA_ARCHITECTURES=86`) et `build-hip` (gfx906, `AMDGPU_TARGETS=gfx906`).
- `llama-server --help` liste `--spec-type none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,draft-dspark,ngram-…` : **pas de `draft-dflash2`**.
- `libllama.so` contient bien le graphe dflash2 (`selector_hidden`, 26 et 21 symboles dflash2), mais `libllama-common.so` (le parcours spéculatif côté hôte) ne connaît que `draft-dflash`.
- Les sources (`~/llama.cpp`, ffc87c3a7, du 18/11) ont un type séparé `draft-dflash2` (`common/speculative.cpp:79`), avec le parcours du treillis du sélecteur (`:1600-1648`).
- **Décision (Fable, provisoire, Benoît peut revenir dessus) : option (B)**. On mesure ce qu'on implémentera : `draft-dflash2` avec le sélecteur, sur un build HEAD. Les builds existants restent intacts. Les options étaient :
  - **(A)** mesurer tout de suite avec les binaires existants : `--spec-type draft-dflash` sur le GGUF DFlash2. Le graphe dflash2 se charge peut-être, mais le sélecteur n'est sans doute pas parcouru, à vérifier dans le log : acceptation, « dflash2 », « selector ». Ce serait alors une base « DFlash, argmax par position », pas DFlash2.
  - **(B)** construire le HEAD dans de nouveaux répertoires, sans toucher aux binaires existants, puis utiliser `draft-dflash2` :
    - `cmake -B build-cuda-hd -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build-cuda-hd -j --target llama-server`
    - `cmake -B build-hip-hd -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 && cmake --build build-hip-hd -j --target llama-server`
    - Les trois modes doivent alors tourner **sur le même build** HD, pour que la comparaison reste juste.

## 1. Options communes
- Reprises de `rtx/run_rtx_llama.sh` (RTX) et de `analyse-ecarts/lcserve.sh` (MI50) : KV q8 (`-ctk q8_0 -ctv q8_0`), slots 1 (`-np 1`), `-c 32768 -b 512 -ub 512 --flash-attn on -ngl 999 --no-warmup`, port 8099.
- Pas de pénalité : llama-server a par défaut `penalty_repeat = 1.0` (coord/s1-penalite.md §3), et `rtx/lp_timings.py` (glouton, 5 invites, 400 tokens, `cache_prompt` false, réflexion désactivée) n'en passe pas.
- Mesure : `python3 rtx/lp_timings.py 8099 <out.json>` (predicted_per_second par invite), et le log serveur pour l'acceptation (`draft acceptance`, `n_draft`, `n_accept`).
- Gardes : RTX < 1 Go utilisé et ≤ 50 °C avant chaque serveur ; MI50 avec la garde jonction ≤ 50 °C et VRAM < 5 % des deux GPU (TP-2), un serveur à la fois, SIGTERM.

## 2. RTX 3090 (sm_86)
```bash
M=$HOME/models/Qwen3.8-27B-UD-Q4_K_M.gguf
D=$HOME/models/Qwen3.8-27B-DFlash2-Q8_0.gguf
B=$HOME/llama.cpp/build-cuda-hd       # build HEAD (option B), pour les 3 modes
LS=$B/bin/llama-server; export LD_LIBRARY_PATH=$B/bin:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0
BASE="-m $M -ngl 999 -c 32768 --flash-attn on -ctk q8_0 -ctv q8_0 -b 512 -ub 512 -np 1 --port 8099 --no-warmup"
# D1 sans spec
$LS $BASE
# D2 draft-mtp n 4 (même ligne que rtx/run_rtx_llama.sh C2)
$LS $BASE --spec-type draft-mtp --spec-draft-n-max 4 -ctkd q8_0 -ctvd q8_0
# D3 DFlash2 n_max 7, sélecteur compris
$LS $BASE -md $D -ngld 999 --spec-type draft-dflash2 --spec-draft-n-max 7
```
- KV du drafter : laissé en F16 par défaut. L'étude note que le KV FP8 n'est pas géré pour cette arch, et ce KV pèse ≈ 42 Mo.
- Relever `nvidia-smi --query-gpu=memory.used` une fois chaque serveur prêt. Cela fournit aussi la mesure d'occupation RTX qui manque à coord/dflash-inventaire-flambeau.md §5.

## 3. MI50 × 2 (gfx906, `-sm layer -ts 1/1`)
```bash
B=$HOME/llama.cpp/build-hip-hd        # build HEAD (option B), pour les 3 modes
LS=$B/bin/llama-server; export LD_LIBRARY_PATH=$B/bin:$LD_LIBRARY_PATH HIP_VISIBLE_DEVICES=0,1
BASE="-m $M -ngl 999 -c 32768 --flash-attn on -ctk q8_0 -ctv q8_0 -b 512 -ub 512 -np 1 -sm layer -ts 1/1 --port 8099 --no-warmup"
# E1 sans spec
$LS $BASE
# E2 draft-mtp n 4 (référence bench_lc.sh)
$LS $BASE --spec-type draft-mtp --spec-draft-n-max 4 -ctkd q8_0 -ctvd q8_0
# E3 DFlash2 n_max 7 (le drafter va sur le device principal par défaut ; garder ce défaut, et le noter si le log en indique un autre)
$LS $BASE -md $D -ngld 999 --spec-type draft-dflash2 --spec-draft-n-max 7
```

## 4. Rapport attendu (par machine)
- t/s par invite et moyenne pour les 3 modes ;
- acceptation et longueur moyenne acceptée pour D2/E2 et D3/E3, à comparer aux planchers des tests nocturnes (≥ 0,518 et ≥ 4,54) ;
- mémoire GPU, et binaire HEAD utilisé (ffc87c3a7, build-*-hd) ;
- avant les runs, vérifier que `llama-server --help` liste `draft-dflash2`.
