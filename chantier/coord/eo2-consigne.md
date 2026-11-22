# eo2 — contrôle de bin_O2 / bin_O2_CUDA / parity_O2 (HEAD 0b782475, branche e4-orphans, worktree ~/Workspace/flambeau-impl-o)

Règles habituelles : un seul processus GPU à la fois, garde MI50 ≤ 50 °C (jonction) et VRAM < 5 %, garde RTX ≤ 50 °C et < 1 Go, arrêt par SIGTERM. Ne rien modifier dans ~/Workspace/flambeau-impl-o (pas de git, pas de sweep lancé depuis ce répertoire). Vérifie d'abord `git -C ~/Workspace/flambeau-impl-o rev-parse HEAD` = 0b782475… et `cat impl/bin_O2.txt`. Reprends le protocole de eo/run_eo.sh et e4f/run_e4f.sh.

## A — passe complète (dispatch figé au commit)
- `git -C ~/Workspace/flambeau-impl-o show 0b782475:dispatch/hip/gfx906.toml > eo2/gfx906.toml`, idem `dispatch/cuda/sm_86.toml`.
- `sweep --all-dispatch` gfx906 (bin_O2, GPU 1) puis sm_86 (bin_O2_CUDA), sorties dans eo2/. **Attendu : 86/86 sur gfx906** (deux lignes MoE retirées par 12495baf), **20/20 sur sm_86**. Liste toute ligne non PASS.

## B — 20 sweeps MoE sm_86 (régénération des certificats)
- Depuis **~/Workspace/flambeau-exp/eo2/certs-run** (pas depuis la worktree), lance les 20 commandes sm_86 de coord/moe-sweep-commands-O2.md (bin_O2_CUDA, CUDA_VISIBLE_DEVICES=0, timeout 900, garde RTX avant chacune). Ne lance PAS les 36 « à part ».
- Rapporte PASS/FAIL et l'erreur max par sweep, et le nombre de .json dans eo2/certs-run/certs/cuda/sm_86/ (attendu 20).

## C — débits (slots 1, kv q8, lp.py, comme e4f)
| run | binaire | mtp | réf |
|---|---|---|---|
| Q1 | bin_O2 MI50 TP-2 | 0 | e4f/F1 29,55 |
| Q2 | bin_O2 MI50 TP-2 | 4 (debug, acceptation) | e4f/F2 39,43, 59,3 % |
| Q3 | bin_O2_CUDA RTX | 0 | e4f/F3 27,59 |
| Q4 | bin_O2_CUDA RTX | 4 | e4f/F4 43,27, 61,2 % |
Textes comparés aux lp.json de e4f (5 invites). Relève T départ / Tmax.

## D — parité
`impl/parity_O2 --nocapture --test-threads=1` comme e4f/C. Attendu : vert.

## Rendu
eo2/RESULTATS.md (format eo), puis écris « eo2 terminé ».
