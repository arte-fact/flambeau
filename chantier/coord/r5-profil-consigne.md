# r5 — étape 5 : profil CUDA de flambeau et de llama.cpp sur la RTX 3090 (sans MTP)

Objectif : attribuer l'écart de 27,5 contre 43,5 t/s noyau par noyau. Mesure seulement, rien à modifier.
Binaires : flambeau `impl/bin_O2_CUDA` (HEAD a45374a0 poussé) ; llama.cpp `~/llama.cpp/build-cuda/bin/llama-server` (commande BASE de rtx/run_rtx_llama.sh, port 8099). Flambeau : la ligne de rtx/run_rtx_fl.sh avec `--mtp 0` (port 8097).
Règles : RTX seule (MI50 inutilisées), garde ≤ 50 °C et < 1 Go avant chaque serveur, un serveur à la fois, SIGTERM. Relever avant/après chaque run : temp, power.draw, clocks.gr, clocks.mem (nvidia-smi).

## A — références sans profileur, à froid
- L1 : llama.cpp, mtp 0, lp.py (comme rtx/C1). Réf : 43,49 t/s.
- F1 : flambeau bin_O2_CUDA, mtp 0, lp.py. Réf : 27,48 t/s.

## B — traces nsys, invite « math », 200 tokens, non streamé
Pour chacun des deux serveurs (F puis L), avec garde ≤ 50 °C :
1. `nsys launch --session-new=<fl|lc> --trace=cuda,nvtx --cuda-graph-trace=node <commande serveur>` ;
2. attendre /health, faire UNE requête de chauffe de 16 tokens ;
3. `nsys start --session=<fl|lc> -o r5/<fl|lc>` ; une requête « math » de 200 tokens (même prompt pour les deux, chat template du serveur, temperature 0) ; `nsys stop --session=<fl|lc>` ; noter le nombre de tokens générés et la durée côté client ;
4. arrêter le serveur.
Puis `nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_trace --format csv -o r5/<fl|lc> r5/<fl|lc>.nsys-rep`.
Si `--cuda-graph-trace=node` échoue, refais avec `graph` et signale-le.

## C — synthèse brute (sans interprétation)
Dans r5/RESULTATS.md : A (t/s par invite, moyennes, horloges) ; B pour chaque serveur : tokens générés, durée client, temps GPU total des noyaux, nombre de lancements, top 25 noyaux (nom, appels, temps total, moyenne) ; chemins des .nsys-rep et des CSV. Puis écris « r5 terminé ».
