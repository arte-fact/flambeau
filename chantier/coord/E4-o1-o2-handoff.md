# (o1)/(o2) — passation (interrompu : limite d'usage atteinte)

Worktree : ~/Workspace/flambeau-impl-o, branche e4-orphans, base 16b3a845, AUCUNE modification encore. HEAD 16b3a845 de flambeau-impl intact.

## (o1) inventaire fait (lecture)
12 descripteurs inactifs gfx906 ; chacun a une ligne propre dans QMATMUL_RECIPES (aucun alias partagé).
- Noyaux à GARDER (dispatchés sm_86) : mmvq_q4_k_r2_dp4a (qmatmul_q4_k_mmvq_dp4a), mmvq_q5_k_dp4a, mmvq_iq4_xs_dp4a → retirer seulement descripteur gfx906 + cert gfx906 ; la recette reste (sert sm_86).
- mmvq_q5_k_r2 (qmatmul_q5_K_mmvq_nw1_r2_gfx906) : oracle nw1_r2 → consigne « garder les oracles r2 » : à trancher (descripteur inactif ET oracle).
- 8 MMQ (iq4_nl/iq4_xs/q3_K/q4_K/q5_K/q6_K wave64, q4_K lds_tiled, q8_0 wave64_tile16) : Dtype dans bench/sweep_mmq.rs (MmqDtype) ET bench/sweep_mmvq.rs (MmqGemmDtype) ; CLI cli/src/main.rs:1133-1247 et 1526-1541 ; pmc_probe.rs pour q4/q5/q6_K wave64 + tile16 ; instantiations.toml. Noyaux à vérifier : référencés par sweeps/pmc seulement → supprimables si aucun autre usage (registry.rs à relire).
- Reste à faire : 18 certs *_gfx906 de certs/cuda/sm_86 + retrait des Dtype IQ* des listes CUDA de sweep_cuda().

## (o2) rien commencé ; audit E4-audit-mmq-moe.md non écrit.
