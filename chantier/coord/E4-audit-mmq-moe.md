# E4 — audit d'usage MMQ / MoE / MMVQ batched (certs orphelins gfx906)

Base : arbre 16b3a845 (lu via `git archive`, la worktree bougeait pendant l'audit). Orphelin = cert sans ligne dans `dispatch/hip/gfx906.toml` : 236. Note : 34dd4eeb (postérieur) a déjà retiré les 8 descripteurs dormants de A2 et leurs certs `_gfx906`.
Méthode : impl_id → stem (sweep `kernel_stem()` / `QMATMUL_RECIPES`) → lanceur `crates/ops/src/launch/*` → appelant runtime (model-ops/forward/models). `qmatmul()` est générique `B` : ce qui y est codé en dur tourne sur HIP **et** CUDA.
Les certs neutres (sans `_gfx906`) ont presque tous un jumeau `certs/cuda/sm_86/` (A 28, Bq 40, Bv 25, C 22) : supprimer le cert HIP casse la paire de bit-match.

## A — MMQ dense non dispatché (40 = 32 BENCH_REFERENCE + 8 descripteurs dormants `m_range=(MAX,MAX)`)
- **A1, stem lancé au runtime (16)** : `qmatmul_{q4_0,q4_1,q5_0,q5_1,q2_k,q8_k,iq2_xxs,iq2_xs,iq2_s,iq3_xxs,iq3_s,iq1_s,iq1_m}_mmq_wave64`, `qmatmul_q4_{0,1}_mmq_4warp_lds`, `qmatmul_q8_0_mmq_oracle`. Même stem que les lignes dispatchées `*_gfx906` (impls/qmatmul.rs:367-676, m≥32 ; oracle m 4..31 l.581). Cert neutre cross-backend (sweep_mmvq.rs:2205-2249, `MmqGemmDtype`) en doublon du cert dispatché. → **KEEP comme oracle cross-backend** (bit-match sm_86).
- **A2, wave64 dormants (16 certs, 8 stems)** : `qmatmul_{q4_K,q5_K,q6_K,q3_K,iq4_nl,iq4_xs}_mmq_wave64[_gfx906]`, `qmatmul_q8_0_mmq_wave64_tile16[_gfx906]`, `qmatmul_q4_k_mmq_lds_tiled` + `qmatmul_q4_K_mmq_lds_tiled_gfx906`. Descripteurs `(MAX,MAX)` (impls/qmatmul.rs:96,278,354,467,480,516,569,616), donc jamais choisis par `B::dispatch_qmatmul` (launch/qmatmul.rs:278) ; l'override `gfx906.local.toml` n'est chargé nulle part (non établi ailleurs que par absence de grep). Seuls appelants : sweep_mmq.rs (Dtype `*Wave64`, `Q4KLdsTiled`), sweep_mmvq `MmqGemmDtype`, pmc_refresh (cli/main.rs:1865, 2006/2020). → les `_gfx906` : **CANDIDATS À SUPPRESSION** (déjà fait par 34dd4eeb). Les neutres : **KEEP comme oracle cross-backend** tant que le jumeau sm_86 est voulu. Sinon cert + entrée BENCH_REFERENCE + Dtype, puis kernel (instantiations.toml:1199,1281-1410,1618) supprimé ou `cfg(unverified)` + `why:` (règle 10).
- **A3, 4warp et q8_0 wave64 non tilé (8 certs, 4 stems)** : `qmatmul_{q4_K,q6_K,q8_0}_mmq_4warp_lds_gfx906`, `qmatmul_{q4_k,q6_k,q8_0}_mmq_4warp`, `qmatmul_q8_0_mmq_wave64[_gfx906]`. Stems `mmq_{q8_0,q4_K,q6_K}_4warp`, `mmq_q8_0_wave64` : pas de recette, pas de lanceur. Seulement chargés par `KERNEL_STEMS` (ops/registry.rs:166-211, chargement sans lancement). Appelants : sweep_mmq (`Q*4Warp`, `Q8_0Wave64`, l.237-245), pmc_refresh (cli/main.rs:1985,1992). → **CANDIDAT À SUPPRESSION** : ce sont des baselines A/B d'un placeholder dépassé (direct_call.rs:244-258). Garder seulement si on veut une A/B vivante. Kernel : règle 10.

## B — MoE (89)
- **Bq tile8 (58 = 38 neutres BR + 20 `_gfx906` DIRECT_CALL)**. Neutre et `_gfx906` certifient le même stem `indexed_moe_mmq_<d>_<gate_up|down>_tile8_dp4a` (sweep_mmvq.rs:3299-3340 ; sweep_moe.rs:3175-3200).
  - Lancés au runtime : 31 stems via moe_experts.rs (gate_up l.1378-1686, down l.1747-1942). Seuils codés en dur : `TILE8_PAIRS_MIN=8` (l.1186), `q4_0_use_tile8`/`q8_0_use_tile8` sur le dtype et `n_pairs` (l.1323-1330). → **KEEP** (44 certs, ex. `indexed_moe_mmq_q4_k_gate_up_tile8`, `..._iq3_s_down_tile8`, `..._q8_0_down_tile8_gfx906`).
  - Non appelés : `q4_1_gate_up`, `q5_0` ×2, `q5_1` ×2 (lanceurs launch/moe.rs:2181-2417 sans appelant) et `q2_k` ×2 (aucun lanceur). Seulement sweep_moe/sweep_mmvq/CLI (cli/main.rs:931-984). 14 certs. → **CANDIDAT À SUPPRESSION** (cert + entrée DC/BR + Dtype + lanceur ; kernel : règle 10). Alternative : les brancher, puisque la portée CLAUDE.md couvre Q4_1/Q5_x/Q2_K.
  - Bug de métadonnées : les 38 certs neutres portent `dtype_weight: "Q8_0"`, codé en dur à sweep_mmvq.rs:3477.
- **Bq lds_tiled Q4_K (2)** : `indexed_moe_mmq_q4_k_{gate_up,down}_lds_tiled`. Lanceurs launch/moe.rs:2558/2625 et méthodes ops_trait.rs:1089/1095 sans appelant runtime. Sweep seul (sweep_mmvq.rs:4288). → **CANDIDAT À SUPPRESSION** (+ méthodes trait/impl ; kernel : règle 10).
- **Bv indexed_moe_mmvq (29 = 26 BR + 3 DC)**
  - Stem lancé au runtime (21) : `indexed_moe_mmvq_{q8_0,q4_0,q4_1}`, `_q4_k_r2` (r2_dp4a si n_tokens<32), `_q4_k_r4` (r4_dp4a si ≥32, launch/moe.rs:126), `_{q3_k,q5_k,q6_k}_r2`, 9 `_iq*_r2`, `_{q8_0,q4_0}_gate_up`, `_q4_k_gate_up_r4` (launch/moe.rs:2934), `_q5_k_gfx906`. Appels dans moe_experts.rs:528-834 et 1976-2154. → **KEEP** (oracles cross-backend des stems runtime).
  - Non lancés (8) :
    - `_q5_0`, `_q5_1` : lanceurs launch/moe.rs:729/767 sans appelant.
    - `_q2_k_r2`, `_q2_k_gfx906` : pas de bras Q2_K dans moe_experts.rs:695/860.
    - `_q4_k_gate_up` : stem non-r4, le runtime prend r4_dp4a.
    - `_q4_k_gate_up_r4_sorted` : lanceur l.2878 sans appelant.
    - `_q4_k_gfx906` : stem HIP `indexed_moe_mmvq_q4_k`, sans lanceur.
    - `_q3_k_gfx906` : sweepe le stem HIP `indexed_moe_mmvq_q3_k` (sweep_moe.rs:2251), pas le stem runtime `q3_k_r2_dp4a`.
    → `_q4_k_gfx906` : **KEEP comme oracle** single-row (A/B documenté, direct_call.rs:236). Les 7 autres : **CANDIDATS À SUPPRESSION** (kernels `.cu` HIP-only `indexed_moe_mmvq_{q2_k,q3_k}.cu` : règle 10).
  - Hors périmètre mais grave (règle 2) : les lignes dispatchées `indexed_moe_mmvq_q4_k_gate_up_gfx906` et `_q4_k_r2_gfx906` certifient les stems HIP `indexed_moe_mmvq_q4_k_gate_up` / `_q4_k_r2` (sweep_moe.rs:1175/418), pas les stems lancés (`gate_up_r4_dp4a`, `q4_k_r{2,4}_dp4a`).

## C — MMVQ batched / gate+up (22 qmatmul, tous en BENCH_REFERENCE ; les 6 `*_batched_slots`/attention/gemv sont hors périmètre)
- **Runtime (14)**, sélection codée en dur dans `qmatmul()` (launch/qmatmul.rs) :
  - Q4_1 m 2..4 → `q4_1_mmvq_batched` (l.61) ; Q8_0 m 2..4 → `q8_0_mmvq_row_tile_batched` (l.74) ; Q4_0 m 2..4 → `q4_0_mmvq_row_tile_batched` (l.124).
  - `rowtile_batched_k_owns` (m 2..8, k%256, l.480) → 7 `*_mmvq_dp4a_row_tile_batched` (q4_k, q5_k, q6_k, iq4_xs, iq3_s, iq3_xxs, iq2_s).
  - Gate+up : `q8_0_gate_up_t128_vdr2` et `q4_0_gate_up_t128_dp4a` (shared_expert.rs:332/339, delta_net.rs:690/696, dense_ffn.rs:100/122) ; `q5_k_gate_up_dp4a` (shared_expert.rs:345) ; `q4_0_gate_up_row_tile_mmvq_batched` (delta_net.rs:1020, n_slots 2..4).
  - → **KEEP**. Ils sont **mal classés** en BENCH_REFERENCE (direct_call.rs:219-221 dit « non invoqués » ; l.296-302 dit le contraire) : à promouvoir en DIRECT_CALL avec une ligne TOML.
- **Non runtime (8)** : `qmatmul_{q8_0,q4_0,q4_k,q6_k}_mmvq_batched`, `q5_k_r2_mmvq_batched`, `q5_k_mmvq_row_tile_batched` (lanceurs launch/qmatmul.rs:408-795 sans appelant), `q4_0_gate_up_dp4a`, `q4_1_gate_up_dp4a` (méthodes ops_trait.rs:100/106 jamais appelées).
  - `q8_0`/`q4_0`/`q5_k_r2` batched servent de baseline aux tests de parité des row-tile runtime (backend-hip/tests/mmvq_*_row_tile_batched.rs:118-119) → **KEEP comme oracle**.
  - Les 5 autres (q4_k/q6_k batched, q5_k row_tile, gate_up q4_0/q4_1 dp4a) n'ont que des auto-tests → **CANDIDATS À SUPPRESSION** (+ lanceur, méthode trait, test ; kernel : règle 10). Commentaire périmé à delta_net.rs:1087 (cite `mmvq_q8_0_batched`, le runtime prend row_tile).

## Synthèse
| Famille | N | Proposition |
|---|---|---|
| A1 wave64/4warp_lds/oracle neutres (stem runtime) | 16 | KEEP oracle cross-backend |
| A2 wave64/lds_tiled dormants (`_gfx906` + neutres) | 16 | `_gfx906` supprimés (34dd4eeb) ; neutres KEEP si jumeau sm_86 voulu |
| A3 4warp + q8_0 wave64 non tilé | 8 | CANDIDAT SUPPRESSION (règle 10) |
| Bq tile8 lancés | 44 | KEEP |
| Bq tile8 q4_1-gu/q5_0/q5_1/q2_k | 14 | CANDIDAT SUPPRESSION (ou câbler) |
| Bq lds_tiled Q4_K | 2 | CANDIDAT SUPPRESSION |
| Bv stems runtime | 21 | KEEP |
| Bv q4_k_gfx906 single-row | 1 | KEEP oracle |
| Bv q5_0/q5_1/q2_k×2/q4_k_gate_up/r4_sorted/q3_k_gfx906 | 7 | CANDIDAT SUPPRESSION |
| C runtime (codé en dur) | 14 | KEEP + reclasser en DIRECT_CALL |
| C baselines de parité | 3 | KEEP oracle |
| C sans appelant | 5 | CANDIDAT SUPPRESSION |
