# r10s — levier 2 : fusion gate+up (SwiGLU) sur sm_86 (plan, rien codé ; base r10b-sm86 1b31581f)

## 0. Ce que la fusion peut encore rapporter après R10s
- llama.cpp `mul_mat_vec_q<type, 1, has_fusion=true>` fait gate et up dans un seul noyau : deux accumulateurs, réduction commune, épilogue `up * silu(gate)` écrit en F32 (mmvq.cu:830-860).
- Chiffre r5 : IQ4_XS 17408 lignes, 113 µs pour gate + up fusionnés. Après R10s, un appel IQ4_XS non fusionné coûte 56,5 µs, donc gate puis up font 113 µs : **le trafic des poids est déjà le même**.
- Ce qu'il reste à gagner : le noyau `swiglu_f32_to_f16` (trace r5 : 12 800 appels, 2,5 µs en moyenne, 32 ms sur 9 068 ms de noyaux, soit 0,35 %), un lancement, l'écriture et la relecture de gate/up en F32 (2 × 17 408 × 4 o par couche), et la seconde lecture du vecteur Q8_1 (5 Ko, en L2).
- **Estimation : de l'ordre de 1 % du temps GPU.** C'est une estimation non mesurée. À confirmer par un microbanc avant de coder plus d'un type (§5).
- Pour comparaison, sur la même trace, `rmsnorm_q8_1_fused` coûte 165 ms (1,8 %) : 129 appels par token à 6,4 µs, lancés sur une grille de 1 × 1024 pour 5120 éléments. C'est un levier voisin, sans doute plus rentable.

## 1. Types et formes
- Uniquement le FFN dense des 64 couches : gate et up ont la forme n = 17408, k = 5120 ; m = 1 (décodage). L'entrée est le Q8_1 de `rmsnorm_q8_1_fused`, la sortie est le `gated` consommé par `ffn_down`.
- Comme llama.cpp (`ggml_cuda_should_fuse_mul_mat` exige `ffn_up.type == ffn_gate.type`), on ne fusionne que les couches dont gate et up ont **le même type**. Dans Qwen3.8-27B-UD-Q4_K_M, c'est le cas de 36 couches sur 64 :

| type | couches | priorité |
|---|---|---|
| IQ4_XS | 20 | 1 |
| Q5_K | 9 | 2 |
| Q4_K | 5 | 3 |
| Q3_K | 1 | — (non fait sauf demande) |
| IQ4_NL | 1 | — |

- Les 28 couches à types mélangés (IQ4_XS/Q4_K 13, IQ4_XS/Q5_K 3, Q3_K/IQ4_XS 3, Q4_K/Q5_K 3, …) restent sur deux MMVQ séparés. Un noyau à deux types `<TG, TU>` multiplierait les instanciations pour un gain inférieur à 1 % : écarté.

## 2. Noyau : réutiliser les corps R10s, un seul gabarit de fusion
- **Corps existants** : `mmvq_iq4_xs_vdr4_row`, `mmvq_q5_k_vdr2_row`, `mmvq_q4_k_vdr2_row` (et `mmvq_q3_k_vdr2_row`, `mmvq_iq4_nl_vdr2_row`). On extrait de chacun la contribution d'un super-bloc en fonction `__device__` « partielle » (`iq4_xs_vdr4_sb(bk, yb, sub) -> float`, etc.). Le noyau une ligne actuel l'appelle tel quel : même code, mêmes certificats.
- **Nouveau gabarit neutre** : `mmvq_gate_up_swiglu_row<Partial, OutT, THREADS>` (kernels-shared, `mmvq_gate_up.cuh`).
  - Une ligne par bloc, 128 threads, la même boucle de super-blocs.
  - Deux accumulateurs (`acc_g` sur `W_gate`, `acc_u` sur `W_up`, même `yb`).
  - Deux `mmvq_block_reduce`, puis un thread écrit `silu(acc_g) * acc_u`.
- **Sortie F16 et non F32** : l'épilogue reprend exactement la formule de `swiglu_f32_to_f16`.
  - Comme `acc_g` et `acc_u` suivent le même ordre de réduction que les deux appels séparés, le `gated` F16 doit être **identique au bit près** au chemin non fusionné.
  - `quantize_f16_to_q8_1` et `ffn_down` restent inchangés.
  - Une sortie F32 suivie de `quantize_q8_1(F32)` économiserait aussi l'arrondi F16, mais changerait les résultats : écarté.
- **Cellules** : `mmvq_<t>_gate_up_swiglu_t128_vdrN` (une entrée, sortie F16), pour iq4_xs, q5_k et q4_k.

## 3. Place dans le dispatch (règles 1 et 3)
- Les fusions actuelles sont des méthodes par type (`mmvq_q8_0_gate_up`, `mmvq_q4_0_gate_up_t128`, `mmvq_q5_k_gate_up`), choisies par des `if dtype ==` dans `dense_ffn.rs` : c'est la forme « trait parallèle » que la règle 3 proscrit. On ne l'étend pas.
- **Un seul contrat** : `Ops::mmvq_gate_up_swiglu_f16(bufs, shape, dtype)` et `Ops::mmvq_gate_up_swiglu_supported(dtype)`, avec une implémentation réelle sur les deux backends (même forme que `mmvq_f16_supported` de E4e).
  - La résolution passe par une nouvelle table de dispatch `[[qmatmul_gate_up]]` (dtype, `m = "1"`, impl, cert), lue comme `[[qmatmul]]` et mise en cache par dtype dans `Registry` (`DtypeCache`, comme `mmvq_m1`).
  - `dispatch/cuda/sm_86.toml` reçoit les 3 lignes. `dispatch/hip/gfx906.toml` n'en reçoit aucune : `supported` y est faux.
- **`dense_ffn.rs`** :
  - si `n == 1`, `gate.dtype == up.dtype` et `ops.mmvq_gate_up_swiglu_supported(dtype)`, un seul appel écrit `gated_f16`, puis `quantize_f16_to_q8_1` comme aujourd'hui ;
  - sinon, chemin actuel inchangé.
  - Pas de test d'architecture (règle 13) : le choix vient du dispatch.
- Les fusions codées en dur Q4_0/Q8_0 (gate et up écrits séparément en F32) restent telles quelles ; les migrer vers la table serait une étape à part.
- **gfx906** : aucune ligne, donc le chemin ne change pas. Les nouvelles cellules ajoutent des `.hsaco` inutilisés, sans toucher aux existants (contrôle par `cmp`, comme pour R10s).

## 4. Certificats (règle 2)
- Un certificat sm_86 par impl : `qmatmul_iq4_xs_gate_up_swiglu_t128_vdr4`, `qmatmul_q5_k_gate_up_swiglu_t128_vdr2`, `qmatmul_q4_k_gate_up_swiglu_t128_vdr2`.
- Sweep `run_gate_up_swiglu_sweep` dans sweep_mmvq :
  - deux poids aléatoires du même type et un vecteur Q8_1 ;
  - référence hôte `silu(W_g·y) * (W_u·y)` déquantifiée, arrondie en F16 ;
  - tolérance du MMVQ du type ;
  - en plus, une égalité au bit près avec le chemin non fusionné (MMVQ gate, MMVQ up, `swiglu_f32_to_f16`), lancé dans le même sweep ;
  - formes 17408×5120 et les 15 formes standard ;
  - noms CLI `<T>_gate_up_swiglu`, et une ligne `--all-dispatch` par impl.
- Enregistrement : descripteurs d'une nouvelle table `QMATMUL_GATE_UP_SM86`, avec les tests de réciprocité existants étendus. `cert-check` lit la nouvelle section du toml.
- Contrôle de bout en bout :
  - textes identiques au bit près entre bin_R10s et le binaire fusionné sur les invites lp.py (attendus identiques, puisque `gated_f16` l'est) ;
  - parity verte ;
  - gfx906 : 86/0, et .hsaco existants identiques.

## 5. Ordre et critère d'arrêt
1. IQ4_XS seul : gabarit, cellule, sweep (bit près et tolérance) et microbanc nsys. Comparer `fusé` avec `gate + up + swiglu_f32_to_f16` à 17408×5120.
   - **Si le gain est sous 3 µs par couche** (moins de 0,2 ms par token), arrêter là : le gain ne justifie pas la nouvelle table. Proposer plutôt `rmsnorm_q8_1_fused` multi-blocs et le graphe CUDA.
2. Si le gain est confirmé : table `[[qmatmul_gate_up]]`, méthode `Ops`, branchement dans `dense_ffn` ; un commit pour l'infrastructure plus IQ4_XS.
3. Puis Q5_K et Q4_K, un commit chacun.
