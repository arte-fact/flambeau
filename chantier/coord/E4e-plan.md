# E4 (e) — décodage à sortie F16 dans le dispatch (worktree ~/Workspace/flambeau-impl-e, branche e4e-f16-dispatch, base 37418569)

## Appels concernés (27B, TP-2, par token et par GPU, traces du Vibe)
iq4_xs_f16 25 × 71 µs, q4_k_f16 11 × 80 µs, q6_k_f16 17, q8_0_f16 8, iq3_s_f16 3, iq4_nl_f16 3, q3_k_f16 1 : ≈ 3,3 ms. Q/K/V/O de l'attention (`standard_attn.rs:189,243,268,1496`) et ffn_down (`dense_ffn.rs:207`), m = 1. Aujourd'hui ces appels passent par une table codée en dur (`mmvq_f16_direct`, 19 bras) : Q4_K et IQ4_XS lancent les anciens noyaux 256 threads, pas R10b ; Q5_K est exclu par la garde de `forward/src/ctx.rs`.

## Porter la variante « sortie F16 » (règles 1 et 3)
- Pas de ligne ni de trait à part : la sortie F16 est une seconde entrée (`<entry>_f16`) du MÊME noyau (même corps, seule l'écriture finale change, via `mmvq_store<fb_fp16_t>`, saturée à ±65 504). La ligne de dispatch m = 1 désigne l'impl ; sa recette porte `entry_f16: Option<&'static str>`.
- Une seule résolution m = 1 (`mmvq_m1_recipe`) pour `mmvq` (F32) et `mmvq_f16_direct` : recette de la ligne de dispatch, ou noyau fixe des dtypes qui contournent la table (Q4_0, Q5_0, Q5_1). La table codée en dur disparaît.
- `Ops::mmvq_f16_supported(dtype)` (implémentation réelle, pas de défaut) : vrai si la recette m = 1 a une entrée `_f16` et que le module la contient. `QuantWeight::supports_decode_to_f16(&ops)` l'appelle ; la liste codée en dur et l'exclusion Q5_K disparaissent. Sinon : F32 + cast, comme aujourd'hui.

## Noyaux à ajouter (instanciations seulement, gabarits déjà paramétrés par `OutT`)
- gfx906 (R10b) : `flambeau_mmvq_q4_k_dp4a_t128_vdr2_q8_1_f16`, `…q5_k_dp4a_t128_vdr2_q8_1_f16`, `…iq4_xs_dp4a_t128_vdr4_q8_1_f16`. Q5_K redevient éligible : même noyau des deux côtés, donc plus de divergence.
- sm_86 : ses lignes m = 1 ont déjà une entrée `_f16` sauf Q8_0 (`mmvq_q8_0_dp4a`) et Q4_1 (`mmvq_q4_1`) : j'ajoute leurs `_f16` (`mmvq_dp4a_row<…, fb_fp16_t>`). Changement sur sm_86 : Q8_0, Q4_1, Q4_K, IQ4_XS… passent désormais par le noyau de leur ligne de dispatch (avant : t128_vdr2 / 256 threads codés en dur). À mesurer sur la RTX.

## Certificats
L'entrée `_f16` est certifiée par le sweep de son impl : `sweep_mmvq::run_shape` lance aussi `<entry>_f16` quand le module le contient, et la forme échoue si une seule sortie diffère du F32 saturé puis arrondi (égalité au bit près, même corps). Format de certificat inchangé ; `tolerance_formula` le mentionne. Les certificats des 3 impl R10b gfx906 et des impl sm_86 Q8_0/Q4_1 sont à régénérer (Vibe).

## mmvq_f16_direct_parity
Comparer, pour chaque dtype où `mmvq_f16_supported` est vrai, `mmvq_f16_direct` à `mmvq` + cast saturé : au bit près (c'est maintenant le même noyau), hors lignes saturées ; plus la vérification de saturation à ±65 504. La comparaison à la référence flottante reste au sweep, avec la tolérance du certificat.

## Vérification
std ; sweeps des impl touchées (gfx906 : Q4_K/Q5_K/IQ4_XS t128 ; sm_86 : Q8_0, Q4_1) ; mmvq_f16_direct_parity sur GPU ; --all-dispatch ; traces « math » + débit (Vibe), qui donneront le gain réel (aucune projection ici).

## Résolution une seule fois (ajout après revue)
`Registry` porte un `DtypeCache<Option<MmvqM1>>` (un `OnceLock` par dtype, index dense `QDtype::index()`) : recette m = 1 et présence de l'entrée `_f16` résolues au premier usage du dtype, puis lues. `mmvq` (F32, m = 1), `mmvq_f16_direct` et `mmvq_f16_supported` passent tous par lui ; avant, le chemin F32 aussi refaisait dispatch + `Recipe::from_impl_id` à chaque appel. Le dispatch m = 1 ne dépend que du dtype (`first_match(table, dtype, Q8_1, m)`), donc le cache est exact. Test hors GPU : `registry::dtype_cache_tests`. Hors périmètre : `resolve_dispatch_recipe` (m > 1, préremplissage / lots) résout encore par appel.
