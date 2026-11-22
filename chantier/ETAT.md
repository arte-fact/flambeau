# Chantier Flambeau — écarts avec llama.cpp (Qwen3.8-27B-UD-Q4_K_M, 2× MI50, KV q8)

Coordinateur : Fable (`threadripper/claude-flambeau-2991662f`). Implémentation : Opus
(`threadripper/claude-flambeau-inference`, worktree `~/Workspace/flambeau-impl`, branche
`feature/cuda-decouple`, rien de poussé). Mesures : Vibe (`threadripper/vibe-flambeau-exp-b4c7fd79`).
Analyse de référence : `ANALYSE-ECARTS.md` (commit `d468d4f1`).

**MI50 (2×, TP-2) — sans MTP : 29,6 t/s (llama.cpp 25,7) ; MTP-4 : 39,4 t/s (llama.cpp 37,3) ; acceptation 59,3 % (58,8 %).** Mesuré sur `16b3a845`, départ ≤ 50 °C ; référence llama.cpp du 19/11.
**RTX 3090 (glouton pur) — sans MTP : 44,7 t/s (llama.cpp 43,5, +2,7 %) ; contexte 4 k : 43,1 t/s ; MTP-4 : **86,6 t/s sur V4** (local, `b0a8d697`, non poussé) ; 82,5 sur V2 (poussé, `cd3ab132` ; 80,0 sur V1, 79,0 sur `f5d29f30` ; llama.cpp 78,0 à la référence du 19/11, **mais 89,2 sur llama.cpp HEAD ffc87c3a7**) ; acceptation 59,5 % ; TTFT 214 ms. Objectifs de l'étape 5 atteints ; écart MTP-4 contre llama.cpp HEAD ramené de −7,5 % (V2) à −3 % (V4).** Mesuré sur `ee32a49c` (g5, attention vec) ; 44,1 sur `2661d7bf` (g4) ; 43,8 sur `a8ccae8c` (g3) ; graphe seul 42,9 sur `b9b69d2a` ; avant le graphe 41,1, départ ≤ 50 °C. MI50 en glouton pur : 29,75 / 41,63 (llama.cpp 25,7 / 37,3). Avec la pénalité 1,05 par défaut : RTX 40,9 / 78,0, MI50 29,2-29,8 / 41,7.
`origin/feature/cuda-decouple` = `cd3ab132` (V2 : index GDN dense, eh_proj Q6_K, sans copie des logits bonus, quantification unique, parité IQ2_XXS, qmatmul-bench ; V1 : mtp0-noload, embed Q4_K sur device, D4, GDN par K états, reportés sur `18549809` ; étapes 1 à 4, tri des orphelins, R10s, batched sm_86, argmax MTP GPU, graphe CUDA qwen35, argmax mtp 0, grille figée et attention vec sm_86, série 7 et roadmap poussés le 21/11), vérifié par `git ls-remote`. PR brouillon #3 vers `main` ouverte le 2026-11-21 (fusion réservée à Benoît).


## HANDOFF — 22/11 au soir (arrêt demandé par Benoît : crédit de tokens)

Reprise par un autre agent. Tout est arrêté, GPU libre. Coordination = ce fichier + `coord/`. Résultats des runs dans `v1/` à `v4/` (dossier `~/Workspace/flambeau-exp`, copie poussée sur la branche `coord/chantier-ecarts-handoff`, dossier `chantier/`).

**Branches (GitHub arte-fact/flambeau ; jamais main, jamais de fusion : réservée à Benoît)**
- `origin/feature/cuda-decouple` = `cd3ab132` (V2, contrôlée, PR #3 en brouillon).
- `origin/feature/cuda-decouple-v4` = **`9c6621b7`** (branche locale `v3` d'Opus, worktree `~/Workspace/flambeau-impl-o` ; = V4 `b0a8d697` + lignes sm_86 Q5_K/Q3_K + FewAct ; détail `coord/opus-handoff.md`). **Ne pas la fusionner telle quelle : elle contient la régression MI50 (lignes gfx906 de 6b69f618).**
- `origin/feature/cuda-decouple-chantier-ecarts` (838f3db3) : à supprimer ou non, décision de Benoît.

**Chiffres (Qwen3.8-27B-UD-Q4_K_M, kv q8, lp_g glouton, t/s ; llama.cpp HEAD RTX MTP-4 = 89,2)**
| | RTX mtp 0 | RTX MTP-4 | MI50 TP-2 mtp 0 | MI50 TP-2 MTP-4 |
|---|---|---|---|---|
| V2 `cd3ab132` | 44,70 | 82,49 | 28,00 | 41,78 |
| V3 `50864a2f` | 44,67 | 82,30 (bit-exact V2) | 28,46 (bit-exact) | **40,36 (−3,4 %)** |
| V4 `b0a8d697` | 44,69 | **86,57 (+4,2 % vs V3, 97 % de llama.cpp)** | = V3 | = V3 (même dispatch gfx906) |
Autres : gemma-4-12b Q4_0 RTX 81,25 en V3 (HTTP 500 en V2), MI50 +7,9 % ; ornith-35b bit-exact. Comparer seulement des runs alternés (salle chaude, MI50 dérive de ±2 %).

**Profilage MI50 D1/D2 (fait, `coord/mi50-mtp4-diag-resultats.md`, scripts `coord/diag-mi50/`, traces brutes perdues avec le scratchpad d'Opus)** : régression V3 ≈ 90 % H2 (Q4_K row-tile → x8 à m = 5 : la tuile x8 coûte ≈ 140 µs fixes sur les formes TP-2 ≤ 5 120 lignes, moins d'une vague sur 60 CU), ≈ 8 % H1 (double quantification DS4). +2,3 ms par cycle et par GPU.

**Lignes sm_86 few MMA** : Q4_K 3..24 actif et certifié (V4, all-dispatch 34/34). Q5_K 4..24 et Q3_K 3..24 : dans `9c6621b7`, sweeps certifiés, cert-check sm_86 39/0 et gfx906 90/0, tests hôte verts ; **non faits : all-dispatch sm_86 (run invalidé par un rebuild), tests GPU CUDA, MTP-4 RTX** ; aucun binaire V5 produit (bin_V4 = b0a8d697). Q3_K lit l'échelle D4, Q4_K/Q5_K Ds4 : sur un site mixte (Q3_K_M), repli batched prévu (option a) avec test de site mixte.

**Décisions provisoires (Benoît peut revenir dessus)**
- push de V2 sur feature/cuda-decouple ; garde thermique MI50 relâchée à ≤ 56 °C (jonction au repos 54-55 °C) ;
- bande sm_86 Q4_K few 3..24 (m = 2 en MMVQ batched) ; Q5_K 4..24 ; Q3_K 3..24 avec repli sur site mixte ;
- câblage few avant F3/F4 ; repli batched des sites non câblés, figé par le test few_activation_sites ;
- D3 accepté, non fait : gfx906 Q4_K row-tile à m = 2..8 comme V2 ; table gfx906 réglée pour TP-2.

**Prochaines étapes, dans l'ordre**
1. Sur `9c6621b7` : build, all-dispatch sm_86, tests CUDA forward et parité (Q5_K/Q3_K few).
2. D3 : remettre gfx906 Q4_K row-tile à 2..8 ; qmatmul-bench gfx906 **aux formes TP-2 par GPU, quantification comptée** (row-tile / x8 / x16, m = 5..16, Q4_K et Q5_K ; Q3_K x8 contre row-tile 2..8) ; toute ligne de 6b69f618 perdante revient à V2 ; cert-check 90/0 gfx906, all-dispatch.
3. Vibe : `coord/v5-consigne.md` (MI50 MTP-4 V2 contre V5 alterné, RTX V4 contre V5, ornith). Si MI50 ≥ V2 et RTX ≥ V4 : push fast-forward sur feature/cuda-decouple, mise à jour PR #3 et billet `mr-flambeau-3.md` (news-contenu, branche redaction).
4. Ensuite : écart restant MTP-4 RTX (≈ 3 %, attention flash_tile n = 5, Q8_0 alpha/beta, trous de lancement → graphes CUDA), tuile 32 lignes few pour m = 9..16, bande n dans la table, expérience parité gemma (MMA prefill), anomalie Q3_K batched n4 > n5, puis étape 6 DFlash.

**Pièges**
- Microbench de décision gfx906 aux formes complètes = faux : toujours formes TP-2 par GPU, quantification comptée.
- Un seul processus GPU à la fois ; pas de build ni clippy pendant une mesure du Vibe (charge CPU) ; ne pas toucher sd-server (RTX, PID 5083) ni le moniteur.
- Jamais checkout/commit/stash/reset dans `~/Workspace/flambeau` ni `~/Workspace/flambeau-impl` (worktrees partagés).
- Les 79-80 échecs de tests sans GPU sont des tests GPU lancés cartes masquées (« no ROCm-capable device ») : pas une régression.
- pp2tp2 demande 4 GPU ; tests TP=2 27B/gemma 31B ont besoin de ctx_cap 4096.
- mcp-agents : `send_prompt(session_id, text)` ; gitlab : projet `artefact/news-contenu`.
- Sur MI50, lignes `mtp.cycle` dédoublées dans les logs (ratio 2, prendre [::2]).

## Résumé — 21/11 (s7, série 7 ; base DFlash2 RTX)

- **s7 vert** (`f5d29f30`) : 86/86, 33/33 ; RTX sans MTP 44,73 (bruit vs 44,86 de G5 le même jour), **MTP-4 79,0 (+1,3 %, IQ3_S batched)** et relance identique ; f16 43,78 ; MI50 29,75 / 41,62 avec textes identiques ; streaming égal au non-streamé (5/5). Divergences de texte sur la RTX : ordre des sommes à m ≥ 2. Push de `18549809` (roadmap + série 7) confié à Opus.
- **Base DFlash2 sur la RTX** (llama.cpp HEAD ffc87c3a7, lp_g) : sans spéculatif **43,47** ; draft-mtp 4 **89,2** (116,7 / 106,1 / 79,7 / 69,9 / 73,8 ; 912/1510 acceptés, ≈ 3,38 tok/cycle) ; **draft-dflash2 109,0** (165,1 / 148,5 / 89,6 / 72,0 / 69,6 ; 972/2435 acceptés). DFlash2 bat leur MTP-4 de +22 %, surtout sur code et math ; il est à égalité sur story et list. VRAM du serveur : 16 783 / 18 041 / 20 433 Mio.
- **Base DFlash2 sur les MI50** (llama.cpp HEAD, -sm layer) : sans spéculatif 25,14 ; draft-mtp 4 **35,63** (58,6 %) ; **draft-dflash2 32,74** (41,4 %, longueur moyenne 4,32). Sur gfx906, DFlash2 perd 8 % contre leur MTP-4 : il gagne sur code et math (53,4 contre 47,3), mais perd sur chat, story et list. Pour mémoire, flambeau en MTP-4 fait 41,62. Acceptation sous les planchers des tests nocturnes de llama.cpp sur nos invites : 0,40-0,41 contre 0,518, et 4,22-4,32 contre 4,54.
- **Verdict (décision provisoire, Benoît peut revenir dessus)** : DFlash2 continue pour la RTX, où il bat MTP-4 à la base. Sur gfx906, la base est négative chez llama.cpp, dont le chemin m = 8 sur MI50 n'est pas le nôtre (lds_tiled x8). gfx906 ne continue que si le microbench de l'étape 2 (coût d'un brouillon m = 8 avec nos noyaux) le justifie ; sinon, résultat nul documenté pour gfx906.
- **Nouvel écart MTP-4** : llama.cpp HEAD 89,2 contre 79,0 chez nous, avec une acceptation par cycle égale (3,38 contre 3,35). C'est donc le coût par cycle qui diffère, ≈ 38 ms contre ≈ 42 ms. Lecture de code (coord/mtp-ecart-head.md) : deux différences de structure. (i) llama.cpp capture des graphes CUDA sur toutes les formes MTP (vérification n=5, rattrapage, brouillon) ; chez nous, seul le décodage n=1 est capturé. (ii) Rollback GDN : chez eux, un état par token et un index ; chez nous, un snapshot de 96 copies D2D, une synchro, puis un rejeu. Attribution par nsys d'un cycle des deux côtés quand le GPU sera libre.
- **Plan VRAM (a)** (coord/vram-embed-q4k-plan.md) : noyau embed_gather_q4_k_f16 sans FMA, égal au bit près à la déquantification hôte, un seul call site (embed_local), appel direct certifié par un sweep à 0 ULP. Gain calculé : −1 743 Mio sur la RTX et −1 743 par rang MI50. Codé sans build (branche vram-embed-q4k : cbcb01f1, 83210609, 0353c9fb) ; à contrôler aussi gemma-4-12b Q4_0 et ornith-35b (token_embd Q4_K).
- **Plan MTP** (coord/mtp-graphe-gdn-plan.md) : D4 (4 brouillons enchaînés sur le device, 1 synchro au lieu de 4, deux arch), GDN par K états et un index (fin du snapshot et du rejeu ; +576 Mio, alloués seulement quand mtp > 0), puis V5 (graphe de vérification, CUDA seulement). Ordre provisoire, à confirmer par nsys. D4 codé sans build (mtp-d4 c193cafc, chaînage seulement avec une table Quant, test à trois chemins) ; GDN K états codé sans build (mtp-gdn-kstates 54aeb95d noyau + sweep différentiel, e6d1f1ea intégration ; fin du snapshot, de la synchro et du rejeu ; +517 Mio net calculé sur sm_86). Prochaine étape : build et certs V1 (mtp0-noload + (a) + D4 + GDN), puis mesure et nsys.
- **v1 vert** (`838f3db3`, v1/RESULTATS.md) : 86/86, 33/33 ; **textes identiques à S7 sur les 30 comparaisons** (RTX mtp 0 et 4, MI50 mtp 0 et 4, ornith-35b). RTX sans MTP 44,69 (bruit), **MTP-4 78,87 → 79,95 et 80,03 (+1,4 %)**, acceptation inchangée (3,35 tok/cycle). **VRAM RTX −2 140 Mio en mtp 0** (calcul −2 137) **et −970 Mio en mtp 4** (calcul −1 226, 256 Mio de moins que prévu, à expliquer par Opus). MI50 29,81 (+0,2 %) et MTP-4 41,04 (−1,4 % non alterné ; **v1b en alternance : −0,1 %, bruit**, le −1,4 % était thermique) ; VRAM 73 → 65 % par carte. ornith-35b 69,87 → 69,46 (bruit). **gemma-4-12b-it Q4_0 : HTTP 500 sur S7 comme sur V1** (`kernel stem 'mmvq_q4_k_r4' not loaded` au préremplissage, sm_86) : bogue antérieur à V1, ouvert.
- **V1 poussé** : `origin/feature/cuda-decouple` = `2ecc291c` (fast-forward depuis `18549809`, 8 commits reportés par cherry-pick ; seul doc/ROADMAP.md diffère de `838f3db3`, donc les mesures v1 restent valables ; ls-remote vérifié). La branche de secours `feature/cuda-decouple-chantier-ecarts` (`838f3db3`) est gardée ; sa suppression revient à Benoît.
- **nsys MTP-4 RTX** (coord/mtp-ecart-nsys.md, Opus) : cycle 43,55 ms contre 40,66 chez llama.cpp HEAD (+2,89). **Tout l'écart est dans la vérification n = 5 (+6,5 ms)** : matmul +3,76 ms (Q5_K et Q4_K en MMVQ batched chez nous, en MMQ « few » chez llama.cpp ; IQ4_XS à égalité), attention +1,33 (tuile de préremplissage, 99 µs par couche contre 16), Q8_0 alpha/beta +0,82, trous de lancement +1,65 (gonflés par CUPTI). Nous reprenons 3,6 ms sur les brouillons et la relecture. V1 contre S7 : copies GDN 608 → 2 µs, cycle −0,70 ms. VRAM : les K états sont alloués pour les 65 couches au lieu des 48 couches GDN (+255 Mio, erreur de e6d1f1ea).
- **Plan (décision provisoire, Benoît peut revenir dessus)** :
  - V2, petits commits : K états sur les seules couches GDN, eh_proj en Q6_K (−968 µs par cycle), pas de copie des logits bonus en glouton, une seule quantification des activations.
  - Ensuite, MMQ few pour m = 2..16 sur sm_86. Un microbench dira d'abord si nos MMQ de préremplissage suffisent ; dans ce cas, ce sera une ligne de dispatch. Ce chemin sert aussi la vérification DFlash2.
  - Puis V5 (graphes) et l'attention multi-lignes n ≤ 8.
  - Le Vibe remesure la MI50 (v1b, alternance S7/V1) pendant qu'Opus code sans GPU ni build.
  - **V2 codé** (branche locale v2, sur `2ecc291c`, non poussée) : index GDN dense (8dc39193 ; −255 et −53 Mio calculés), eh_proj en Q6_K (9beeb2de ; les brouillons peuvent bouger), sans copie des logits bonus (e42da802), quantification unique (45013ccf). Opus y ajoute le commit de parité IQ2_XXS et le harnais qmatmul-bench (étape 0 de coord/mmq-few-plan.md : sm_86 n'a aucune MMQ K-quant routée, mais la MMQ x8 partagée est déjà compilée pour CUDA ; si un candidat existant bat le MMVQ à m = 5 et m = 9, une ligne de dispatch suffit, sinon on porte mul_mat_q_few). Ordre : build après v1b, mesure v2 (Vibe), puis microbench (Opus).
  - Étape 0 codée : v2 compte 6 commits (1c52145a parité IQ2_XXS : même argmax, cosinus ≥ 0,995, préfixe ≥ 2 ; 265ea8ef qmatmul-bench, ids C0 à C4 sous la feature bench, inatteignables par le dispatch). **Constat** : sur gfx906, à m = 3..8, un raccourci row-tile batched de qmatmul passe avant la ligne x8 de la table, donc ce choix se fait hors dispatch (esprit des règles 1 et 2). Décision provisoire : on mesure les deux au microbench, puis on fait du gagnant une ligne de la table, avec son cert, et on supprime le raccourci.
- **gemma-4 sur sm_86, deux bogues antérieurs à V1** : 12B Q4_0 en HTTP 500 (`mmvq_q4_k_r4` non chargé) ; test real_gguf_cuda_parity (12B Q4_K) qui diverge de llama.cpp au 5e token. Diagnostic par lecture confié à Opus (coord/gemma-sm86-diag.md), sans GPU, pendant v2.
  - (1) **Q4_0 en HTTP 500** : output_head.rs:82-99 appelle en dur `mmvq_q4_k_r4`, un noyau HIP sans port CUDA, dès que la tête liée est en Q4_K et que le vocabulaire fait au moins 131 072. Encore un choix fait hors de la table (règles 1 et 13). Décision provisoire : on l'ajoute à l'inventaire des raccourcis ; le microbench mesure r4 comme candidat C5 ; correctif dans le même commit que les autres raccourcis.
  - (2) **Parité 12B** : le GGUF « Q4_K_XL » ne contient que du Q4_0. Le test était exact jusqu'au 19/06 (b538847e). Suspect principal : l'attention de préremplissage MMA sur sm_86 (182ef2f5, e586c6a9), utilisée dès m ≥ 16. Expérience à un seul build (flash_tile forcé, branche jetable), puis bisect si besoin.
  - Les 79 tests en échec sans GPU sont des tests GPU lancés avec les cartes masquées, la même liste en V1 et en V2. Les tests session_* sur modèle réel (TP et PP sur MI50) n'ont pas été relancés : ils passent avant tout push de V2.
  - Ordre après v2 : build incrémental des binaires de bench (C5 ajouté par cd3ab132, sans changement de noyau), microbench, puis session_* sur MI50, puis l'expérience gemma.
  - **Décisions provisoires après le microbench** (Benoît peut revenir dessus) :
    - port de mul_mat_q_few sur sm_86, en plusieurs sessions, plan de découpage `coord/mmq-few-port-plan.md` ;
    - commit des raccourcis sur la branche v3 : décisions gfx906 1 à 3, branche use_r4_lmhead supprimée, ce qui corrige aussi le HTTP 500 de gemma Q4_0 ;
    - `mmvq_q4_k_r4` passe sous cfg(unverified) au lieu d'être supprimé, faute de confirmation de Benoît ;
    - dtypes encore non mesurés du raccourci : bench à venir ;
    - push de V2 après v2b et les session_* sur MI50.
  - **v3** (locale, sur v2) : 6b69f618, les raccourcis mesurés deviennent des lignes de table (gfx906 Q4_K, Q5_K, Q3_K ; tête r4 supprimée, noyau sous cfg(unverified)). Certs à produire : Q4_K x16, puis re-sweep Q3_K et Q4_K x8. Plan de port `coord/mmq-few-port-plan.md` : F0 ActLayout comme type (66c52ff7, codé, non testé), F1 quantification au format fragment (e2b00e94, codé, cert à produire au build), puis F2 Q4_K (11db931c, codé : noyau mmq_few_q4_k_mma, 236-242 registres sans spill, sweep et bench C6, aucune ligne de dispatch), F3 Q5_K, F4 Q3_K, F5 les autres dtypes, cible m = 2..24.
    - Décision provisoire : F3 et F4 attendent la validation de F2 sur GPU, pour ne pas empiler trois dtypes sur une structure non mesurée (règle 14). v2b a ralenti, garde relevée à ≤ 56 °C sur les deux cartes, jonction au repos 54-55 °C.
    - Séquence après v2b (`coord/v3-sequence.md`) :
      1. session_* sur les 2 MI50 (bin V2), avant le push de V2 ;
      2. build de v3 ;
      3. les 5 certs ;
      4. cert-check, tests et textes du Vibe ;
      5. C0/C6 sur la RTX, qui valide F2 ;
      6. bench gfx906 des raccourcis restants (≈ 3 h 30), pendant lequel Opus code F3 et F4.
    - **Étape 1 (session_* sur les 2 MI50, v2 `cd3ab132`)** : 4 tests verts (9B PP=2 et TP=2, 9B Q3_K_S, 27B TP2 = PP2), 3 SKIP (modèles absents), 2 échecs qui ne viennent pas de V2. pp2tp2 demande 4 GPU. Qwen3.5-27B Q4_0 TP=2 sort en OOM, à l'identique sur V1, parce que le test ne fixe pas ctx_cap et que le KV est alloué sur 262 144 tokens. Décision provisoire : push de v2 accordé ; **poussé, `origin/feature/cuda-decouple` = `cd3ab132`** (fast-forward, 7 commits, ls-remote vérifié). Commit test seulement sur v3 (59fb0ba4) : ctx_cap 4096 pour les tests 27B Q4_0 et gemma 31B, qui passent désormais ; le nom du GGUF gemma 31B est aligné sur le fichier présent. Build v3 fait : 276 hsaco identiques à v2, et les cubins few sont présents.
    - **Étape 4 verte (v3 `50864a2f`, impl/bin_V3.txt)** : certs commités (x16 Q4_K nouveau, x8 Q3_K/Q4_K re-sweepés, quantize few à l'octet près, Q4_K few MMA erreur max 8,4e-4 pour 3e-2) ; cert-check 90/0 et 33/0 ; all-dispatch 89/89 et 33/33 ; 595 tests ok, 80 échecs = GPU masqués (79 + gemma 31B qui ne saute plus) ; tests GPU CUDA et HIP verts, croisés identiques à V2. bin_V3 5c6bbda2…, bin_V3_CUDA eacb135a…. Séquence modifiée : après l'étape 5 (C0/C6 RTX), GPU au Vibe pour le contrôle v3 (`coord/v3-consigne.md`), Opus code F3/F4 sans GPU ; étape 6 après « v3 terminé » ; push de v3 après ce rendu.
    - **Étape 5 (C0/C6 RTX, nsys, coord/mmq-few-resultats.md)** : coût Q4_K de la vérification 27B, ratio C6/C0 : m2 1,029 ; m3 0,966 ; m4 0,882 ; m5 0,782 ; m8 0,555 ; m9 0,446 ; m16 0,343 ; m24 0,243 (répétitions à ≤ 1,5 %). Contre llama.cpp sur 4096×14336 : C6 devant à m ≤ 8 (43,2 contre 45,6 µs à m5), 2,6 % derrière à 9..16 (tuile 32 lignes non instanciée, spill). Bande retenue sm_86 Q4_K : few MMA m = 3..24, m = 2 en MMVQ batched. Décision provisoire : câblage du buffer few dans le forward (7 sites) et ligne Q4_K avant F3/F4, pour mesurer le gain MTP de bout en bout. GPU rendu au Vibe pour v3.
    - **Câblage few codé (v3, sans GPU)** : 41982716 (QmatmulBuffers.act_q8_1_few ; repli explicite sur le batched hors tuilage ou sans buffer few ; quantize_f32_q8_1_few pour ssm_out), 791daf50 (buffers few dans le pool et le scratch GDN, 7 sites câblés ; attn_mixed, routeur MoE, têtes en repli), 89237dc6 (sm_86 Q4_K : batched m=2, few m=3..24, batched ≥ 25). Clippy vert, tests hôte verts sauf le test de présence du cert quantize_f32 few. Schéma validé (décision provisoire) à deux conditions : le cert du batched doit couvrir m = 3..24 pour les sites en repli ; un test fige la liste des sites câblés et en repli. F3 puis F4 sur ce modèle. Aucun build pendant les mesures du Vibe. Créneau suivant, après « v3 terminé » : séquence RTX d'Opus (cert-check attendu 35/0), MTP-4 du Vibe sur ce binaire, puis étape 6 MI50.
    - **F3/F4 codés, non compilés** : 8afec00e (test few_activation_sites : fige site par site les câblés et les NULL), 542eb3be F3 (noyau factorisé dans mmq_few_mma.cuh, décodeur par type, Q5_K ; build.rs CUDA hache désormais les en-têtes, avant une modif d'en-tête ne recompilait pas les .cu), 0b54b98e F4 (Q3_K, anneau non aligné, mma k16, activation D4). Bench seulement, pas de ligne. Condition 1 : les noyaux du repli sont couverts n1..n8 par le cert batched ; l'enchaînement des colonnes ne l'est pas (déjà vrai avant pour m ≥ 2) → cas qmatmul complet m=9/17/24 ajouté au sweep batched (accepté). À vérifier au build : un cubin change-t-il à source égale (cert périmé possible). Réponse : non par construction (src/kernels n'avait aucun en-tête avant 41982716) ; hashes des 259 cubins de 50864a2f relevés pour comparaison au rebuild. b4f6a7cf : 6 cas d'enchaînement (9/17/24 colonnes via la recette dispatchée) par cert batched Q4_K/Q5_K/Q3_K sm_86, 120 → 126 formes, re-sweep au créneau RTX ; cert-check attendu inchangé 35/0 et 90/0.
    - **v3 contrôlé** : RTX bit-exact ; gemma Q4_0 corrigé ; **régression MI50 MTP-4 −3,4 %** (lignes gfx906 de 6b69f618, cycle +1,6 % malgré le microbench, et acceptation −1,7 point par divergence des textes). Push de v3 bloqué. Étape 6 remplacée par un diagnostic rocprofv3 V2 contre V3 d'un cycle MTP-4 (plan d'Opus attendu). En cours : séquence RTX d'Opus.
    - **Créneau RTX (b0a8d697)** : 259 cubins identiques à l'octet (trou build.rs sans effet) ; 0 spill sur les 9 noyaux few ; sweeps verts (quantize F32 few, few Q5_K et Q3_K, batched 126 formes dont l'enchaînement, pire cas à 0,66 de la tolérance) ; cert-check sm_86 35/0, gfx906 90/0 ; all-dispatch sm_86 34/34 ; tests en cours.
    - **Régression MI50, lecture des logs** : sur code et math, mêmes cycles et même texte, mais +3,1 % par cycle (77,57 → 79,95 ms ; 78,97 → 81,43), reproduit sur les deux alternances. C'est du temps par cycle ; l'acceptation ne bouge que sur les invites divergentes. Seul Q4_K change de noyau à m = 5 (row-tile → x8). Hypothèses : H1 double quantification (le x8 lit DS4 ; ≈ 100 lancements de plus par vérification, non comptés au bench) ; H2 bench fait aux formes complètes sur un GPU, alors qu'en TP-2 les formes sont divisées par deux (décision Q4_K m = 5 prise sur 0,6 %). Plan validé (`coord/mi50-mtp4-diag-plan.md`) : D1 rocprofv3 V2/V3 alterné, D2 ventilation par cycle H1/H2, D3 seulement après accord. Ordre GPU : fin du créneau RTX → v4 du Vibe → D1. Règle retenue : microbench de décision gfx906 aux formes TP-2 par GPU, quantification comptée.
    - **bin_V4 (b0a8d697)** : bin_V4 7068b617…, bin_V4_CUDA 04ac45a7… ; 259 cubins identiques + 3 nouveaux, 0 spill ; tests sans GPU 599 ok / 80 échecs habituels ; tests CUDA verts ; MoE cosinus 1.0000000, IQ2_XXS 0,996185 (0,997056 en V3, attendu du few Q4_K CUDA). C6 sm_86 : few bat le batched dès m = 3 pour Q5_K (+0,9 % seulement à m3, 6 331 contre 7 510 à m5, 7 713 contre 32 004 à m24) et Q3_K (403 contre 554 à m3). Bandes proposées Q3_K 3..24, Q5_K 4..24, en attente du v4. D1/D2 prêts (d1.sh, d2.py). v4 lancé ; D1 après « v4 terminé ».
    - **v4 vert** : RTX MTP-4 +4,2 % (86,57), 97 % de llama.cpp. Lignes sm_86 Q3_K few 3..24 et Q5_K few 4..24 acceptées (décision provisoire). D1 lancé sur MI50 ; les lignes sont codées pendant D1, sans build.
    - **D1/D2 (coord/mi50-mtp4-diag-resultats.md)** : régression MI50 V3 ≈ 90 % H2, ≈ 8 % H1 (4 traces à ±1 %). Par cycle et par GPU : +2,3 ms d'occupation, dont matmul Q4_K row-tile → x8 +2,1 ms, quantification DS4 +0,28 ms (12,9 → 92 lancements), Q8_1 −0,09 ms. Aux formes TP-2 (≤ 5 120 lignes), la tuile x8 (MMQ_Y = 64) coûte ≈ 140 µs fixes (48 à 96 blocs pour 60 CU, moins d'une vague) ; le microbench aux formes complètes doublait les blocs. Décisions provisoires : gfx906 Q4_K row-tile à m = 2..8 (comme V2, retire aussi H1) ; bench gfx906 aux formes TP-2, quantification comptée (row-tile/x8/x16, m = 5..16, Q4_K et Q5_K ; Q3_K x8 contre row-tile à 2..8), toute ligne de 6b69f618 perdante revient à V2 ; bande n notée pour plus tard. sm_86 : Q5_K few 4..24 codé (build en cours) ; Q3_K few 3..24 avec repli batched sur les sites mixtes Q3_K + Q4_K/Q5_K (échelles D4 contre Ds4 incompatibles ; aucun modèle sur disque concerné, un Q3_K_M le serait), test de site mixte. Ensuite : contrôle V5 du Vibe (MI50 MTP-4 contre V2, RTX contre V4).
    - Anomalie Q3_K n4 : 128 contre 168 registres, ce qui étale les chargements. Elle passe après le port.

## Résumé — 21/11 (g5, attention vec sm_86)

- **Attention de décodage vec_dpos sur sm_86** (Opus, `ee32a49c`, seuils fixés d'avance passés) : **g5 vert**, 86/86, 29/29. RTX court 44,33 → **44,66 (+0,7 %)**, **contexte 4 k 40,4 → 43,11 (+6,5 %)**, MTP-4 77,96 (bruit), MI50 29,77 avec textes identiques. Divergences de texte attendues (ordre des sommes, mêmes quasi-ex-æquo).
- stop-trim (`cbfd2223`) : débit identique, textes identiques à la base sur lp_g.
- Suite : push `ee32a49c`, puis série 7 (certs gfx906 17408 + stop-trim + r2-sm86 + IQ3_S batched) à contrôler avant push ; base llama.cpp DFlash2 (build HEAD).

## Résumé — 21/11 (étape 6 préparée)

- **Note d'analyse DFlash2** : coord/dflash-analyse.md (sur la base de dflash-etude-llamacpp.md et de dflash-inventaire-flambeau.md). Première étape : base mesurée llama.cpp draft-dflash2 contre draft-mtp sur RTX et MI50 ; si DFlash2 ne bat pas MTP, on s'arrête là. Commandes en préparation (Opus).

## Résumé — 21/11 (g3, argmax GPU en mtp 0)

- **Argmax glouton sur GPU en mtp 0** (Opus, `a8ccae8c`, couvre aussi la pénalité 1,05) : **g3 vert**, 86/86, 27/27. RTX sans MTP 42,97 → **43,78 t/s (+1,9 %, bit-exact)**, avec pénalité +2,2 %. MTP-4 78,44, MI50 29,72, textes identiques. **Les deux objectifs RTX sont atteints (43,5 / 78,0).** Le pipeline (b) `eba8486f` est un résultat nul (43,77), non poussé (coord/g3-pipeline-nul.md).
- Poussé : `a8ccae8c`. **g4 vert** (`2661d7bf`, grille figée) : RTX q8 43,71 → **44,11 (+0,9 %, bit-exact)**, MTP-4 78,36, f16 neutre, MI50 29,87 ; 1 capture sous 1024 tokens. g5 (attention vec sm_86) : 1re passe nulle (×0,83) ; **2e passe g5b `aadfacb3` : seuils passés** (×1,87 à 181 tokens, ×1,60 à 1 k, ×2,28 à 4 k, ×2,74 à 8 k ; n_par 14), câblage en cours ; prérequis f722655e (attention par la table) gardé. stop-trim : 4 commits (stream = non-stream, sémantique OpenAI/llama-server, décision provisoire).

## Résumé — 21/11 (g2, graphe CUDA qwen35)

- **Graphe CUDA du décodage qwen35** (Opus, `850cb09b` + `b9b69d2a`) : **g2 vert**, 86/86, 27/27. RTX sans MTP 41,16 → **42,91 t/s (+4,3 %, bit-exact)**, MTP-4 neutre (78,2), MI50 29,57 (−0,6 %, textes identiques). Reste −1,4 % contre llama.cpp.
- Trace (sous nsys) : 22,25 ms de noyaux par token, 3,99 ms d'inactivité, cuGraphLaunch 2,72 ms par token côté hôte, recapture tous les 33 tokens (6,7-12,3 ms). Analyse : coord/g2-suite.md (Opus).

## Résumé — 21/11 (g1, glouton pur)

- **À conditions égales (sans pénalité des deux côtés)** : RTX **MTP-4 78,56 contre 78,0 (objectif atteint)**, sans MTP 41,13 contre 43,5 (−5,5 %). MI50 29,75 / 41,63 contre 25,7 / 37,3 (+16 % / +12 %). La pénalité 1,05 coûte 0,7 % sur RTX et 1,8 % sur MI50 sans MTP. Le −1 % MI50 de s1 était du bruit (textes identiques).
- Écart sans MTP RTX (coord/g-nomtp.md) : ≈ 1,07 ms/token de GPU inactif (lancements hors graphe) + 0,39 ms d'attention splitk. Graphe CUDA qwen35 en cours (Opus, branche locale g2-decode-graph).

## Résumé — 21/11 (s1, échantillonnage MTP sur GPU)

- **Argmax du brouillon et de la vérification sur GPU** (Opus, `dd2cbb8d` + `993a8f2f`, op `argmax_f32` certifiée sur les deux arch) : **s1 vert**, textes bit-exacts sur 8 comparaisons. **RTX MTP-4 72,04 → 77,96 t/s (+8,2 %)** ; MI50 MTP-4 39,44 → **41,71 (+5,8 %)** ; sans MTP inchangé. GPU inactif 8,6 → 3,8 ms par cycle (cycle mur 43,5 ms).
- Attention : flambeau tourne avec la pénalité 1,05 par défaut, llama.cpp sans. g1 refait les mesures en glouton pur.

## Résumé — 21/11 (r10m, batched m ≥ 2 sur sm_86)

- **MMVQ batché R10s m ≥ 2 sur sm_86** (Opus, 7 commits, HEAD `81632285`) : **r10m vert** 86/86, 27/27. RTX MTP-4 45,42 → **71,97 t/s (+58 %)**, sans MTP 40,88 (inchangé), **TTFT 773 → 214 ms** (préremplissage par 8). MI50 29,80, textes identiques.
- Trace MTP-4 : **39,0 ms par cycle** hors préremplissage (avant 75,1) ; préremplissage 343 ms GPU pour 67 tokens.
- Reste vers 78 : la tête Q6_K m = 1 du brouillon (1 069 appels à 267 µs, ≈ 6,3 ms/cycle, 13,6 % du GPU) et Q5_K/Q4_K batchés (×1,3-1,5 à n = 5). Sans MTP : 40,9 contre 43,5.

## Résumé — 20/11 (R10b sm_86)

- **R10b porté sur sm_86** (Opus, branche locale `r10b-sm86`, 7 commits sur `a45374a0`, HEAD `1b31581f`) : µs par appel, 17408×5120 : IQ4_XS 113→57 (llama 56), Q5_K 118→72 (71), Q4_K 94→60 (59), Q3_K 134→52, IQ4_NL 99→60, IQ3_S 99→59. gfx906 : .hsaco identiques octet pour octet. **r10s vert** : 86/86, 20/20 ; RTX sans MTP 27,51 → **40,83 t/s (+48 %)**, MTP-4 43,12 → **45,50 (+5 %)** ; MI50 29,79 (+1,6 %, textes identiques). Trace : 5,79 s de noyaux (contre 9,07), 6,32 s client. **Le MTP ne profite presque pas** : la vérification à 5 tokens passe par l'ancien noyau batched (53 ms sur 75 par cycle, llama.cpp ≈ 56 ms par cycle). Prochain levier : batched R10s n = 2..8 sur sm_86 (Opus). Ensuite : préremplissage sm_86 ligne par ligne (1,3 s pour 67 tokens, TTFT).

## Résumé — 20/11 (r5)

- **Profil nsys RTX (r5, coord/r5-analyse.md)** : flambeau 9,07 s de noyaux pour 200 tokens contre 4,54 s chez llama.cpp ; GPU occupé à 94 %, donc le graphe CUDA pèse ≤ 5 %. **Cause : les MMVQ IQ4_XS / Q5_K / Q4_K / Q3_K sur sm_86, ×1,6 à ×2,1 plus lents à forme égale** (Q6_K à parité). Levier 1 : R10b sur sm_86 (Opus) ; levier 2 : fusion gate+up ; levier 3 : graphe qwen35.

## Résumé — 20/11 19:15

- **eo (`d438e9c9`) vert** : sweeps 88/88 et 20/20 ; MI50 29,43 t/s, RTX 27,48 t/s sans MTP (−0,4 %, bruit) ; textes identiques. Baisse à 4 clients q8 : **non reproductible**, `d35c6963` et `16b3a845` donnent tous deux ≈ 18,7 t/s en alternance (le 23,7 d'e4c était une condition de mesure). Point clos.
- **eo2 vert sur `0b782475`** (branche e4-orphans complète). Certificats MoE : `a45374a0`. **Poussé : `origin/feature/cuda-decouple` = `a45374a0`** (fast-forward, 10 commits).
- Suite : étape 5, rattraper llama.cpp sur la RTX 3090 (profil nsys flambeau contre llama.cpp CUDA). Adaptation du llama-monitor : **faite** (`ebd9026`, branche locale `flambeau-preset`, 41/41 tests, 2 presets ajoutés, 8 intacts, monitor sur :7778, aucune instance lancée).

## Résumé — 20/11 13:40

- **Étape 4 : contrôle final vert** (`e4f/RESULTATS.md`, `16b3a845`) : sweeps 88/88 et 20/20 ; MI50 29,55 / 39,43 t/s ; RTX 27,59 / 43,27 ; parité verte ; streamé = non streamé en MTP-4 ; 4 clients q8 sans erreur. **Poussé : `origin/feature/cuda-decouple` = `16b3a845`.**
- Tri des certificats (validé par Benoît) : `34dd4eeb` + `d438e9c9` écrits sur la branche `e4-orphans`, compilation et contrôle à suivre, push ensuite.
- ~~À expliquer : 4 clients q8 23,7 → 19,0 t/s~~ : non reproductible (eo, 20/11 19:15).
- Audit MMQ / MoE / lots : **suppression des ≈ 37 candidats validée par Benoît** (commits séparés par famille, chez Opus, tests verts avant push). En attente de Benoît : trou de la règle 2 côté MoE gfx906 (noyaux dp4a exécutés non certifiés), adaptation du llama-monitor.
- Ensuite : (5) RTX 3090, (6) DFlash.

## Chiffres de référence

Protocole : 5 invites `lp.py`, glouton, 400 tokens, moyenne des t/s par invite, départ de
chaque serveur à jonction ≤ 60 °C. Les invites s'enchaînent sans refroidir : les dernières sont
mesurées GPU chauds, pour les deux moteurs.

| | flambeau départ (e073c460) | flambeau HEAD (4d6f9509) | llama.cpp, même session | llama.cpp, ancienne référence (à chaud) |
|---|---|---|---|---|
| Sans MTP | 20,6 t/s | **28,15** | 25,71 | 25,9 |
| MTP-4 | 26,5 t/s | **38,28** | 37,32 | 33,1 |
| Acceptation MTP | 49,3 % | **58,9 %** (3,34 tokens/cycle) | 58,8 % (3,45) | 58,8 % (3,37) |

Mesure finale par invite (code / math / chat / story / list), dossier `final/` :
- flambeau sans MTP : 29,8 / 29,7 / 28,9 / 27,3 / 25,1 ; llama.cpp : 26,4 / 26,3 / 26,0 / 25,2 / 24,8 ;
- flambeau MTP-4 : 54,9 / 48,2 / 34,9 / 24,3 / 29,1 ; llama.cpp : 50,7 / 43,8 / 34,2 / 28,6 / 29,4.

RTX 3090 (un seul GPU, CUDA, KV q8_0, mêmes 5 invites) :

| | flambeau | llama.cpp CUDA (`ac931b5bc`) |
|---|---|---|
| Sans MTP | **27,13 t/s** | 43,49 t/s |
| MTP-4 | **40,97 t/s** (départ à 71 °C) ; 42,53 avec départ ≤ 50 °C (`4cf1823f`) | 77,97 t/s (départ à 61 °C) |
| Acceptation MTP | 59,6 % (3,37 tokens/cycle) | 59,6 % (3,46 tokens/cycle) |
| VRAM, serveur chargé | 19,5 Go | 16,8 Go |
| Premier token (invites de 25 à 67 tokens) | 0,8 à 2,2 s | non relevé |

Historique des étapes (binaires cumulés, sessions différentes) :

Par invite (code / math / chat / story / list) :
- sans MTP, R1 : 22,5 / 22,6 / 20,8 / 19,3 / 18,6 ;
- sans MTP, R3+R4 : 24,2 / 23,9 / 22,1 / 20,5 / 19,1 ;
- sans MTP, cumul jusqu'à R8 : 26,6 / 25,6 / 23,5 / 22,2 / 21,2 ;
- MTP-4, cumul jusqu'à R8 : 53,3 / 48,3 / 34,9 / 24,3 / 28,9 ;
- sans MTP, `bin_R10a` : 27,9 / 27,1 / 25,2 / 23,4 / 22,2 (référence `bin_R8` de la même session : 26,6 / 26,3 / 24,7 / 22,7 / 21,3) ;
- MTP-4, `bin_R10a` : 55,0 / 48,3 / 34,8 / 24,4 / 29,2 ;
- sans MTP, `bin_R10b` : 29,8 / 29,3 / 27,9 / 26,2 / 24,4 (référence `bin_R8` de la même session : 26,6 / 26,4 / 24,7 / 22,5 / 21,0) ;
- MTP-4, `bin_R10b` : 55,1 / 48,6 / 34,8 / 24,4 / 29,1 ;
- MTP-4, R1 : 44,8 / 42,5 / 31,1 / 23,1 / 20,5.

## Recommandations

| # | Description | Gain attendu | Statut | Prochaine action (responsable) |
|---|---|---|---|---|
| R1 | Découpage des têtes GDN en TP-2 (têtes V appariées à K par `j % 16`) | MTP +3 t/s, acceptation ≈ 57 % | **Validée.** Commit `4ffd04f5`. Mesuré : 57,9 %, 32,4 t/s ; écart t0 avec llama.cpp f16 : 0,18 à 0,41 nat. Résidu TP/PP = bruit (voir `analyse-ecarts/parity/RESIDU-PARITE.md`) | Aucune |
| — | Test de parité 27B TP/PP (décodage forcé, KL ≤ 0,05, écart ≤ 0,6 nat à p ≥ 1 %, argmax si marge ≥ 0,5) | garde-fou | **Validé.** Commit `6da0441b`. Échoue avant R1 (9 à 20 positions sur 33 hors seuils), passe après (0 sur 33) | Aucune |
| R2 | Protocole thermique du banc (départ ≤ 60 °C, garde VRAM < 5 %, moteurs alternés) | fiabilité | **Appliqué** dans tous les runs depuis r5r8. Effet mesuré : la référence llama.cpp MTP passe de 33,1 (à chaud) à 37,3 t/s (départ à froid) | Aucune |
| R3 | Position du 1er token décodé (L+1 → L, 4 sites de `decode_loop.rs`) | correction ; acceptation ≥ | Implémentée, non commitée (`impl/R3.patch`, `bin_R3`). Sans MTP : 20,5 t/s (inchangé, attendu). Déterminisme : test non discriminant (l'ancien binaire donne aussi 3 sorties identiques). **Validée.** Commit `43c80326`. MTP-4 : 33,6 t/s, 58,1 % (R1 seul : 32,4 t/s, 57,9 %). Avant correctif, la sortie de « list » dépendait de la requête précédente ; après, 3 sorties identiques | Aucune |
| R4 | Split-K de l'attention de décodage dès 33 positions | base +1,4 t/s | **Validée.** Commit `1e35466f`. Certificats Q8 et F16 verts (16 et 21 formes). Mesuré : 21,7 contre 20,5 t/s (**+1,2**) | Aucune |
| R5 | MMVQ Q4_K à 256 threads (`qmatmul_q4_k_mmvq_dp4a_t256`) | base +1,9 t/s | **Validée.** Commit `11bb074f`. Mesuré : 21,97 → 23,19 t/s (+1,2) ; Q4_K 8 704 lignes 120 → 86 µs. Certificats gfx906 et sm_86 verts (15 formes) | Aucune |
| R6 | Vérification MTP : lm_head par lot, une seule copie vers l'hôte (version réduite, sans argmax GPU) | MTP +2 à 3 t/s | **Validée.** Commit `f4c086eb`. Mesuré avec R8 : MTP-4 33,6 → 37,9 t/s, acceptation 58,9 %. Certificats gfx906 et sm_86 verts (105 formes) | Aucune |
| R7 | `rmsnorm_q8_1_fused` à 1 024 threads | base +0,7 t/s | **Validée.** Commit `4de7ffbb` (avec la correction du sweep générique). Mesuré : 27 → 10,7 µs par appel, +0,5 t/s avec R6. Certificats gfx906 et sm_86 verts | Aucune |
| R8 | MTP dans la fenêtre d'arrêt (24 premiers tokens) | MTP +0,8 t/s | Patch prêt, `bin_R8` compilé (chemin SSE non fait). **Mesuré avec R6 : MTP-4 37,9 t/s**, sans MTP inchangé (23,78). Commit `c4d83a39` | Aucune (chemin SSE à faire plus tard) |
| R9 | lm_head découpée par vocabulaire, argmax combiné | base +0,6, MTP +1 t/s | À faire | Après R5-R8 |
| R10a | MMVQ une ligne à 128 threads (Q5_K, IQ4_XS, Q4_K) | base ≈ −9 ms par token (hypothèse) | **Abandonnée** (hypothèse non confirmée). Mesuré : 24,31 → 25,16 t/s ; +7 % par appel seulement. Patch conservé (`impl/R10a.patch`), non commité | Aucune |
| R10b | MMVQ une ligne, 128 threads : 16 poids par thread (Q4_K, Q5_K) ou 32 (IQ4_XS), comme llama.cpp, plus chargements alignés de 4 octets | base ≥ 26 t/s visé | Implémentée (`impl/R10b.patch`, `bin_R10b`), non commitée. Certificats gfx906 verts (3 × 15 formes). **Mesuré : 24,23 → 27,51 t/s** ; MTP-4 38,38 t/s, 58,9 %. Par appel : IQ4_XS 71 → 51 µs, Q5_K 52 → 49, Q4_K 79 → 78. Fidélité : écart ≤ 0,22 nat contre `bin_R8` (`r10b2/`). **Validée.** Commit `4d6f9509` | Aucune |
| R10c | Q5_K et Q4_K restent à ≈ 9 ns par ligne contre ≈ 6 chez llama.cpp ; chemin à sortie F16 hors dispatch (≈ 3 ms par token) | à chiffrer | À faire | Diagnostic (Fable) |
| R12 | MTP sur le chemin streamé (SSE), par fusion des deux boucles historiques de `decode_loop.rs` | streaming avec MTP : ≈ 25 → ≈ 34 t/s | **Validée.** Commit `430b7da1`, poussé. `decode_loop.rs` 1 900 → 1 361 lignes, nouveau `token_emit.rs`. Mesuré (`r12/RESULTATS2.md`) : streamé MTP-4 24,6 → 33,8 t/s (0,987 du non streamé) ; textes, arrêts et compteurs identiques ; non streamé inchangé ; préfixe de séquence d'arrêt retenu en flux comme llama.cpp. Trois défauts du chemin MTP corrigés | Aucune |
| E5 | Rattraper llama.cpp sur la RTX 3090 : profilage du chemin CUDA, puis découpage R10b dispatché sur sm_86, puis les pistes chiffrées | sans MTP 27,1 → 43,5 t/s ; MTP-4 41,0 → 78,0 t/s | À faire, après l'étape 4 | Profilage CUDA (Fable, mesures par le Vibe) |
| R11 | All-reduce TP : moins de copies, d'événements, de lancements | base +1 t/s | À faire | Après R10 |

## Runs du Vibe

| Run | Statut | Résultat |
|---|---|---|
| R1 : base, MTP-4, cmp_lp | Terminé | 20,8 t/s ; 32,4 t/s, 57,9 % ; écart t0 0,29 / 0,33 / 0,18 / 0,36 / 0,41 nat |
| Parité v2 avant / après R1 | Terminé | échoue avant, passe après |
| `parity_diag` (Fable) | Terminé | résidu TP/PP = bruit |
| A1 déterminisme (`bin_R3` et ancien binaire) | Terminé, non concluant | sorties identiques dans les deux cas |
| A2 MTP-4 `bin_R3` | Terminé (reprise) | 33,6 t/s ; 58,1 %, 3,32 tokens/cycle |
| B1 certificats split-K (`bin_R4`) | Terminé | verts |
| B2 base `bin_R4` puis `bin_R3` | Terminé | 21,7 et 20,5 t/s |
| B3 traces rocprofv3 | `bin_R4` faite et analysée ; `bin_R3` perdue (serveur tué avant l'écriture) | avec R4 : attention 3,6 → 0,9 ms par token et par GPU ; matmul inchangés (≈ 29 ms) |
| r5r8 : `bin_R4`, `bin_R5`, `bin_R7`, `bin_R8` | Terminé (`r5r8/RESULTATS.md`) | 21,97 / 23,19 / 23,70 / 23,78 t/s sans MTP ; MTP-4 37,91 t/s, 58,9 % |
| r10a : 10 sweeps, `bin_R8`, `bin_R10a` | Terminé (`r10a/RESULTATS.md`) | sweeps verts (sm_86 Q4_K refait, vert) ; 24,31 → 25,16 t/s sans MTP ; MTP-4 38,33 t/s, 58,9 % |
| r10b : compteurs matériels | Terminé (`r10b/`) | flambeau Q5_K : MemUnitBusy 89-93 %, VALUBusy 64-67 % ; llama.cpp : 61-74 % et 49-77 % |
| r10b2 : 3 sweeps vdr, `bin_R8`, `bin_R10b` | Terminé (`r10b2/RESULTATS.md`) | sweeps verts ; 24,23 → 27,51 t/s sans MTP ; MTP-4 38,38 t/s, 58,9 % |
| final : `bin_HEAD` et llama.cpp alternés, parité | Terminé (`final/RESULTATS.md`) | sans MTP 28,15 contre 25,71 ; MTP-4 38,28 contre 37,32 ; acceptation 58,9 % contre 58,8 % ; parité verte |
| fidelite : perplexité appariée (5 runs) | Terminé (`fidelite/`) | flambeau 7,762 / 7,760 / 7,743 ; llama.cpp 7,832 / 7,842 |
| rtx : llama.cpp CUDA, RTX 3090 | Terminé (`rtx/RESULTATS_LLAMA.md`) | 43,49 t/s sans MTP ; 77,97 t/s en MTP-4, 59,6 % ; 16,8 Go de VRAM |
| r12 : `bin_HEAD` et `bin_R12`, streamé contre non streamé | Terminé (`r12/RESULTATS.md`) | voir R12 |
| rtx : flambeau CUDA, RTX 3090 | Terminé (`rtx/RESULTATS_FLAMBEAU.md`) | 27,13 t/s sans MTP ; 40,97 t/s en MTP-4, 59,6 % ; streamé = non streamé |
| stab : non-régression `247a91f8`, MI50 et RTX, parité | Terminé (`stab/RESULTATS.md`) | MI50 28,57 / 38,47 t/s ; RTX 27,01 / 40,23 t/s ; parité verte ; textes identiques |
| certs : `sweep --all-dispatch` sur les deux architectures | sm_86 terminé ; gfx906 planté (`stab/CERTS.md`) | sm_86 : 20 PASS sur 20 ; gfx906 : SIGSEGV au 2e sweep |
| certs2 : passe complète après correctifs | Terminé (`stab/CERTS2.md`) | gfx906 88 PASS sur 88 ; sm_86 20 PASS sur 20 ; rc=0 sur les deux |
| e4 : contrôle à mi-parcours de l'étape 4 | Terminé (`e4/RESULTATS.md`) | sweeps verts ; MI50 27,66 / 38,46 t/s ; RTX 27,01 / 40,83 t/s ; textes identiques aux références |
| e4b : ordonnanceur par la boucle commune, charge à 4 clients | Terminé (`e4b/RESULTATS.md`) | 1 slot : validé ; **4 clients : HTTP 500 sur les deux binaires** ; RTX 27,20 t/s |
| e4c : vérification de la concurrence | Terminé (`e4c/RESULTATS.md`) | MI50 q8 : 0 erreur à 1, 2, 4 clients, textes identiques ; f16 : 36,9 t/s à 4 clients, textes divergents ; RTX 4 slots : HTTP 500 en non streamé |
| e4e : chemin F16 dans le dispatch, concurrence RTX | Terminé (`e4e/RESULTATS.md`) | sweeps verts ; MI50 +3,1 % sans MTP ; RTX −5,6 % non concluant (thermique) ; concurrence RTX à 4 slots : 0 erreur |
| rtx2 : chemin F16 sur RTX, température contrôlée | Terminé (`e4e/RESULTATS_RTX2.md`) | sans MTP 27,35 / 26,85 (réf.) contre 26,91 / 26,85 ; MTP-4 42,53 contre 42,80 |
| e4h : contrôle de `7a260d46` | Terminé (`e4h/RESULTATS.md`) | sweeps verts ; MI50 29,32 / 39,16 t/s ; RTX 27,52 / 43,23 t/s ; textes identiques |
| e4f : contrôle final de l'étape 4 | Terminé (`e4f/RESULTATS.md`) | tout vert ; MI50 29,55 / 39,43 ; RTX 27,59 / 43,27 ; 4 clients q8 : 19,02 t/s agrégés |
| r5 : profil nsys RTX flambeau / llama.cpp | Terminé (`r5/RESULTATS.md`, analyse coord/r5-analyse.md) | 9,07 s contre 4,54 s de noyaux ; MMVQ ×1,6-2,1 |
| s1 : argmax MTP sur GPU (`993a8f2f`) | Terminé (`s1/RESULTATS.md`) | 86/86, 27/27, argmax PASS ×2 ; RTX 40,88 / 77,96 ; MI50 29,41 / 41,71 ; textes identiques |
| g5 : attention vec sm_86 (`ee32a49c`) + stop-trim | Terminé (`g5/RESULTATS.md`) | 86/86, 29/29 ; RTX 44,66 / 77,96, 4 k 43,11 (+6,5 %) ; MI50 29,77 ; stop-trim neutre |
| s7 : série 7 (`f5d29f30`) | Terminé (`s7/RESULTATS.md`) | 86/86, 33/33 ; RTX 44,73 / 79,0 ; f16 43,78 ; MI50 29,75 / 41,62 ; stream = non-stream |
| dflash-base : llama.cpp HEAD, RTX D1–D3, MI50 E1–E3 | Terminé (`dflash-base/RESULTATS.md`) | RTX : 43,47 / MTP-4 89,25 / DFlash2 108,98 ; MI50 : 25,14 / 35,63 / 32,74 |
| v1 : mtp0-noload + embed Q4_K + D4 + GDN K états (`838f3db3`) | Terminé (`v1/RESULTATS.md`) | 86/86, 33/33 ; textes identiques à S7 partout (6 × 5) ; RTX 44,69 / 79,95-80,03 (+1,4 %) ; VRAM RTX −2 140 / −970 Mio ; MI50 29,81 / 41,04 (−1,4 %), VRAM −8 points par carte ; gemma-4-12b Q4_0 en échec sur S7 et V1 |
| v1b : MI50 MTP-4 en alternance S7/V1 | Terminé (`v1b/RESULTATS.md`) | S7 41,23 / 40,67, V1 40,94 / 40,89 : −0,1 % (bruit), textes identiques ; le −1,4 % de v1 était thermique |
| v2 : index GDN dense, eh_proj Q6_K, sans copie des logits bonus, quantification unique, parité IQ2_XXS, qmatmul-bench (`265ea8ef`) | Terminé (`v2/RESULTATS.md`) | 86/86, 33/33 ; textes identiques à V1 partout ; RTX 44,70 / **82,49 et 82,44 (+3,2 %)**, acceptation 60,3 % ; VRAM RTX −66 / −390 Mio ; MI50 MTP-4 41,80 contre 41,50 (bruit), VRAM 63 % ; MI50 mtp 0 28,64 (run unique ; v2b en alternance : −0,4 %, bruit) ; ornith +2,0 % |
| mmq-few étape 0 : microbench qmatmul (Opus) | Terminé (`coord/mmq-few-resultats.md`) | sm_86 : aucune MMQ existante ne bat le MMVQ batched (×1,6 à ×3,3 plus lentes), d'où le port de mul_mat_q_few ; gfx906 : row-tile meilleur à petit m, x8 à partir de m = 5 (Q4_K) ou 8 (Q5_K), x16 Q4_K à m = 9..16 (−24 %), x8 Q3_K à m = 2 (−34 %) ; tête de sortie r4 2,0× plus lente que la ligne m = 1 |
| v2b : MI50 mtp 0, V1 et V2 en alternance | Terminé (`v2b/RESULTATS.md`) | V1 28,11, V2 28,00 (−0,4 %, bruit), textes identiques ; garde ≤ 56 °C sur les deux cartes ; niveau absolu plus bas que la veille pour les deux binaires (jonctions à 90-91 °C, salle plus chaude) |
| v3 : contrôle V2 contre V3 (RTX, MI50, ornith, gemma Q4_0) | Terminé (`v3/RESULTATS.md`) | RTX bit-exact, 44,67 / 82,30 (bruit) ; **MI50 MTP-4 −3,4 %** (41,78 → 40,36 ; acc 57,9 → 56,2 %, 3,27 → 3,21 tok/cyc, cycle 78,3 → 79,5 ms) avec textes divergents (math 777, chat 766, story 358, list 61) ; MI50 mtp 0 bit-exact, +0,4 % ; ornith bit-exact −1,0 % ; gemma Q4_0 : RTX 81,25 au lieu de HTTP 500, MI50 +7,9 % bit-exact |
| v4 : RTX MTP-4 de bout en bout, few MMA Q4_K (V3 contre V4) | Terminé (`v4/RESULTATS.md`) | **V4 86,57 contre V3 83,06 (+4,2 %), 97 % de llama.cpp (89,2)** ; acceptation 60,3 %, 3,38 tok/cyc ; textes divergents math (1115) et chat (842), w4 = w2 ; mtp 0 44,69 bit-exact |
| v5 : MI50 MTP-4 contre V2, RTX contre V4, ornith | Annulée (arrêt du 22/11 au soir), consigne prête pour la reprise | — |
| g4 : grille figée (`2661d7bf`) | Terminé (`g4/RESULTATS.md`) | 86/86, 27/27 ; RTX 44,11 / 78,36 ; f16 43,70 ; MI50 29,87 ; textes identiques |
| g3 : argmax GPU mtp 0 (`a8ccae8c`) | Terminé (`g3/RESULTATS.md`) | 86/86, 27/27 ; RTX 43,78 / 78,44 (lp_g) ; (b) nul ; MI50 29,72 ; textes identiques |
| g2 : graphe CUDA qwen35 (`b9b69d2a`) | Terminé (`g2/RESULTATS.md`) | 86/86, 27/27 ; RTX 42,91 / 78,21 (lp_g) ; MI50 29,57 ; textes identiques |
| g1 : glouton pur (sans pénalité) | Terminé (`g1/RESULTATS.md`) | RTX 41,13 / 78,56 ; MI50 29,75 / 41,63 ; pénalité −0,7 % / −1,8 % sans MTP |
| r10m : batched m ≥ 2 sur sm_86 (`81632285`) | Terminé (`r10m/RESULTATS.md`) | 86/86, 27/27 ; RTX 40,88 / 71,97 t/s, TTFT 214 ms ; MI50 29,80 ; 39,0 ms/cycle |
| r10s : R10b sur sm_86 (`1b31581f`) | Terminé (`r10s/RESULTATS.md`) | 86/86, 20/20 ; RTX 40,83 / 45,50 t/s ; MI50 29,79 |
| eo2 : contrôle de `0b782475` (e4-orphans complet) | Terminé (`eo2/RESULTATS.md`) | 86/86, 20/20 ; 20 sweeps MoE sm_86 verts (max 1,24e-2) ; MI50 29,31 / 39,53 ; RTX 27,48 / 43,22 ; textes identiques ; parité verte |
| eo : contrôle de `d438e9c9`, 4 clients q8 alternés | Terminé (`eo/RESULTATS.md`) | sweeps 88/88, 20/20 ; MI50 29,43, RTX 27,48 t/s ; 4 clients q8 : E4c 18,75 / 18,43, E4f 18,93 / 18,86 t/s ; jonction 100-103 °C, sclk moyen ≈ 1,0 GHz |

## Points ouverts

- **Test IQ2_XXS HIP/CUDA (hors V1)** : premier commit fautif dee02058, le préremplissage HIP à petit m passe en MMQ LDS-tiled x8 (certifiée). Cosine des logits HIP/CUDA 0,998417 → 0,997644 ; le glouton 2 bits bascule sur une quasi-égalité au 3ᵉ token (coord/v1-parity-iq2xxs.md). **Décision provisoire, Benoît peut revenir dessus** : ce n'est pas un bug. L'assert d'égalité exacte entre backends devient un préfixe commun minimal plus un seuil de cosine sur les logits de préremplissage, comme pour llama.cpp. Le test sera corrigé par Opus après V1.

- Billet décision branches périmées (477a5f4d) : **tranchée** (c17ba5f6). Benoît a dit oui ; feature/quant-perf-sweep et feature/tool-calling-fixes, vérifiées dans main, sont supprimées. chore/comment-debt (f88c29e7) n'est pas dans main : elle sera supprimée à la fusion de la PR #3.
- **Interview blog** : médias déposés dans news-contenu/medias/2026/11/21/flambeau-interview-*.svg (`0b498f0d`, `8379cfbe`) ; script et PNG dans media/interview/. On attend les questions du rédacteur.

- **VRAM RTX 19,5 contre 16,8 Go chez llama.cpp (coord/vram-rtx.md)** : token_embd en F16 sur le GPU (2,4 Go, contre 0 chez llama.cpp, qui le garde sur l'hôte) ; poids MTP chargés même en --mtp 0 (0,39 Go). Décision provisoire : (b) ne pas charger blk.64 en mtp 0 (en cours) ; (a) embedding Q4_K avec gather quantifié (−1,74 Go) après le verdict dflash-base.

- ~~Couverture des certs lds_tiled gfx906~~ **clos (21/11)** : les 13 sweeps gfx906 passent aux formes 17408 (m 3 à 512, pire écart 0,53 de la tolérance, Q5_K), certs étendus dans `77cf1d6a` (branche locale gfx906-lds-17408). Le cert sm_86 q4_k lds_tiled passe aussi aux formes qwen35 (`c88f76df`, gardé). L'échec vu dans la microbench g6 dépend du générateur de poids : à éclaircir si g6 revient, hors dispatch.
- **g6 (préremplissage MMQ sm_86) mis de côté** (décision provisoire) : les MMQ wave64 certifiés sont 3 à 10× plus lents que le batched ; il faudrait de nouveaux noyaux (MMA int8 ou lds_tiled warp 32). (Le cert sm_86 q4_k lds_tiled, finalement vert, est gardé.)

- **Trous de la règle 2 sur sm_86 (coord/r2-trous-sm86.md)** : `qmatmul_q4_0_mma` (préremplissage Q4_0 à ubatch ≥ 128, table Rust différente du TOML) et `attention_prefill_mma` (tout préremplissage CUDA en KV F16, choisi selon la présence du module) sont atteignables, avec des certs faits à la main. Code fait (branche r2-sm86, 39ebd4b2 sweeps, 5aebee0b ligne Q4_0 coupée + test concordance Rust↔TOML, 85155e55 [[attention_prefill]] par la table) ; sweeps verts (Q4_0 mma 16 formes, attention_prefill_mma 7), certs intégrés ; cert-check sm_86 32/0. Part avec la série 7.

- **Décisions provisoires (Benoît peut revenir dessus)** : pénalité 1,05 par défaut gardée ; pipeline (b) de g3 non poussé (nul). **À soumettre en billet « décision »** dès que GUIDE-BILLET-DECISION.md existe : suppression des branches distantes périmées `feature/quant-perf-sweep` et `feature/tool-calling-fixes` (déjà dans `main`). Billet MR de la PR #3 : `news-contenu` redaction `billets/2026/11/21/mr-flambeau-3.md` (`564fa0d6`).

- **Chaîne d'arrêt et espace final** (vu en g3, antérieur à g3) : en non-streaming, l'espace final est rogné, pas en streaming. À corriger plus tard.

- **mtp 0 contre MTP-4 en glouton pur (coord/g2-mtp-vs-mtp0.md)** : divergence tardive (RTX 2/5, MI50 4/5), mots quasi équivalents. La sémantique de vérification est correcte (model.rs:228-240) ; la cause est l'ordre des sommes batchées m = 5 contre m = 1. Non bloquant. Preuve possible : écart top-1/top-2 aux points de divergence.

- **Comparabilité avec llama.cpp (21/11, coord/s1-penalite.md)** : notre serveur impose une pénalité de répétition de 1,05 à température < 0,1 (state.rs:203-219, `9b3ddbba`), alors que llama-server n'en met pas (1,00). Le coût par token reste comparable, mais pas les textes ni l'acceptation MTP. Après s1, mesure en glouton pur (`repetition_penalty: 1.0` dans lp.py). Le défaut du serveur relève de Benoît.

- MoE : le runtime ne lit pas encore le dispatch MoE (choix r2/r4 par une constante publique, testée contre les prédicats des lignes) ; pas de lignes MoE sur sm_86. Règle 1 partiellement respectée.
- **[Traité dans `12495baf`, branche e4-orphans, non compilé]** Règle 2, MoE gfx906 (trouvé par l'audit, vérifié par grep) : les lignes de dispatch `indexed_moe_mmvq_q4_k_r2_gfx906` et `indexed_moe_mmvq_q4_k_gate_up_gfx906` certifient les noyaux scalaires `q4_k_r2` et `q4_k_gate_up`, alors que le runtime lance les variantes `*_dp4a` (`r2_dp4a`, `r4_dp4a`, `gate_up_r4_dp4a`, choisies en dur). Les noyaux réellement exécutés pour le MoE Q4_K ne sont pas couverts par ces certificats. Hors modèle 27B dense ; touche Qwen3.6 MoE.
- 14 noyaux par lots et gate+up lancés au runtime sont classés comme « référence non invoquée » (`direct_call.rs`) : à reclasser. Les 38 certificats tile8 neutres portent `dtype_weight: "Q8_0"` codé en dur (métadonnée fausse).
- sm_86, trous de la règle 2 : `qmatmul_q4_0_mma` (descripteur actif sans ligne de dispatch ni sweep) et `attention_prefill_mma` (lancé directement, sans sweep ni enregistrement). À traiter à l'étape 5.
- sm_86 : le chemin F16 change de découpage pour Q4_K (256 threads → 64 × 2 lignes), Q8_0 et Q3_K ; sans effet mesurable au niveau du serveur. À reprendre par microbanc à l'étape 5.
- Protocole RTX : la carte bride fortement à chaud (84 °C). Garde ≤ 50 °C au départ désormais ; fréquence, puissance et température maximale relevées à chaque run.
- À 4 clients en KV q8, le débit agrégé (≈ 18,7 t/s) est inférieur à celui d'un client seul (≈ 28 t/s) : coût de l'alternance entre slots sans regroupement. Lié au point suivant.
- Concurrence en KV q8 : corrigée (`d35c6963`) en refusant le regroupement. Manque fonctionnel : un chemin d'attention q8 multi-slots, pour retrouver le traitement par lots en q8.
- RTX à plusieurs slots : erreur 500 corrigée (`4cf1823f`). Restent pour l'étape 5 : un graphe de décodage par slot (12,9 t/s agrégés à 4 clients), et l'écart 23,6 contre 27,2 t/s entre serveur à 4 slots et à 1 slot.
- KV f16 à 4 clients : textes non déterministes d'un client à l'autre. À mesurer (KL entre décodage par lots et décodage seul).
- La résolution de dispatch pour plus d'un token (préremplissage, lots) se refait encore à chaque appel.
- Le llama-monitor interroge `rocm-smi` toutes les 500 ms : différence avec les runs de référence, à surveiller sur les débits MI50.
- Le choix de chemin MoE (`n_pairs >= 8`, seuil par topologie) est codé en dur hors dispatch ; la variable `FLAMBEAU_MOE_TILE8_DECODE` a été retirée sans le remplacer par une ligne de dispatch.
- `ops/tests/mmvq_f16_direct_parity` échoue depuis R10b (égalité au bit près exigée entre deux noyaux différents) ; à redéfinir avec le chemin F16.
- 244 certificats gfx906 et 220 sm_86 du dépôt correspondent à des impls absentes des tables de dispatch ; la passe complète ne les régénère pas. Laissés en place. Tri planifié dans l'étape 4 : inventaire et proposition de suppression, rien supprimé sans validation de Benoît.
- Trois lignes de dispatch gfx906 pour poids F16 (`mmvq_f16_q8_1`, `mmq_f16_q8_1`, `mmq_f16_tile`) n'ont aucun sweep qui écrive leur certificat sous ce nom. À réaligner après le premier `--all-dispatch` (règle 2).
- CUDA sans HIP n'est pas une configuration prise en charge : le chemin forward est conditionné par HIP jusque dans les bibliothèques de modèles (signalé par Opus, non traité).
- Écart CUDA sur RTX 3090 (−38 % sans MTP, −47 % en MTP-4) : devenu l'étape 5. **Correction (lecture du code par Opus, coord/r5-chemin-cuda.md) : qwen35 ne passe jamais par le graphe CUDA** (l'append KV refuse la capture, standard_attn.rs:668/698 ; les variantes _dpos ne servent qu'à gemma4). 1554 noyaux lancés un par un par token, 1 DtoH de 970 Ko en mémoire paginée par token. Trace r5 : effet ≤ 5 %, devient le levier 3.
- Chemin ordonnanceur (utilisé quand `--mtp 0`), défauts préexistants à traiter à l'étape 4 : le flux émet la séquence d'arrêt en entier, et `completion_tokens` diffère de 1 entre streamé et non streamé.
- Les binaires llama.cpp de référence (`build-hip`, `build-cuda`) sont au commit `ac931b5bc` (10/11), pas au `ffc87c3a7` du dépôt source.
- MTP : sur « story », flambeau est derrière llama.cpp (24,3 contre 28,6 t/s) avec une acceptation de 31,8 % contre 40,8 %. Textes différents, donc pas directement comparable, mais c'est l'invite à regarder en premier.
- Fidélité contre llama.cpp : **close**. Écarts ≤ 0,2 nat sauf 3 positions isolées à ≈ 1,1 nat ; perplexité flambeau 7,76 contre 7,83 pour llama.cpp (`fidelite/VERDICT.md`). Aucun correctif.
- Mon premier contrôle de fidélité de R10b ne testait rien : t0 sort du préremplissage, qui n'utilise pas les noyaux modifiés. Refait sur les positions décodées.
- Chemin de décodage à sortie F16 hors dispatch (table codée en dur dans `ops/launch/qmatmul.rs`) : ≈ 3,3 ms par token et par GPU, non touché par R10a ni R10b.
- Les traces rocprofv3 du Vibe ne couvrent que ≈ 66 à 70 tokens même pour 130 à 200 demandés : suffisant pour l'analyse, cause non cherchée.
- `stash@{0}` « lot R3-R8 (impl) » dans le dépôt : sauvegarde d'Opus, à supprimer une fois tout commité.
- `tests/mixed_batch_microbench.rs` ne compile plus (argument `mtp_max_draft` manquant), antérieur au chantier.
- Le MTP n'est pas actif sur le chemin streamé (SSE).
