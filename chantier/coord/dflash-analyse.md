# DFlash2 dans flambeau — note d'analyse (étape 6, avant toute implémentation)

Sources : coord/dflash-etude-llamacpp.md (llama.cpp ffc87c3a7, lecture de code) ; coord/dflash-inventaire-flambeau.md (Opus, fichier:ligne). Aucune mesure DFlash n'existe encore sur notre matériel.

## Ce que c'est
Un drafter séparé de 5 couches (Q8_0, 2,06 Go) émet un bloc de 7 tokens en UNE passe non causale. Il est conditionné par l'entrée de 5 couches de la cible (6/20/34/48/62), injectée dans son propre KV à chaque token décodé par la cible. DFlash2 ajoute une conv locale à 2 prises et un sélecteur bas rang (top-16 par position, treillis parcouru sur l'hôte). Le drafter emprunte token_embd et la lm_head de la cible.

## Pourquoi ça pourrait battre MTP-4
- MTP-4 aujourd'hui : 3,36 tokens par cycle, 4 passes de brouillon séquentielles (chacune avec la lm_head complète, ≈ 4,8 ms par cycle sur RTX), puis une vérification à m = 5.
- DFlash2 : une seule passe de brouillon à m = 8, longueur acceptée ≥ 4,54 (plancher des tests nocturnes llama.cpp, Qwen3.8-27B Q4_K_M, n_max 7), puis une vérification à m = 8.
- Coût brut du brouillon par cycle : ≈ 1,77 Go de Q8_0 + lm_head 1,04 Go à m = 8, soit ≈ 2,8 Go lus contre ≈ 4 × 1,04 Go + la couche nextn pour MTP-4. **Aucune projection de débit tant que ce n'est pas mesuré** (règle 7).

## Ce qui manque côté flambeau (inventaire Opus)
1. Prises des entrées de 5 couches, toutes lignes, vers un tampon [n, 25 600] F16. Faciles en TP-2 ; en PP, il faudrait du pair-à-pair. Le MTP ne capture que l'état final.
2. Cycle : on réutilise la vérification, l'argmax GPU, l'acceptation et le rejeu GDN (model.rs:210-273). Il faut extraire « vérifier, accepter, rejouer » en fonction commune (règle 14 : MTP + DFlash = 2 sites, à factoriser avant le 3e).
3. Ops :
   - attention non causale : a priori par l'offset de q sur flash_tile, à certifier ;
   - nouveaux noyaux : conv dynamique groupée (petit) et score de paire du sélecteur (16 × 16 × 256) ;
   - top-16 : topk_softmax_f32 existant.
4. Loader d'un second GGUF (arch dflash, crate modèle) et `--draft-model` côté serveur (règles 12 et 13 : hook générique, pas de branche d'arch dans le serveur).
5. Mémoire :
   - MI50 TP-2 : tient largement ;
   - RTX 24 Go : cible 16,46 Go + drafter 2,06 Go + KV + pool. À relever avec nvidia-smi pendant un serve avant de conclure.
6. m = 8 ou 9 est couvert par les noyaux sur les deux arch (batched sm_86, lds_tiled x8 gfx906), mais non mesuré.

## Ordre proposé (étapes, pas de dates)
1. **Base mesurée** : llama.cpp en draft-dflash2 contre draft-mtp, sur RTX et MI50, même protocole lp_g (t/s, acceptation, longueur acceptée). Plus le relevé mémoire RTX de flambeau en serve. Si DFlash2 ne bat pas MTP chez llama.cpp sur notre matériel, on s'arrête là (résultat nul documenté).
2. Microbench flambeau du coût d'un brouillon : 5 couches Q8_0 à m = 8 et lm_head à m = 8 sur les deux arch, avec les noyaux existants.
3. Implémentation par étapes, chacune certifiée et sans régression MTP :
   - prises de couches ;
   - loader et drafter sans sélecteur (argmax par position, DFlash « 1 ») ;
   - cycle factorisé ;
   - conv DFlash2 ;
   - sélecteur.
4. Mesure e2e contre MTP-4 et contre llama.cpp DFlash2.

## Base mesurée (dflash-base/RESULTATS.md, llama.cpp HEAD ffc87c3a7, lp_g)
- RTX : sans spéculatif 43,47 ; MTP-4 89,25 ; **DFlash2 108,98 (+22 %)**.
- MI50 : sans spéculatif 25,14 ; MTP-4 35,63 ; **DFlash2 32,74 (−8 %)**. Gain sur code et math, perte sur chat, story et list.
- Verdict provisoire : on continue sur sm_86. Sur gfx906, la suite dépend du microbench de l'étape 2 avec nos noyaux m = 8 ; sinon résultat nul pour gfx906.
