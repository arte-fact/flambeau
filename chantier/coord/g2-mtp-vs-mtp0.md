# g2 — mtp 0 contre MTP-4 en glouton pur (g1/lp_g1.json contre lp_g2.json, plus lp_h1/lp_h2 ; sans GPU)

1. **Oui, ça diverge encore, mais peu sur la RTX.**
   - RTX : code, math et chat sont identiques. story diverge au caractère 666 sur 838, list au caractère 771 sur 913 (« to redirect » contre « to direct »), donc tard, vers le 150ᵉ token.
   - MI50 TP-2 : 4 invites sur 5 divergent (math au caractère 777, chat 847, story 89, list 61), et code est identique.
   - Chaque divergence est un choix de mot quasi équivalent : « shadiest » contre « darkest », « gray void » contre « night », « redirect » contre « direct ». Aucun texte n'est dégradé, et la suite reste cohérente.
2. **La sémantique de vérification est correcte (lecture de qwen35-v2/src/model.rs:228-240).**
   - En glouton, le brouillon i est accepté si et seulement si l'argmax de la ligne cible i est égal au token brouillon.
   - Le token émis au refus, ou en bonus, est l'argmax de la ligne qui suit le dernier accepté.
   - Chaque token émis est donc l'argmax de la cible à sa position. Il n'y a ni pénalité ni biais en jeu (repetition_penalty 1,0).
3. **La cause est donc le calcul batché, ce qui est attendu et non bloquant.**
   - En MTP-4, les logits cibles viennent de la vérification à m = k + 1 = 5 lignes : MMVQ batché par tuiles de lignes, attention sur plusieurs lignes, GDN rejoué, tête LM batchée.
   - En mtp 0, ils viennent du chemin à m = 1. L'ordre des sommes diffère, et un quasi-ex-æquo bascule.
4. **Quatre indices concordent :**
   - Les divergences sont tardives, et les MI50 en ont davantage, ce qui est cohérent avec la réduction TP-2 qui s'y ajoute.
   - G2 et G4 sont identiques sur 5 invites : MTP-4 est déterministe.
   - Les mtp 0 RTX et MI50 divergent entre eux de la même façon : math au caractère 253, chat 590, story 215, list 771, le même point que RTX mtp 0 contre MTP-4. Il y a donc des quasi-ex-æquo dans ces textes, indépendamment de MTP.
   - L'acceptation est de 59,6 % sur la RTX, exactement celle de llama.cpp. Un écart de sémantique, comme une position décalée ou un KV faux, la ferait chuter et ferait diverger tôt et partout.
5. **Preuve directe possible plus tard (GPU, 1 run) :** un mtp 0 avec logprobs (`lp_g.py … 1`) donnerait l'écart entre le top-1 et le top-2 aux points de divergence. S'il est inférieur à environ 0,05 nat, c'est confirmé. Pas de code à changer.
