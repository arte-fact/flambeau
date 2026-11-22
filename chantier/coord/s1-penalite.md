# s1 — pénalité de répétition par défaut (lecture seule, sans GPU)

1. **Origine** : `crates/server/src/state.rs:203-219`, commit `9b3ddbba` (2026-06-02, « server: rep_penalty GGUF chain + low-temp 1.05 fallback »). La valeur se résout dans l'ordre suivant : requête, puis GGUF, puis 1,0. Mais si la température est inférieure à 0,1 et que ni la requête ni le GGUF ne fixent la pénalité, elle est forcée à 1,05. Le commentaire du code invoque une boucle « Paris. » à température 0 sur Qwen3.6-27B UD-Q3_K_XL. Le GGUF Qwen3.8-27B ne porte que `general.sampling.temp`, `top_k` et `top_p`, donc lp.py (`temperature: 0`, sans pénalité) tourne chez nous **avec une pénalité de 1,05**.
2. **Portée chez nous** : la pénalité s'applique à tout l'historique généré (`apply_penalties`, sampling.rs:488 : comptes sur l'historique entier, sans fenêtre `last_n`), et à chaque token de la cible comme du brouillon MTP.
3. **llama-server** : `penalty_repeat = 1.00f` (désactivée) et `penalty_last_n = 64` (common/common.h:246-247). Le GGUF ne porte pas `general.sampling.penalty_repeat` (llama-arch.cpp:203), et `rtx/run_rtx_llama.sh` ne passe pas `--repeat-penalty`. Dans nos comparaisons, **llama.cpp tourne donc en glouton pur, sans pénalité**.
4. **Comparabilité** : les deux serveurs ne décodent pas la même distribution. On compare un glouton pénalisé à 1,05 (flambeau) à un glouton pur (llama.cpp).
   - **Ce qui reste comparable** : le coût par token et par cycle, puisque l'application de la pénalité est négligeable (quelques centaines de paires, un noyau).
   - **Ce qui ne l'est plus** :
     - les textes diffèrent ;
     - l'acceptation MTP aussi, puisqu'elle dépend des tokens produits, donc le nombre de tokens par cycle et le t/s MTP ;
     - la longueur jusqu'à l'EOS (par exemple « code » : 203 tokens chez nous, contre 400 demandés).

   Les écarts MTP (72 contre 78 t/s, 4,4 tokens par cycle chez llama.cpp) mélangent donc coût du noyau et acceptation.
5. **Pour des mesures comparables** (à décider) :
   - soit passer `"repetition_penalty": 1.0` dans le corps de lp.py, ce qui donne un glouton pur des deux côtés et désactive notre repli 1,05, puisqu'une valeur fixée par la requête l'emporte ;
   - soit passer `--repeat-penalty 1.05 --repeat-last-n -1` à llama-server, pour une pénalité sur tout l'historique, la plus proche de notre sémantique.

   La première option ne demande aucun changement de code et donne une référence propre. Le chemin GPU de s1 couvre les deux cas, avec ou sans pénalités.
