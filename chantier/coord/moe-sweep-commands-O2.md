# Sweeps MoE MMVQ / gate_up — commandes exactes (binaires O2, HEAD 0b782475)
Répertoire de sortie : chaque sweep écrit `./certs/<backend>/<arch>/<impl_id>.json` **relativement au répertoire courant** (`repo_root()` = `current_dir()`, cli/src/main.rs:738 ; `Cert::write_to_disk`, bench/src/cert.rs). Lancer depuis la racine de la worktree (`cd ~/Workspace/flambeau-impl-o`), sinon les certificats atterrissent ailleurs. Une commande = un certificat (nom = `impl_id`, sans suffixe d'arch). Ces 20 sweeps sont les seuls touchés par 25c926d7 (`run_moe_mmvq_sweep` / `run_moe_gate_up_sweep`, 256 experts).
Sur gfx906, chaque sweep était précédé d'une garde : attendre que la température edge du GPU 1 soit ≤ 50 °C (`rocm-smi -d 1 --showtemp`). Script utilisé : `moe_sweeps.sh` (scratchpad), timeout 900 s par sweep.

## gfx906 (lancé, 20/20 PASS)

```sh
cd ~/Workspace/flambeau-impl-o
HIP_VISIBLE_DEVICES=1 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2 sweep --arch gfx906 --op qmatmul --dtype Q8_0_moe_mmvq   # -> ./certs/hip/gfx906/indexed_moe_mmvq_q8_0.json
HIP_VISIBLE_DEVICES=1 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2 sweep --arch gfx906 --op qmatmul --dtype Q4_0_moe_mmvq   # -> ./certs/hip/gfx906/indexed_moe_mmvq_q4_0.json
HIP_VISIBLE_DEVICES=1 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2 sweep --arch gfx906 --op qmatmul --dtype Q4_1_moe_mmvq   # -> ./certs/hip/gfx906/indexed_moe_mmvq_q4_1.json
HIP_VISIBLE_DEVICES=1 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2 sweep --arch gfx906 --op qmatmul --dtype Q4_K_r2_moe_mmvq   # -> ./certs/hip/gfx906/indexed_moe_mmvq_q4_k_r2.json
HIP_VISIBLE_DEVICES=1 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2 sweep --arch gfx906 --op qmatmul --dtype Q4_K_r4_moe_mmvq   # -> ./certs/hip/gfx906/indexed_moe_mmvq_q4_k_r4.json
HIP_VISIBLE_DEVICES=1 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2 sweep --arch gfx906 --op qmatmul --dtype Q5_K_r2_moe_mmvq   # -> ./certs/hip/gfx906/indexed_moe_mmvq_q5_k_r2.json
HIP_VISIBLE_DEVICES=1 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2 sweep --arch gfx906 --op qmatmul --dtype Q6_K_r2_moe_mmvq   # -> ./certs/hip/gfx906/indexed_moe_mmvq_q6_k_r2.json
HIP_VISIBLE_DEVICES=1 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2 sweep --arch gfx906 --op qmatmul --dtype Q3_K_r2_moe_mmvq   # -> ./certs/hip/gfx906/indexed_moe_mmvq_q3_k_r2.json
HIP_VISIBLE_DEVICES=1 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2 sweep --arch gfx906 --op qmatmul --dtype IQ4_NL_r2_moe_mmvq   # -> ./certs/hip/gfx906/indexed_moe_mmvq_iq4_nl_r2.json
HIP_VISIBLE_DEVICES=1 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2 sweep --arch gfx906 --op qmatmul --dtype IQ4_XS_r2_moe_mmvq   # -> ./certs/hip/gfx906/indexed_moe_mmvq_iq4_xs_r2.json
HIP_VISIBLE_DEVICES=1 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2 sweep --arch gfx906 --op qmatmul --dtype IQ2_XXS_r2_moe_mmvq   # -> ./certs/hip/gfx906/indexed_moe_mmvq_iq2_xxs_r2.json
HIP_VISIBLE_DEVICES=1 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2 sweep --arch gfx906 --op qmatmul --dtype IQ2_XS_r2_moe_mmvq   # -> ./certs/hip/gfx906/indexed_moe_mmvq_iq2_xs_r2.json
HIP_VISIBLE_DEVICES=1 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2 sweep --arch gfx906 --op qmatmul --dtype IQ2_S_r2_moe_mmvq   # -> ./certs/hip/gfx906/indexed_moe_mmvq_iq2_s_r2.json
HIP_VISIBLE_DEVICES=1 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2 sweep --arch gfx906 --op qmatmul --dtype IQ3_XXS_r2_moe_mmvq   # -> ./certs/hip/gfx906/indexed_moe_mmvq_iq3_xxs_r2.json
HIP_VISIBLE_DEVICES=1 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2 sweep --arch gfx906 --op qmatmul --dtype IQ3_S_r2_moe_mmvq   # -> ./certs/hip/gfx906/indexed_moe_mmvq_iq3_s_r2.json
HIP_VISIBLE_DEVICES=1 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2 sweep --arch gfx906 --op qmatmul --dtype IQ1_S_r2_moe_mmvq   # -> ./certs/hip/gfx906/indexed_moe_mmvq_iq1_s_r2.json
HIP_VISIBLE_DEVICES=1 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2 sweep --arch gfx906 --op qmatmul --dtype IQ1_M_r2_moe_mmvq   # -> ./certs/hip/gfx906/indexed_moe_mmvq_iq1_m_r2.json
HIP_VISIBLE_DEVICES=1 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2 sweep --arch gfx906 --op qmatmul --dtype Q8_0_gate_up_moe_mmvq   # -> ./certs/hip/gfx906/indexed_moe_mmvq_q8_0_gate_up.json
HIP_VISIBLE_DEVICES=1 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2 sweep --arch gfx906 --op qmatmul --dtype Q4_0_gate_up_moe_mmvq   # -> ./certs/hip/gfx906/indexed_moe_mmvq_q4_0_gate_up.json
HIP_VISIBLE_DEVICES=1 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2 sweep --arch gfx906 --op qmatmul --dtype Q4_K_gate_up_r4_moe_mmvq   # -> ./certs/hip/gfx906/indexed_moe_mmvq_q4_k_gate_up_r4.json
```

## sm_86 (équivalents, non lancés)

```sh
cd ~/Workspace/flambeau-impl-o
CUDA_VISIBLE_DEVICES=0 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2_CUDA sweep --arch sm_86 --op qmatmul --dtype Q8_0_moe_mmvq   # -> ./certs/cuda/sm_86/indexed_moe_mmvq_q8_0.json
CUDA_VISIBLE_DEVICES=0 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2_CUDA sweep --arch sm_86 --op qmatmul --dtype Q4_0_moe_mmvq   # -> ./certs/cuda/sm_86/indexed_moe_mmvq_q4_0.json
CUDA_VISIBLE_DEVICES=0 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2_CUDA sweep --arch sm_86 --op qmatmul --dtype Q4_1_moe_mmvq   # -> ./certs/cuda/sm_86/indexed_moe_mmvq_q4_1.json
CUDA_VISIBLE_DEVICES=0 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2_CUDA sweep --arch sm_86 --op qmatmul --dtype Q4_K_r2_moe_mmvq   # -> ./certs/cuda/sm_86/indexed_moe_mmvq_q4_k_r2.json
CUDA_VISIBLE_DEVICES=0 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2_CUDA sweep --arch sm_86 --op qmatmul --dtype Q4_K_r4_moe_mmvq   # -> ./certs/cuda/sm_86/indexed_moe_mmvq_q4_k_r4.json
CUDA_VISIBLE_DEVICES=0 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2_CUDA sweep --arch sm_86 --op qmatmul --dtype Q5_K_r2_moe_mmvq   # -> ./certs/cuda/sm_86/indexed_moe_mmvq_q5_k_r2.json
CUDA_VISIBLE_DEVICES=0 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2_CUDA sweep --arch sm_86 --op qmatmul --dtype Q6_K_r2_moe_mmvq   # -> ./certs/cuda/sm_86/indexed_moe_mmvq_q6_k_r2.json
CUDA_VISIBLE_DEVICES=0 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2_CUDA sweep --arch sm_86 --op qmatmul --dtype Q3_K_r2_moe_mmvq   # -> ./certs/cuda/sm_86/indexed_moe_mmvq_q3_k_r2.json
CUDA_VISIBLE_DEVICES=0 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2_CUDA sweep --arch sm_86 --op qmatmul --dtype IQ4_NL_r2_moe_mmvq   # -> ./certs/cuda/sm_86/indexed_moe_mmvq_iq4_nl_r2.json
CUDA_VISIBLE_DEVICES=0 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2_CUDA sweep --arch sm_86 --op qmatmul --dtype IQ4_XS_r2_moe_mmvq   # -> ./certs/cuda/sm_86/indexed_moe_mmvq_iq4_xs_r2.json
CUDA_VISIBLE_DEVICES=0 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2_CUDA sweep --arch sm_86 --op qmatmul --dtype IQ2_XXS_r2_moe_mmvq   # -> ./certs/cuda/sm_86/indexed_moe_mmvq_iq2_xxs_r2.json
CUDA_VISIBLE_DEVICES=0 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2_CUDA sweep --arch sm_86 --op qmatmul --dtype IQ2_XS_r2_moe_mmvq   # -> ./certs/cuda/sm_86/indexed_moe_mmvq_iq2_xs_r2.json
CUDA_VISIBLE_DEVICES=0 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2_CUDA sweep --arch sm_86 --op qmatmul --dtype IQ2_S_r2_moe_mmvq   # -> ./certs/cuda/sm_86/indexed_moe_mmvq_iq2_s_r2.json
CUDA_VISIBLE_DEVICES=0 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2_CUDA sweep --arch sm_86 --op qmatmul --dtype IQ3_XXS_r2_moe_mmvq   # -> ./certs/cuda/sm_86/indexed_moe_mmvq_iq3_xxs_r2.json
CUDA_VISIBLE_DEVICES=0 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2_CUDA sweep --arch sm_86 --op qmatmul --dtype IQ3_S_r2_moe_mmvq   # -> ./certs/cuda/sm_86/indexed_moe_mmvq_iq3_s_r2.json
CUDA_VISIBLE_DEVICES=0 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2_CUDA sweep --arch sm_86 --op qmatmul --dtype IQ1_S_r2_moe_mmvq   # -> ./certs/cuda/sm_86/indexed_moe_mmvq_iq1_s_r2.json
CUDA_VISIBLE_DEVICES=0 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2_CUDA sweep --arch sm_86 --op qmatmul --dtype IQ1_M_r2_moe_mmvq   # -> ./certs/cuda/sm_86/indexed_moe_mmvq_iq1_m_r2.json
CUDA_VISIBLE_DEVICES=0 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2_CUDA sweep --arch sm_86 --op qmatmul --dtype Q8_0_gate_up_moe_mmvq   # -> ./certs/cuda/sm_86/indexed_moe_mmvq_q8_0_gate_up.json
CUDA_VISIBLE_DEVICES=0 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2_CUDA sweep --arch sm_86 --op qmatmul --dtype Q4_0_gate_up_moe_mmvq   # -> ./certs/cuda/sm_86/indexed_moe_mmvq_q4_0_gate_up.json
CUDA_VISIBLE_DEVICES=0 timeout 900 ~/Workspace/flambeau-exp/impl/bin_O2_CUDA sweep --arch sm_86 --op qmatmul --dtype Q4_K_gate_up_r4_moe_mmvq   # -> ./certs/cuda/sm_86/indexed_moe_mmvq_q4_k_gate_up_r4.json
```

## Certificats MoE de certs/cuda/sm_86 hors de ces 20 commandes

Aucun certificat MoE de `certs/cuda/sm_86` n'échappe au binaire CUDA. Les 36 suivants ne sont pas écrits par les 20 commandes ci-dessus et ne sont pas touchés par 25c926d7. bin_O2_CUDA sait quand même les régénérer :
- 31 certificats tile8, `indexed_moe_mmq_<dtype>_{gate_up,down}_tile8.json` (q8_0, q4_0, q4_k, q5_k, q6_k, q3_k et 9 IQ ×2, plus q4_1_down) : `--op qmatmul --dtype <D>_{gate_up,down}_tile8_moe_mmq`, par exemple `Q4_K_down_tile8_moe_mmq` (cli/src/main.rs:1482-1512).
- 5 certificats de routage et combinaison, `moe_combine_f16`, `moe_combine_no_residual_f16`, `moe_combine_no_residual_f32`, `moe_combine_two_residuals_f16` et `moe_sort_by_expert` : `sweep --arch sm_86 --op <nom>`, via `sweep_pointwise_cb`.
