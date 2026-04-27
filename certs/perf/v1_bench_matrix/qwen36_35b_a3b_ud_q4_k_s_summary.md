# qwen36_35b_a3b_ud_q4_k_s — V1-BENCH-S1 sweep

GGUF: `/artefact/models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf`  
Rig: 4× MI50 PCIe 3.0 x16, 100 W cap, ROCm 7.1.1; tp4 omitted (rig {2,3} BAR1 fault per project_rig_gpu23_link_fault). Hybrid pp2tp2 uses devices 0,2,1,3 to keep {2,3} out of any TP group.

Prefill lengths: [8, 32, 128, 512, 2048]  
Decode lengths: [16, 64, 256]

```
topology         pp tp  load_s  pp8     pp32    pp128   pp512   pp2048  tg16    tg64    tg256   
pp4               4  1   15.88     84.5    93.3   495.9   605.7   606.4    63.0    62.1    57.5 
pp2tp2            2  2   19.48    140.5   148.4   401.5   458.6   426.9    69.2    64.5    60.6 
```
