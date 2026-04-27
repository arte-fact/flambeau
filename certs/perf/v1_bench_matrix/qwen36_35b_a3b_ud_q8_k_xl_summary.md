# qwen36_35b_a3b_ud_q8_k_xl — V1-BENCH-S1 sweep

GGUF: `/artefact/models/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf`  
Rig: 4× MI50 PCIe 3.0 x16, 100 W cap, ROCm 7.1.1; tp4 omitted (rig {2,3} BAR1 fault per project_rig_gpu23_link_fault). Hybrid pp2tp2 uses devices 0,2,1,3 to keep {2,3} out of any TP group.

Prefill lengths: [8, 32, 128, 512, 2048]  
Decode lengths: [16, 64, 256]

```
topology         pp tp  load_s  pp8     pp32    pp128   pp512   pp2048  tg16    tg64    tg256   
pp4               4  1   50.60     78.5    96.1   529.3   651.7   647.0    61.2    59.6    55.4 
```
