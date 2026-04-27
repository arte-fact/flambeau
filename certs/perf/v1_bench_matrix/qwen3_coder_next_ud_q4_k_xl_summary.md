# qwen3_coder_next_ud_q4_k_xl — V1-BENCH-S1 sweep

GGUF: `/artefact/models/Qwen3-Coder-Next-UD-Q4_K_XL.gguf`  
Rig: 4× MI50 PCIe 3.0 x16, 100 W cap, ROCm 7.1.1; tp4 omitted (rig {2,3} BAR1 fault per project_rig_gpu23_link_fault). Hybrid pp2tp2 uses devices 0,2,1,3 to keep {2,3} out of any TP group.

Prefill lengths: [8, 128, 512]  
Decode lengths: [64]

```
topology         pp tp  load_s  pp8     pp128   pp512   tg64    
```
