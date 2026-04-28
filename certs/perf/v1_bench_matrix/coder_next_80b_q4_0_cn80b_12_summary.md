# coder_next_80b_q4_0_cn80b_12 — V1-BENCH-S1 sweep

GGUF: `/artefact/models/Qwen3-Coder-Next-Q4_0.gguf`  
Rig: 4× MI50 PCIe 3.0 x16, 100 W cap, ROCm 7.1.1; tp4 omitted (rig {2,3} BAR1 fault per project_rig_gpu23_link_fault). Hybrid pp2tp2 uses devices 0,2,1,3 to keep {2,3} out of any TP group.

Prefill lengths: [128, 512, 2048, 4096]  
Decode lengths: [64, 128, 256]

```
topology         pp tp  load_s  pp128   pp512   pp2048  pp4096  tg64    tg128   tg256   
pp2tp2            2  2  102.23    519.7   799.3   908.3   867.9    46.4    45.3    43.3 
```
