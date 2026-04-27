# qwen3_coder_next_q4_0 — V1-BENCH-S1 sweep

GGUF: `/artefact/models/Qwen3-Coder-Next-Q4_0.gguf`  
Rig: 4× MI50 PCIe 3.0 x16, 100 W cap, ROCm 7.1.1; tp4 omitted (rig {2,3} BAR1 fault per project_rig_gpu23_link_fault). Hybrid pp2tp2 uses devices 0,2,1,3 to keep {2,3} out of any TP group.

Prefill lengths: [8, 32, 128, 512, 2048]  
Decode lengths: [16, 64, 256]

```
topology         pp tp  load_s  pp8     pp32    pp128   pp512   pp2048  tg16    tg64    tg256   
pp4               4  1   83.01     69.9   149.3   426.1   537.6   560.1    40.9    40.4    37.9 
```
