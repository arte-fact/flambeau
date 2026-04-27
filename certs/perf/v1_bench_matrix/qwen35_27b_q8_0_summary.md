# qwen35_27b_q8_0 — V1-BENCH-S1 sweep

GGUF: `/artefact/models/Qwen3.5-27B-Q8_0.gguf`  
Rig: 4× MI50 PCIe 3.0 x16, 100 W cap, ROCm 7.1.1; tp4 omitted (rig {2,3} BAR1 fault per project_rig_gpu23_link_fault). Hybrid pp2tp2 uses devices 0,2,1,3 to keep {2,3} out of any TP group.

Prefill lengths: [8, 32, 128, 512, 2048]  
Decode lengths: [16, 64, 256]

```
topology         pp tp  load_s  pp8     pp32    pp128   pp512   pp2048  tg16    tg64    tg256   
pp4               4  1   23.02      6.9     6.8    96.7    95.9    91.6    19.3    19.0    18.8 
pp2tp2            2  2   13.83     13.0    13.2   171.6   184.9   174.0    30.0    29.3    27.6 
```
