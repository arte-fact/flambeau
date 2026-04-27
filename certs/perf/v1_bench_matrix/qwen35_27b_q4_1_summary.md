# qwen35_27b_q4_1 — V1-BENCH-S1 sweep

GGUF: `/artefact/models/Qwen3.5-27B-Q4_1.gguf`  
Rig: 4× MI50 PCIe 3.0 x16, 100 W cap, ROCm 7.1.1; tp4 omitted (rig {2,3} BAR1 fault per project_rig_gpu23_link_fault). Hybrid pp2tp2 uses devices 0,2,1,3 to keep {2,3} out of any TP group.

Prefill lengths: [8, 32, 128, 512, 2048]  
Decode lengths: [16, 64, 256]

```
topology         pp tp  load_s  pp8     pp32    pp128   pp512   pp2048  tg16    tg64    tg256   
pp4               4  1   14.84     20.4    19.1   170.9   188.4   181.2    20.8    20.7    20.0 
pp2tp2            2  2    7.08     36.6    36.6   253.0   314.6   320.5    31.3    30.7    28.8 
```
