# qwen35_9b_q4_1 — V1-BENCH-S1 sweep

GGUF: `/artefact/models/Qwen3.5-9B-Q4_1.gguf`  
Rig: 4× MI50 PCIe 3.0 x16, 100 W cap, ROCm 7.1.1; tp4 omitted (rig {2,3} BAR1 fault per project_rig_gpu23_link_fault). Hybrid pp2tp2 uses devices 0,2,1,3 to keep {2,3} out of any TP group.

Prefill lengths: [8, 32, 128, 512, 2048]  
Decode lengths: [16, 64, 256]

```
topology         pp tp  load_s  pp8     pp32    pp128   pp512   pp2048  tg16    tg64    tg256   
mesh1             1  1    5.20     67.0    65.2   498.3   609.1   586.0    50.4    49.6    48.3 
pp2               2  1    1.97     73.1    67.0   504.1   616.1   589.5    60.0    59.2    56.0 
tp2               1  2    2.30    117.4   120.1   730.8   951.7   998.8    71.2    69.1    65.0 
pp4               4  1    2.11     82.9    70.2   520.2   627.6   596.0    54.8    53.9    50.5 
pp2tp2            2  2    3.30    123.0   122.7   702.9   914.5   967.0    69.7    67.7    62.9 
```
