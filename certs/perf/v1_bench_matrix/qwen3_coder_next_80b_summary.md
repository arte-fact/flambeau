# qwen3_coder_next_80b — V1-BENCH-S1 sweep

GGUF: `/artefact/models/Qwen3-Coder-Next-Q4_0.gguf`  
Rig: 4× MI50 PCIe 3.0 x16, 100 W cap, ROCm 7.1.1; tp4 omitted (rig {2,3} BAR1 fault per project_rig_gpu23_link_fault). Hybrid pp2tp2 uses devices 0,2,1,3 to keep {2,3} out of any TP group.

Prefill lengths: [128, 512, 2048]  
Decode lengths: [64]

```
topology         pp tp  load_s  pp128   pp512   pp2048  tg64    
pp4               4  1   83.49    426.4   537.6   555.6    41.3 
pp2tp2            2  2  101.88    164.9   179.9   181.9    46.2 
```
