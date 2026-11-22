// Few-column Q4_K MMQ on sm_80+ int8 tensor cores (mmq_few_mma.cuh).
// Entries c1_t1, c2_t1, c3_t1: 1..3 groups of 8 columns. The 32-row tile
// (NCG 2, NMT 2) is not instantiated: it spills 20 B at 255 registers where
// llama.cpp's fits in 252.

#include "mmq_few_mma.cuh"

FLAMBEAU_MMQ_FEW_ENTRY(q4_k, mmq_few::DecQ45K<false>, 1, 1)
FLAMBEAU_MMQ_FEW_ENTRY(q4_k, mmq_few::DecQ45K<false>, 2, 1)
FLAMBEAU_MMQ_FEW_ENTRY(q4_k, mmq_few::DecQ45K<false>, 3, 1)
