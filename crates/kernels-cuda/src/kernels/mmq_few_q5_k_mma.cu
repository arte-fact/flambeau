// Few-column Q5_K MMQ on sm_80+ int8 tensor cores (mmq_few_mma.cuh).
// Entries c1_t1, c2_t1, c3_t1: 1..3 groups of 8 columns.

#include "mmq_few_mma.cuh"

FLAMBEAU_MMQ_FEW_ENTRY(q5_k, mmq_few::DecQ45K<true>, 1, 1)
FLAMBEAU_MMQ_FEW_ENTRY(q5_k, mmq_few::DecQ45K<true>, 2, 1)
FLAMBEAU_MMQ_FEW_ENTRY(q5_k, mmq_few::DecQ45K<true>, 3, 1)
