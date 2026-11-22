// Few-column Q3_K MMQ on sm_80+ int8 tensor cores (mmq_few_mma.cuh): 110-B
// super-blocks through the unaligned ring, k = 16 MMAs (a scale per 16
// values), D4 activation. Entries c1_t1, c2_t1, c3_t1.

#include "mmq_few_mma.cuh"

FLAMBEAU_MMQ_FEW_ENTRY(q3_k, mmq_few::DecQ3K, 1, 1)
FLAMBEAU_MMQ_FEW_ENTRY(q3_k, mmq_few::DecQ3K, 2, 1)
FLAMBEAU_MMQ_FEW_ENTRY(q3_k, mmq_few::DecQ3K, 3, 1)
