# V2.26.a-i8-redux — finer ubatch sweep

Post-V2.27.d guard sweep across ub ∈ {128, 144, 176, 256, 320, 512}.
Many configs correctly rejected by the GDN tail-race guard (good —
would have silently produced wrong results before V2.27.d).

| ubatch | lanes | L=1024 | L=2048 | L=4096 | guard verdict |
|---:|---:|---:|---:|---:|---|
| **128** | 2 | **1833** | **2093** | **2016** | safe (L divisible) |
| 144 | 2 | — | — | — | **rejected** (L=512 tail=80) |
| 176 | 2 | 1630 | — | — | **rejected at L=2048** (tail=112) |
| 256 | 2 | 1417 | 1802 | 1863 | safe |
| 320 | 2 | — | — | — | **rejected** (L=1024 tail=64) |
| 512 | 2 | 863 | 1298 | 1491 | safe |

**Verdict: ub=128 is the plateau optimum.** Smaller ubatches would
theoretically fill the 1F1B pipeline faster, but they're either
forbidden by the GDN race guard (hybrid models) or produce diminishing
returns because per-ubatch infrastructure overhead (peer-copy, Mutex
lock, aux-stream dispatch) rises.

V2.26.a-i7's `ub=128 u_lanes=2` recommendation stands unchanged.
Result is within noise of V2.26.a-i7's snapshot (1988 → 2016 at
L=4096 = +1.4 %, within run-to-run variance).

No code change; closing i8 empirically.
