# V2.26 head-to-head: flambeau vs llama.cpp / llamacpp-turbo on gfx906

End-of-V2.26 status check across the available models on the 4× MI50
rig. Two questions answered:

1. Where does flambeau stand vs upstream after the V2.26 cycle's 2–3×
   async-prefill win?
2. What's the next optimization lever?

## Compatibility matrix

| target | flambeau | llamacpp-turbo | stock llama.cpp |
|---|:---:|:---:|:---:|
| Qwen3.5-9B-Q4_1 (hybrid dense) | ✓ | ✓ | ✗ (rocblas 7.2.1 mul_mat crash) |
| Qwen3.6-35B-A3B-UD-Q4_K_S (MoE) | ✓ | ✗ (load fail) | ✗ (load fail) |
| Qwen3.6-35B-A3B-Q4_0 (MoE) | — | ✗ (load fail) | — |
| Qwen3.6-27B-Q8_0 (dense) | — (no loader) | — (not tested) | — |

Qwen3-family MoE on gfx906 lands only on flambeau — both upstreams
fail to load our gguf files (the memory's V2.3.c.1 note matches:
"llama.cpp prefill genuinely crashes on Qwen3 MoE+SSM on gfx906 per
upstream bugs"). Head-to-head numbers therefore only cover 9B.

## 9B Q4_1 Mesh<4> — the comparable configuration

| phase | llamacpp-turbo | flambeau (best cfg) | flambeau Δ |
|---|---:|---:|---:|
| pp1024 | 1013 tok/s | **1839** (ub=128 lanes=2, async legacy or graph) | **+81 %** |
| pp4096 | 963 tok/s | **1988** (ub=128 lanes=2) | **+106 %** |
| pp4096 best | 963 | **2065** (ub=128 lanes=3) | **+115 %** |
| tg64 | 73.5 tok/s | **53** (sync single-stream) | **−28 %** |

flambeau prefill is now 1.8× to 2.1× ahead of turbo on the only
config they both run. Decode still trails by 28 %.

### Why the split

**Prefill wins** came from V2.26.a's cross-lane barrier removals
(V2.26.a-i5a `upload_positions_range` sync + V2.26.a-i7b batched
embed). Those fixes amortise across the aux-stream 1F1B pipeline,
which turbo doesn't appear to run — its prefill path is effectively
single-stream per rank.

**Decode trails** because the single-token path doesn't use aux
streams at all — there's nothing for V2.26.a's async-barrier
removals to amortise. The 73.5 vs 53 gap is at the per-token kernel
level.

## 35B MoE-only numbers — flambeau standalone

| config | prefill pp1024 | decode tg64 | load |
|---|---:|---:|---:|
| Qwen3.6-35B-A3B-UD-Q4_K_S Mesh<4> sync | 716 tok/s | 52 tok/s | 5.7 s |

Mesh<4> sync path only — the 35B bench harness
(`tests/perf_baseline_qwen3_moe.rs`) doesn't yet have
FLAMBEAU_ASYNC_UBATCH plumbing, so the V2.26.a async win isn't
exercised here. Extending the harness to the async path is ~50 lines
of test-side work and should lift 35B prefill towards the same
~2× improvement we saw on 9B (V2.26.a fixes are model-agnostic —
they're at the forward-driver layer, not per-kernel).

Cert historical check — V2.8 numbers (`certs/perf/qwen3_6_35b_a3b_ud_q4_k_s_mesh4.json`)
had pp=1024 ~640 tok/s on Mesh<4>. Now 716 at the same config:
+12 %. Incidental drift from kernel-side tuning accumulated over
V2.9–V2.26.

## Next optimization levers — ordered by expected ROI

### 1. Decode path — where the 28 % gap to turbo lives

Decode issues ~tens of kernels per layer × 36 layers = ~1000 kernel
launches per decoded token. On the single-stream decode path,
per-launch overhead directly adds to wall time. Candidates:

- **Graph capture for decode**: the one workload where graph capture
  actually fits. Single capture at session init (or first token),
  replay thousands of times with updated `pos` slot (already
  wired) + token-id slot (needs i5b4-style work). Expected 10–25 %
  decode win; could close most of the gap to turbo.
- **Fused decode kernels that haven't been ported yet**: check
  `candle-hip-kernels` for D1/D2/D3/D4 that are still null — each
  fusion kills a launch.

### 2. Qwen3.6-35B MoE async harness

Fastest way to lift 35B prefill into the post-V2.26.a regime is to
port the FLAMBEAU_ASYNC_UBATCH / FLAMBEAU_UBATCH / FLAMBEAU_U_LANES
env-driver code from `perf_baseline_qwen35_9b.rs` into
`perf_baseline_qwen3_moe.rs`, re-cert. No kernel work needed —
infrastructure change on the test side. Expected gain:
~1.8×–2× on 35B prefill Mesh<4>.

### 3. 27B dense Qwen3.6 loader

Not yet loadable in flambeau — needs a new arch dispatch for
qwen3.6-27B-Q8_0 / UD-Q8_K_XL. Dense 27B is a clean single-card
target (fits 16 GB at Q4/Q5, needs 2× at Q8). Scope: ~V2.2-sized
loader work but on a new arch (probably reuses qwen35 path with
different config defaults — same hybrid? Need to check).

### 4. Remaining kernel headroom on 9B

With async saturated, the 9B prefill is now compute-bound (2065
tok/s at L=4096 ≈ 1000 ms of kernel work on 4 ranks). Further gains
would need per-kernel work. Not obvious where — we already beat
turbo by 2× on this config. Probably diminishing returns.

## Known bugs (filed, unrelated to head-to-head)

- **ubatch=64 async parity failure** on 9B Q4_1 at L ∈ {1024, 4096}.
  `last_id=96519 / 96128 / 82` vs expected `220 / 62 / 248046`. Small
  ubatch size mishandled in the async loop.
- **ubatch=192 async parity failure** on 9B Q4_1 when ubatch doesn't
  divide L (tail-ubatch bug).
- **Qwen3-family MoE GGUF upstream load failures** on turbo + stock
  llama.cpp on gfx906. Not our problem; matches earlier memory.

## Regeneration

```
# turbo 9B
export LD_LIBRARY_PATH=/opt/rocm-7.1.1/core-7.13/lib:/opt/rocm-7.1.1/lib:$LD_LIBRARY_PATH
/artefact/llamacpp-turbo/llama-cpp-gfx906-turbo/build/bin/llama-bench \
  -m /artefact/models/Qwen3.5-9B-Q4_1.gguf -ngl 99 -sm layer -mg 0 -ts 1,1,1,1 \
  -p 1024,4096 -n 64 -r 2

# flambeau 9B best async
FLAMBEAU_ASYNC_UBATCH=1 FLAMBEAU_UBATCH=128 FLAMBEAU_U_LANES=2 \
  FLAMBEAU_MESH_RANKS=4 FLAMBEAU_QWEN35_GGUF=/artefact/models/Qwen3.5-9B-Q4_1.gguf \
  ./target/release/deps/perf_baseline_qwen35_9b-* perf_baseline_qwen35_9b --nocapture

# flambeau 35B sync (async not yet plumbed in this harness)
FLAMBEAU_MESH_RANKS=4 FLAMBEAU_QWEN3_GGUF=/artefact/models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf \
  FLAMBEAU_TG_LEN=64 \
  ./target/release/deps/perf_baseline_qwen3_moe-* perf_baseline_qwen3_moe --nocapture
```
