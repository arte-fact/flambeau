# V2.29.c — Q4_1 MMVQ A/B across certified variants: null (t128 stands)

Post-V2.29.a audit, swept the four existing certified Q4_1 MMVQ
variants to see if any beats the current default `t128` at the
9B Q4_1 Mesh<4> decode shape.

## Matrix

| impl_id | threads | rows/block | math | prefill L=1024 | decode tg=64 | Δ decode |
|---|---:|---:|---|---:|---:|---:|
| **qmatmul_q4_1_mmvq_t128_gfx906** (default) | 128 | 1 | DP4A | **823** | **54.97** | — |
| qmatmul_q4_1_mmvq_dp4a_gfx906 | 256 | 1 | DP4A | 364 | 51.52 | −6.3 % |
| qmatmul_q4_1_mmvq_nw1_r2_gfx906 | 64 | 2 | scalar | 820 | **24.05** | **−56 %** |
| qmatmul_q4_1_mmvq_r2_dp4a_gfx906 | 256 | 2 | DP4A | 822 | 47.17 | −14.2 % |

Parity bit-exact on every variant (decode `last_id=30`).

## Why every alternative is worse

1. **`mmvq_q4_1` (256t × 1 row, DP4A)** — 256-thread blocks halve
   occupancy vs 128t (2× fewer blocks resident). At m=1 each row
   is launched as one block; 256t blocks limit to 40 resident
   blocks on gfx906 vs 128t's 80 resident blocks. Under-utilised
   grid on small-m dispatch.
2. **`mmvq_q4_1_nw1_r2` (64t × 2 rows, scalar)** — pre-DP4A scalar
   inner loop, ~4× slower arithmetic per operand. Never a win over
   DP4A variants on gfx906; only certified as historical reference.
3. **`mmvq_q4_1_r2_dp4a` (256t × 2 rows, DP4A)** — matches V2.24.a.2's
   original null finding. Two output accumulators per thread double
   the VGPR pressure over single-row t128, pushing the compiler into
   scratch spills (measured) or lower occupancy. VGPR pressure wins
   > theoretical throughput here.
4. **`mmvq_q4_1_t128` (128t × 1 row, DP4A)** — V2.24.a.3 cert pick.
   128-thread blocks = 80 resident on gfx906, 1-row accumulator
   keeps VGPR low, DP4A extracts 4× throughput over scalar. Local
   optimum.

## Implication for the 28 % decode gap

The 9B decode gap to turbo (53 → 73.5 tok/s, Δ = 28 %) cannot be
closed by variant-swapping the existing Q4_1 MMVQ kernels. Every
certified alternative is slower.

Closing it needs a **new kernel design**. Candidates (not in this
iteration):

1. **VDR=2 port** — V1.7.6.f memory landed this for Q8_0 with a
   measurable win. VDR (vector-dot-ratio 2) loads 2 quant blocks
   per thread per iter, improving memory coalescing + hiding load
   latency behind compute. Port to Q4_1: load 2× Q4_1 blocks (each
   32 quants) per thread per step → `__builtin_amdgcn_sdot4 ×2`
   per nibble-pair. Kernel-level work; new cert required.
2. **Asymmetric fusion** — the V1.7.6.f memory also landed an
   `attn_qkv+gate` asymmetric fusion pattern. For Qwen3.5/3.6
   full-attn, the Q-proj + gate-proj share the same input and could
   fuse into one wider MMVQ with a post-split. Kernel + Rust-side
   wire work.
3. **MMQ-tile1** — MMQ usually activates at m ≥ 128, but a tile1
   variant (m=1 with MMQ-style LDS-tiled weight loads) might win
   over MMVQ by amortising weight reads across threads better.
   Research-level; unclear if anyone has shipped one.

All three are new-kernel work, scope > V2.29.c's A/B intent.

## Recommendation

- **Keep default `qmatmul_q4_1_mmvq_t128_gfx906`.** No change.
- **Update V2.29 queue:** demote V2.29.c as "null-swept, no variant
  win". Promote VDR=2 port as V2.29.f-new if continuing the decode
  track; otherwise pivot to V2.29.b (attn flash-tile) which still
  has 22.3 % of prefill wall as addressable.

## Gate

- Sync + async paths: last_id bit-exact across all four variants.
- `impls.rs` + dispatch table restored to default (t128).
- No code change committed from this iteration other than this cert.

## Regeneration

```bash
BIN=$(ls -t target/release/deps/perf_baseline_qwen35_9b-* | grep -v '\.d$' | head -1)
for V in t128 dp4a nw1_r2 r2_dp4a; do
  sed -i "s|qmatmul_q4_1_mmvq_[a-z_0-9]*_gfx906|qmatmul_q4_1_mmvq_${V}_gfx906|g" \
    crates/backend-hip/src/impls.rs
  cargo build --release -p flambeau-qwen3-moe --tests --features hip 2>&1 | tail -1
  BIN=$(ls -t target/release/deps/perf_baseline_qwen35_9b-* | grep -v '\.d$' | head -1)
  echo "=== $V ==="
  FLAMBEAU_MESH_RANKS=4 FLAMBEAU_QWEN35_GGUF=/artefact/models/Qwen3.5-9B-Q4_1.gguf \
    $BIN perf_baseline_qwen35_9b --nocapture 2>&1 | grep -E "decode|prefill L=1024"
done
# Restore:
sed -i "s|qmatmul_q4_1_mmvq_[a-z_0-9]*_gfx906|qmatmul_q4_1_mmvq_t128_gfx906|g" crates/backend-hip/src/impls.rs
```
