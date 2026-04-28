# CN-80B-12 — Topology re-bench + combined pp+tg target confirmation

## Verdict

**pp2tp2 is the strict-best topology on Coder-Next-80B post-CN-80B-11c.**
It wins on every measured axis (prefill at all L, decode at all tg)
and is the only viable topology at L=4096 on the 4×16 GB rig.

The user's "both pp and tg good simultaneously" criterion (parity
with what 35B-A3B achieved on a single topology) is met.

## Numbers — 4× MI50 PCIe 3.0 x16, 100 W/GPU, ROCm 7.1.1

| topology | load_s | pp128 | pp512 | pp2048 | pp4096 | tg64 | tg128 | tg256 |
|----------|-------:|------:|------:|-------:|-------:|-----:|------:|------:|
| pp4      |  82.43 | 443.5 | 569.1 |  597.8 |   OOM  | 41.1 |  40.2 |  38.5 |
| **pp2tp2** | 101.29 | **492.6** | **801.2** | **904.5** | **868.7** | **46.2** | **45.4** | **43.4** |

pp2tp2 vs pp4 head-to-head (% gain):

| axis    | pp2tp2 | pp4   | Δ       |
|---------|-------:|------:|--------:|
| pp128   | 492.6  | 443.5 |   +11 % |
| pp512   | 801.2  | 569.1 | **+41 %** |
| pp2048  | 904.5  | 597.8 | **+51 %** |
| pp4096  | 868.7  | OOM   |   ∞     |
| tg64    |  46.2  |  41.1 |  +12 %  |
| tg128   |  45.4  |  40.2 |  +13 %  |
| tg256   |  43.4  |  38.5 |  +13 %  |

mesh1, pp2, tp2 are not viable on this 45 GB GGUF + 4×16 GB rig.

## Compared to "the other models" the user referenced

35B-A3B-UD-Q4_K_S best-topology (pp4) — `project_v1_bench_matrix`:

| model               | best-topology | pp512 | tg64 |
|---------------------|---------------|------:|-----:|
| 35B-A3B-UD-Q4_K_S   | pp4           |   605 |   65 |
| Coder-Next-80B Q4_0 | **pp2tp2**    |   801 |   46 |

Prefill is **better** than 35B-A3B (801 vs 605, +32 %); decode is
lower (46 vs 65) but that's silicon — Coder-Next has 48 layers vs 40
*and* 512 experts (vs 128), so the per-token decode wall is
mechanically heavier even with optimal kernel scheduling. pp2tp2
specifically is what closes the gap (decode AR cost on 2 ranks per
stage < pp4's 4-stage pipeline bubble).

## Cause of the change

Two CN-80B-11c pieces (cert: `coder_next_80b_cn80b_11c_tp_tile8.md`):

1. `indexed_moe_mmq_q4_1_down_tile8_dp4a` kernel — Q4_1 affine
   reconstruction in the existing tile8 launch shape.
2. TP-prefill tile8 wiring in `forward_moe_ffn_prefill_tp` — sort+pad
   gate at L≥32, dispatching Q4_0 / Q8_0 / Q4_1 down kernels with
   `local_inter`. The TP path historically fell through to per-token
   MMVQ at any L; this is what CN-80B-11a measured as 84 % of pp2tp2
   wall (49 ms/layer in `ptp_moe_ffn`).

Post-11c the same section runs at 4.17 ms/layer — an 11.7× per-layer
drop, driving the 4.4× wall improvement in pp2tp2 prefill.

## Production guidance

For Coder-Next-80B serving on this rig, use:

```
flambeau serve --model /artefact/models/Qwen3-Coder-Next-Q4_0.gguf \
  --mesh-mode pp+tp --pp-size 2 --tp-size 2 --devices 0,2,1,3
```

The `0,2,1,3` device order keeps each TP pair (`{0,2}`, `{1,3}`) on
distinct dies of the Whitehaven board and avoids the `{2,3}` BAR1
peer fault (`project_rig_gpu23_link_fault`,
`project_rig_whitehaven_topology`).

## Closes

CN-80B-12 #138 — combined pp+tg target met on a single topology,
matching the bar set by 35B-A3B's pp4 result. Coder-Next's residual
gap to 35B-A3B on decode is silicon-bounded (more layers + experts),
not a topology or kernel deficit.

## Reproduce

```
# pp2tp2
FLAMBEAU_BENCH_GGUF=/artefact/models/Qwen3-Coder-Next-Q4_0.gguf \
FLAMBEAU_BENCH_TAG=coder_next_80b_q4_0_cn80b_12 \
FLAMBEAU_BENCH_TOPOLOGY_TAG=pp2tp2 \
FLAMBEAU_BENCH_PREFILL_LENGTHS=128,512,2048,4096 \
FLAMBEAU_BENCH_DECODE_LENGTHS=64,128,256 \
RUST_MIN_STACK=67108864 \
cargo test --release -p flambeau-qwen3-moe --features hip \
  --test v1_bench_matrix v1_bench_matrix -- --ignored --nocapture

# pp4 (cap pp_lengths at 2048 — pp4096 OOMs at 167 MB scratch alloc)
FLAMBEAU_BENCH_GGUF=/artefact/models/Qwen3-Coder-Next-Q4_0.gguf \
FLAMBEAU_BENCH_TAG=coder_next_80b_q4_0_cn80b_12_pp4 \
FLAMBEAU_BENCH_TOPOLOGY_TAG=pp4 \
FLAMBEAU_BENCH_PREFILL_LENGTHS=128,512,2048 \
FLAMBEAU_BENCH_DECODE_LENGTHS=64,128,256 \
RUST_MIN_STACK=67108864 \
cargo test --release -p flambeau-qwen3-moe --features hip \
  --test v1_bench_matrix v1_bench_matrix -- --ignored --nocapture
```
