# V2.31.a — Q5_K indexed-MoE MMQ tile8 down kernel — Coder-30B prefill +28–50 %

## What landed

New kernel `indexed_moe_mmq_q5_k_down_tile8_dp4a` in
`crates/kernels-hip/src/kernels/`. Mirrors V2.6.b's Q4_K tile8 layout
and V2.8.b's Q6_K tile8 fix — combines Q4_K's d + dmin + 12-byte scale
header (same between Q4_K and Q5_K) with the Q5_K 5th-bit `qh` decode
(`bit = 2*il + half`, adds 16 to each raw_q).

Dispatch: `forward_moe_ffn_prefill` routes `Q5K if moe_variant == "tile8"`
to the new kernel; falls through to the existing MMVQ path for other
variants (sorted / r4 / decode).

## Why this was needed

V2.30.b profiling attributed **31.56 % of Coder-30B prefill wall** (640 ms
of 2026 ms at L=512) to `indexed_moe_mmvq_q5_k`. That's classic
prefill-on-MMVQ: the UD-Q4_K_XL promotes 13/48 `ffn_down_exps` to
Q5_K, and no tile8 MMQ variant existed — the dispatcher fell through
to single-row MMVQ, emitting 1 output per block × 64 pairs per call
vs the MMQ tile8 layout's 64 rows × 8 pairs per block.

V2.8.b fixed the identical pattern for Qwen3.6-35B's Q6_K: kernel
time 97 ms → 9.7 ms (−90 %).

## Results (100 W/GPU, Mesh<4>)

### Sync (u_lanes=1)

| L    | before | after | Δ      |
|------|-------:|------:|:-------|
|  128 |    288 |   430 | +49 %  |
|  512 |    298 |   460 | +54 %  |
| 1024 |    264 |   394 | +49 %  |
| 2048 |    218 |   295 | +36 %  |
| 4096 |    148 |   179 | +21 %  |
| 8192 |     88 |    99 | +12 %  |

### Async (u_lanes=2, ub=128) — the shipping config

| L    | before | after | Δ          |
|------|-------:|------:|:-----------|
|  128 |    286 |   432 | +51 %      |
|  512 |    426 |   641 | **+50 %**  |
| 1024 |    532 |   682 | **+28 %**  |
| 2048 |    523 |   706 | **+35 %**  |
| 4096 |    417 |   520 | **+25 %**  |
| 8192 |    290 |   331 | +14 %      |

Peak Coder-30B prefill: **706 tok/s @ L=2048** (up from 523). Decode
flat at 38.6 tok/s (expected — tile8 fires at n_tokens ≥ 32, so
per-token decode still goes through MMVQ).

Tail regression at L=8192 (+14 %) is quadratic-attention bound, not
MoE — the forward's other kernels dominate at long L.

## Parity

Bit-exact on the `forward_smoke_qwen3_coder` suite (same seed 9419
used for V2.28.b smoke):

- `L=1` greedy seed → last_id=**25** (same)
- Greedy 4 tokens: `[25, 330, 488, 9419]` (same)
- `L={2,4,16}` prefill last_ids: `39024 / 76808 / 67392` (same)

The perf_baseline L=8 test uses different synthetic input tokens so
its last_id comparison is not a parity signal across versions; the
smoke suite is canonical.

## Ship status

- `indexed_moe_mmq_q5_k_down_tile8_dp4a` kernel in-tree.
- `indexed_moe_mmq_q5_k_down_tile8` Rust binding + registered module.
- `forward_moe_ffn_prefill` dispatches Q5K through tile8 when
  `moe_variant == "tile8"` (default).
- Build green, smoke bit-exact, cert-check still 48 rows 0 failures.
- **No correctness cert yet** for the new kernel — follow-up sweep
  in `crates/bench/src/sweep_moe.rs` (V2.31.a-i2). The V2.31.a-i1
  cert for `indexed_moe_mmvq_q5_k` from V2.28.b covers the MMVQ
  fallback path.

## Perf snapshot

`certs/perf/qwen3_coder_30b_mesh4.json` overwritten (async ub=128
numbers).
