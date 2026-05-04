# P2.9b-i2-F throughput cert RE-RUN — post-#290+#291+#292 pipelined attempt

**Date**: 2026-05-04
**Model**: Qwen3.6-27B-Q4_1
**Topology**: pp2tp2 — `--devices hip:0,2,1,3 --pp-size 2 --tp-size 2`
**Build**: post-#290 (`forward_decode_pipelined_hybrid` written), post-#291
(scheduler dispatches under `FLAMBEAU_DECODE_PIPELINE=1` + `n_stages==2`
+ `n>=4`), post-#292 (correctness fix: use `forward_gdn_decode_batched_tp`
at N=1 instead of `forward_gdn_decode_tp`).

## Result: pipelined N=2 is CORRECT but PERF-NEGATIVE — gate at N≥4

### Output divergence (RESOLVED)

Same prompt ("Write a short poem about the sea (8 lines).", temp=0.0,
max_tokens=64) across modes after the GDN-batched fix:

| mode                                                | md5      | first 60 chars |
|-----------------------------------------------------|----------|----------------|
| N=1 single                                          | 59f23a62 | "The waves crash down with thunderous sound,..." |
| N=2 batched (no pipeline)                           | 59f23a62 | (identical to N=1) |
| **N=2 pipelined (post-#292 GDN-batched fix)**       | **59f23a62** | **(bit-identical to N=1!)** |

Pre-fix, pipelined produced md5 c2547432 ("rocky shore...") because it
called `forward_gdn_decode_tp` (single-slot path, equivalent to
`FLAMBEAU_GDN_NO_BATCHED=1`) which dispatches through a different
qmatmul code path than `forward_gdn_decode_batched_tp` at N=1 and
produces numerically different row-0 output. Swapping to the batched-N=1
helper restores bit-identical output across all three modes.

### Wall-clock

| mode                         | concurrent wall | speedup vs sequential |
|------------------------------|-----------------|-----------------------|
| N=2 batched (no pipeline)    | 6.04 s          | 0.96× (sequential = 5.80 s) |
| N=2 pipelined (post-fix)     | 7.09 s          | **0.82× (regression)** |

Pipelining at PP=2/N=2 has a design ceiling of 1.33× (from
`pipelined_decode.md` speedup table). The per-slot host launch overhead
(~1000 launches per slot serialised vs ~512 for batched at N=2) eats
through that ceiling, leaving a net regression.

The win curve crosses zero around N=4 where the ceiling rises to 1.6×.

### Memory budget at INFLIGHT_SLOTS=4

OOM at slot-pool boot:
```
Error: pre-alloc Inflight slot 2 at boot
Caused by: alloc KvCache<F16> (TP) for layer 27: out of memory
```

27B Q4_1 model + 4 KV slots over 64 layers × 16 heads × 128 head_dim ×
F16 × pp_stage=2 exceeds the gfx906 16GB budget. Validating pipelining
at N=4 on the 27B Q4_1 needs:

- A smaller KV ctx (default 4096 → 1024).
- A smaller weight quant (e.g., Q4_0 at 14.7 GB instead of Q4_1 at 16.1 GB).
- A different model that exercises the GDN+MoE path on smaller weights.

## Decisions

1. **Scheduler gate raised to N≥4** (`crates/server/src/routes.rs`):
   previously `pipeline_enabled && n_stages == 2 && n >= 2`, now
   `n >= 4`. At N=2, pipelining is strictly worse than batched on
   PP=2.

2. **Function uses batched-GDN at N=1** internally
   (`forward_gdn_decode_batched_tp` with `layer_states=&mut [s]`,
   `n=1`). This is the correctness-critical change to ship.

3. **The 3× cert gate stays open**. Pipelining was projected to deliver
   1.6× at PP=2/N=4 (per the design table), combined with batched-attn
   (1.05×) and a future batched-MMVQ (1.5–2× per #288). Until we can
   validate N≥4 on memory-constrained 27B, the projected combined
   benefit is unverified. #288 (batched-MMVQ kernels) remains the
   highest-leverage path to the gate.

## Reproduce

```bash
# Server (pipelined, requires N≥4 to engage):
FLAMBEAU_BATCHED_DECODE=1 FLAMBEAU_INFLIGHT_SLOTS=4 FLAMBEAU_DECODE_PIPELINE=1 \
  ./target/release/flambeau serve \
  --model /artefact/models/Qwen3.6-27B-Q4_1.gguf \
  --devices hip:0,2,1,3 --mesh-mode pp+tp --pp-size 2 --tp-size 2 --port 8080
# (currently OOMs at INFLIGHT_SLOTS=4; reduce ctx or model)

# Server (no pipeline, reference at N=2):
FLAMBEAU_BATCHED_DECODE=1 FLAMBEAU_INFLIGHT_SLOTS=2 \
  ./target/release/flambeau serve [...]

# Bisect debug: blocking peer_copy variant (rules out async issues):
FLAMBEAU_BATCHED_DECODE=1 FLAMBEAU_INFLIGHT_SLOTS=2 FLAMBEAU_DECODE_PIPELINE=1 \
  FLAMBEAU_PIPELINE_BLOCKING_COPY=1 \
  ./target/release/flambeau serve [...]

# Client (sends N=1, then 2 concurrent at temp=0):
python3 /tmp/test_compare.py
```
