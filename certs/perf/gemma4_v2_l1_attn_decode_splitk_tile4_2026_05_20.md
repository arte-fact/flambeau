# L1 — attention_decode_f16_splitk_chunk tile-2 → tile-4

Date: 2026-05-20
Status: **shipped.** SWA-parity tests green. Coherence verified on E4B.

## Change

`kernels-hip/src/kernels/attention_decode_f16_splitk.cu`: the inner
loop's K-row tile bumped from 2 to 4. Each iter now processes four
consecutive K/V tokens sharing one cross-warp LDS roundtrip and one
`__syncthreads`. Score-parts LDS bumped from `MAX_WARPS * 2` to
`MAX_WARPS * 4` (128 B at head_dim=512).

## Measurement (E4B-Q4_0, SD, prompt 291 tok, decode 128 tok)

| metric | before L1 | after L1 | delta |
|---|---:|---:|---:|
| `attention_decode_f16_splitk_chunk` total | 554.2 ms | 371.6 ms | **−33 %** |
| `attention_decode_f16_splitk_chunk` mean | 103.9 µs | 69.7 µs | −33 % |
| total kernel time | 3351 ms | 3115 ms | −7 % |
| share of total | 16.5 % | 11.9 % | −4.6 pp |

Output unchanged ("Paris." short / coherent at 820 tok long-SWA).
SWA + softcap parity tests all green
(`cargo test -p flambeau-ops --test swa_softcap_parity`):
- splitk_matches_single_pass_gemma4_hd512
- swa_decode_splitk_matches_window_4
- (9 more) — all green.

## Why it pays

`mean(us)` dropped 1:1 with tile size growth — the inner loop is
heavy on per-iter `__syncthreads` and cross-warp reduce overhead.
Going from 2 K rows per sync to 4 doubles the per-sync useful work
without growing the LDS budget meaningfully.

VGPR cost: 2 extra per-warp partial floats + 2 extra v_X floats per
thread. Still fits within wave64 occupancy on gfx906; no observed
register-pressure regression in the trace.

## Applies to all 3 gemma4 variants

Same kernel hot in E4B (16% before), 31B (8.5%), 26B-A4B (13%).
The 33% drop maps directly onto each.

## Tile-8 not attempted

VGPR + LDS budgets would still fit, but at chunk_size = 16 / 32 / 64
(splitk default for n ≤ 1024) tile-8 gives at most one outer iter
per chunk — diminishing returns. Re-investigate when split-K dispatch
runs at larger chunks (n_tokens_kv > 4096).

## L2 audit + L3 findings (companion work)

**L2 audit:** `rmsnorm_quant_q8_1` already in use at every n=1
pre-norm site (standard_attn.rs:148, dense_ffn.rs:52). The hot
standalone `rmsnorm_f16` calls are Q/K/V-norm, attn_post_norm,
ffn_post_norm — none followed by Q8_1 quantize, so the proposed
"audit + fuse" doesn't directly apply. Remaining fusion candidate
(rmsnorm(delta) + add_residual for gemma4 post_*) is ~1% saved per
arch; deferred.

**L3 copyBuffer audit:** the dominant chunk is **KV-append DtoD**
(2 memcpys × per-layer × per-token = ~10 752 launches per E4B
request, ~54 ms = 1.6 %). The PLE DtoH+HtoD on E4B is ~22 MB each
direction per prefill chunk = ~6 ms wall, only ~0.2 %. Position-
buffer per-layer HtoD is 5376 launches × ~4.5 µs = 24 ms, ~0.7 %.
None individually large enough to justify the structural change
(KV-append fusion or full PLE-on-GPU build) inside this lever push.
Tracked as future work; the splitk tile-4 win covers the
~7 % wall reduction the cert estimated for L1+L2+L3 combined.
