# MTP-5e — perf cert: spec-decode vs baseline on Qwen3.6-27B / Mesh<4> PCIe

**Verdict:** spec-decode is **net-negative** on this rig (4× MI50 / PCIe 3.0
x16 / 100 W power cap / hybrid GDN+full-attn arch). 87.5 % acceptance is
real and stable, but the per-macro fixed costs eat the savings.

| metric                  | baseline (L=1 loop) | spec-decode (paired L=2 verify) |
|-------------------------|--------------------:|--------------------------------:|
| tokens                  | 59                  | 59 (32 macros)                  |
| wall                    | 2969 ms             | 3500 ms                         |
| **ms/token**            | **50.33**           | **59.33**                       |
| first 16 tokens match   | —                   | **bit-identical** vs baseline   |
| accept rate             | n/a                 | 87.5 % (28/32)                  |

Same prompt, same prefill prime, same model load. Both branches greedy.

## Why spec-decode loses on PCIe-only Mesh<4>

Per-macro cost decomposition (cumulative over 32 macros, paired-L2 path):

| stage                          | total ms | per macro   |
|--------------------------------|---------:|------------:|
| GDN snapshot (always)          | 65       | 2.0         |
| MTP draft (always)             | 49       | 1.5         |
| base verify (paired L=2)       | 3069     | **95.9**    |
| GDN restore (12.5 % rejects)   | 8        | 2.0 × 0.125 |
| L=1 redo (12.5 % rejects)      | 195      | 49 × 0.125  |
| **total**                      | **3386** | **~106**    |

Tokens-per-macro at 87.5 % accept = 0.875 × 2 + 0.125 × 1 = **1.875**.
→ effective ms/tok = 106 / 1.875 ≈ **56.5 ms/tok** (vs 59.3 measured;
delta is loop overhead + Vec allocs).

Compare baseline forward_one_token_pp_logits on the same topology:
**50.3 ms/tok**. Baseline wins by ~6–9 ms/tok.

## The MTP-5e primitive only saved 2.7 ms/macro

The L=2 paired-logits primitive (one PP forward body + 2× LM-head pass)
*was* expected to be the dominant lever. Measured:

- old (2× sequential L=1 verify): 98.6 ms/macro
- new (1× L=2 paired verify):     **95.9 ms/macro**

Saved ~2.7 ms/macro (2.7 %). The MTP-5c projection assumed an L=2 wall
multiplier of ~1.5× the L=1 wall ("L=2 ≈ 75 ms"); reality is **1.92×**
(95.9 / 50). The body of forward_one_token_pp_logits at L=1 on PCIe-only
Mesh<4> is dominated by rank-to-rank peer_copy_via_host (~10–50 µs ×
n_ranks per layer) and AR sync, not by per-layer compute. At L=2, the
PCIe hand-off cost is unchanged (still one hidden chunk per hop, just
twice as wide), but compute *also* doubles — so total wall scales
~linearly with L on this topology.

## Implications by topology

| topology                       | expected spec-decode verdict              |
|--------------------------------|-------------------------------------------|
| 4× MI50 PCIe (this rig)        | **net-negative** by ~18 % (this cert)     |
| 4× MI50 + xGMI / NVLink 4x     | likely net-positive: hand-off cost drops, |
|                                | L=2 multiplier approaches 1.5×            |
| Single-device dense Qwen       | likely net-positive: no GDN snapshot,     |
|                                | no per-rank sync; L=2 amortises cleanly   |
| Hybrid arch + PCIe (this case) | structurally hostile to spec-decode       |

## Acceptance is at the published-head ceiling

87.5 % matches the K=1 1-ahead measurement from MTP-5c and the community
67–69 % range adjusted for prose vs code. The Q8_0 MTP head, 1-ahead
verify, and structural guards (GDN snapshot/restore, KV rollback) are
all working as intended. A higher number requires a fine-tuned MTP head
(FastMTP, multi-week training) — see `mtp_inv_5_aeon_same_head_ceiling.md`.

## Gating

Spec-decode is shipped behind opt-in env vars:

- `FLAMBEAU_SPEC_MTP=<path-to-mtp.gguf>` — server loads MTP head
- Greedy sampling — required (rejection sampling is MTP-5g)
- PP topology — required (TP / hybrid is MTP-5f)

Default off. Users who specifically want acceptance-rate telemetry or
who run on a future xGMI/NVLink rig can enable it.

## Next levers (filed)

- **MTP-5f** (PP+TP / TP) — TP topologies have lower per-step rank-sync
  cost, may flip the L=2 multiplier favourable
- **MTP-5g** (rejection sampling) — orthogonal; non-greedy decode users
- HipEvent-attribution per-stage timing on the L=2 body — currently we
  measure the whole primitive as one wall; finding the bottleneck inside
  the body (peer-copy vs layer compute vs LM head) would inform whether
  a different verify shape (e.g. K=2 spec) could win

## Reproducer

```
FLAMBEAU_PERF_AB_TOKENS=60 cargo test --release \
    -p flambeau-qwen3-moe --features hip \
    --test mtp_spec_decode_perf_ab -- --nocapture
```

Test source: `crates/models/qwen3-moe/tests/mtp_spec_decode_perf_ab.rs`.
Rig: 4× MI50 (gfx906) / PCIe 3.0 x16 / 100 W cap / ROCm 7.1.1.
Model: `/artefact/models/Qwen3.6-27B-Q4_0.gguf` + `Qwen3.6-27B-mtp.gguf`
(Q8_0 MTP linears, MTP-INV-4 default).

## Cleanup note

Both `mtp_spec_decode_smoke` and `mtp_spec_decode_perf_ab` SIGSEGV on
process exit *after* the test prints PASS and all measurements complete.
This is in dispose / drop ordering, not in the spec-decode hot path.
Filed as separate cleanup; does not affect this cert's numbers.
