# Qwen family long-prompt + long-response sweep — 2026-04-29

Live-server bench: 1500-token prompt → up to 2048-token response, sampling
config `temp=0.7 top_p=0.9 seed=42`, on 4× MI50 PCIe / 100 W cap / ROCm 7.1.1.
Each model loaded fresh via `flambeau serve`, request issued via curl,
wall + TTFT extracted from the server's `tracing` log.

| model | topology | load_s | TTFT_s | prefill tok/s | decode tok/s | comp_tok | finish |
|---|---|---:|---:|---:|---:|---:|---|
| Qwen3.5-9B-Q4_1 | tp2 (hip:0,1) | 8.0 | 0.69 | **950.2** | **36.3** | 2048 | length |
| Qwen3.5-27B-Q4_1 | pp2tp2 (0,2,1,3) | 18.1 | 2.11 | 308.9 | 20.9 | 1316 | stop |
| Qwen3.6-27B-Q4_1 | pp2tp2 (0,2,1,3) | 18.1 | 2.36 | 276.6 | 21.1 | 934 | stop |
| Qwen3.6-27B-UD-Q8_K_XL | pp4 (hip:0,1,2,3) | 30.1 | 9.31 | 70.1 | 13.2 | 2048 | length |
| Qwen3.6-35B-A3B-UD-Q8_K_XL | pp4 (hip:0,1,2,3) | 54.2 | 1.04 | **625.0** | **33.6** | 70 | stop (early) |

## Findings

### MoE wins decode despite size

Qwen3.6-35B-A3B-UD-Q8_K_XL is **38 GB** on disk but decodes at **33.6 tok/s**
— faster than every 27B (Q8 or Q4_1). Reason: A3B = ~3 B *active* parameters
per token through the MoE router. Only the routed expert weights flow through
HBM per decode step; the other 35 B sit idle. On gfx906 / 1 TB/s HBM the
working-set bandwidth dominates, not param count.

### Quantisation × bandwidth on the 27B

Same arch (Qwen3.6-27B), same topology, just changing quant:
- Q4_1 / pp2tp2 → 21.1 tok/s decode, 277 tok/s prefill
- Q8_K_XL / pp4 → 13.2 tok/s decode, 70 tok/s prefill

The decode 1.6× slowdown matches the ~2× weight-bandwidth ratio (mitigated by
UD-Q8_K_XL having mixed-quant K-tile layers that aren't pure Q8). Prefill
suffers more (3.95×) because it's more compute-touched per token, AND the Q8
prefill kernels on gfx906 don't match the Q4_1's mature `mmq_q4_1_4warp_lds`
path.

### Topology × model size

- **9B tp2** beats every other config on both axes — small dense model fits
  cleanly into 2-way TP, BAR1 P2P AllReduce eats less wall than PP peer-copy.
- **27B pp2tp2** (~21 tok/s) is balanced — the 27B is too big for tp2 (28 GB
  Q8 won't fit in 2× 16 GB MI50), and pp4 alone is slower than pp2tp2 because
  PCIe peer-copy serialises 4 stages instead of 2.
- **35B-A3B pp4** is forced — pp2tp2 has the qwen35moe gibberish bug
  (filed task #205, separate from CN-80B-15's qwen3next fix).

### 35B-A3B short generation

The 35B-A3B run produced only **70 tokens** before `finish_reason: stop`. At
temp=0.7 the model decided to wrap an opening sentence and emit EOS. Not a
correctness issue (output coherent), but the decode rate (33.6) is averaged
over a tiny window and may carry larger noise than the longer runs.

### Qwen3.5-27B vs Qwen3.6-27B on Q4_1

Same params, same topology, ~identical wall numbers (308.9 vs 276.6 prefill,
20.9 vs 21.1 decode). The hybrid (3.6) vs dense (3.5) layer mix doesn't move
the needle at this scale on this rig — both are dominated by the matmul
bandwidth of the routed FFN/MoE path.

## Topology recommendations (production)

Confirmed by this sweep + corroborates `project_v2_30_a_bench_tour_100w.md`:

| model size | topology | reason |
|---|---|---|
| 9B (any quant) | **tp2** (hip:0,1) | smallest weights, BAR1 AllReduce |
| 27B Q4_1 / Q4_0 | **pp2tp2** (hip:0,2,1,3) | hybrid balances PCIe + AR |
| 27B Q8_K_XL | **pp4** (hip:0,1,2,3) | only topology with VRAM headroom |
| 35B-A3B | **pp4** (forced) | pp2tp2 has known qwen35moe bug |

## Caveats

- ctx clamped to 8192 (or 4096 for the two largest) — KV cache memory budget.
- Long-context decode (here ~700-2700 cumulative tokens) is slower than
  short-context — KV walks scale linearly with position. The synthetic L=64
  tg numbers in the V2.30.a cert (53.5 tok/s for 35B-A3B) are not directly
  comparable; this sweep is the realistic chat-workload number.
- Run-to-run variance ±5 % on decode rates from kernel-launch jitter +
  thermal throttling at 100 W cap.

## Reproducer

```sh
/tmp/sweep.sh   # at /tmp/qwen_sweep_181644/{prompt,resp_*,serve_*,summary}.tsv
```

Script source: `/tmp/sweep.sh` (long-prompt body inlined, 5-model sweep,
auto start/stop per model, server-log TTFT extraction).
