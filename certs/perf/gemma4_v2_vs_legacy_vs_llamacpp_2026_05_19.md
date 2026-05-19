# gemma-4 — flambeau-v2 vs flambeau-legacy vs llama.cpp baseline

Hardware: 4× MI50 (gfx906), `LD_LIBRARY_PATH=/opt/rocm-host/lib`.
Prompt 725 / 702 tokens (real_task_pp1024 DP4A technical prompt; the
flambeau-v2 / legacy / llama.cpp tokenizers land at slightly different
counts), 128 decode tokens, greedy (temperature 0). One warmup +
one measured turn per arm. `flambeau serve --inflight-slots 1
--ctx-cap 4096`; llama.cpp `--ctx-size 4096 -ngl 999 --flash-attn on
--no-mmap --threads 8`.

Bench harness: `scripts/bench/gemma4_v2_vs_legacy_vs_llamacpp.py`.

## Headline by case

### gemma4-31B-it-Q4_0  (TP2 on hip:0,1)

| stack            | prefill t/s | decode t/s | output |
|------------------|------------:|-----------:|--------|
| **flambeau-v2**  | **286.7**   | 18.34      | coherent |
| flambeau-legacy  | 19.6 (!)    | 18.88      | coherent |
| **llama.cpp**    | 184.5 (row) | **21.10**  | coherent |

- v2 prefill **1.55× llama.cpp**, **14.6× legacy**.
- v2 decode **0.87× llama.cpp** (13% gap), 0.97× legacy.
- Legacy prefill is unusably slow (~19 t/s on a 17 GB Q4_0 model on TP2).
  Confirms the brief: legacy gemma4 is not a competitive baseline; the
  bar is llama.cpp.

### gemma4-31B-it-Q8_0  (4 GPUs)

Flambeau on pp2tp2 (the user's prod topology, devices 0,2,1,3 to
avoid the {2,3} link-faulted-for-TP pair); llama.cpp on pp4 same 4
devices (llama.cpp can't do TP-of-PP).

| stack                       | prefill t/s | decode t/s | output      |
|-----------------------------|------------:|-----------:|-------------|
| **flambeau-v2** (pp2tp2)    | **128.1**   | **18.87**  | coherent    |
| flambeau-legacy (pp2tp2)    | 20.8        | 20.32      | **`<pad>` output — broken on Q8_0 31B** |
| llama.cpp (pp4)             | 96.8        | 15.59      | coherent    |

- v2 prefill **1.32× llama.cpp**.
- v2 decode **1.21× llama.cpp**. **v2 ahead on both axes for Q8_0 dense.**
- pp4 on flambeau OOMs (gemma4 embedding is 2.8 GB; pp4 packs it
  entirely onto rank 0 → exceeds 16 GB MI50). pp4 on flambeau is
  unfeasible without TP-sharded embedding load.

## Out-of-scope variants

| variant            | reason                                                                 |
|--------------------|------------------------------------------------------------------------|
| gemma-4-E4B-Q4_0   | gemma-4n per-layer side-channel embedding; v2 + legacy both unsupported. llama.cpp 70.8 t/s decode SD. |
| gemma-4-26B-A4B-Q8_0 | v2 dense-only; legacy MoE-prefill explicitly errors "MoE prefill not supported". llama.cpp 66.2 t/s decode PP2. |
| gemma-4-31B-Q8_0 pp4 (flambeau) | rank-0 OOM (2.8 GB embedding + per-rank layers + scratch > 16 GB). |

## Reading

For dense gemma4 on this rig the comparison is:

- **31B-Q4_0**: llama.cpp leads decode by 13%. v2 leads prefill by 55%.
- **31B-Q8_0**: v2 leads both decode (21%) and prefill (32%).

The Q4_0 decode gap mirrors the qwen3.6-27B pre-lever shape (8–15%
behind legacy/llamacpp on Q4_0 decode, prefill ahead). The qwen-side
levers shipped in commit `00400dd` (F16-direct mmvq, F16 router,
event-gated AR) target exactly this pattern; the same composites are
used by gemma4-v2, so those levers should propagate the win to
gemma4. Next step is to rocprofv3-trace 31B-Q4_0 TP2 decode to confirm
the hot kernels are the same as on qwen and quantify the headroom.

## Files / data

- Bench JSON: `certs/perf/gemma4_v2_vs_legacy_vs_llamacpp_2026_05_19.json`
- Bench script: `scripts/bench/gemma4_v2_vs_legacy_vs_llamacpp.py`
- Server logs (per arm): `/tmp/gemma4_bench_{v2,legacy,llamacpp}_<port>.log`
