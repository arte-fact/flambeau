# 27B/pp2tp2 config sweep — pre-collapse fast-path active

- **Generated:** 2026-05-06T23:50:04+00:00
- **Commit:** 27cdabc
- **Profile baseline:** `bench/profiles/optimized.toml`
- **Configs:** baseline, prefill_ubatch_1024, ar_fuse_q8_1, all_optins
- **Concurrencies:** [1, 4]
- **Runs/cell:** 2 measured + 1 warmup

## Cells

| config | N | runs | prefill ms | decode ms | agg tg t/s | Δ prefill | Δ tg | err |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `baseline` | 1 | 2 | 11360 | 2726 | 23.5 | — | — | |
| `baseline` | 4 | 2 | 30859 | 4015 | 62.3 | — | — | |
| `prefill_ubatch_1024` | 1 | 2 | 10868 | 2718 | 23.5 | +4.3% | +0.3% | |
| `prefill_ubatch_1024` | 4 | 2 | 29736 | 4831 | 49.3 | +3.6% | -20.8% | |
| `ar_fuse_q8_1` | 1 | 2 | 11443 | 2763 | 23.2 | -0.7% | -1.3% | |
| `ar_fuse_q8_1` | 4 | 2 | 31084 | 4155 | 60.2 | -0.7% | -3.4% | |
| `all_optins` | 1 | 2 | 11460 | 2731 | 23.4 | -0.9% | -0.2% | |
| `all_optins` | 4 | 2 | 31073 | 4088 | 60.0 | -0.7% | -3.6% | |

## Winners per metric per N

| N | best prefill (config, ms) | best decode-tg (config, t/s) |
|---:|---|---|
| 1 | `prefill_ubatch_1024` (10868 ms) | `prefill_ubatch_1024` (23.5 t/s) |
| 4 | `prefill_ubatch_1024` (29736 ms) | `baseline` (62.3 t/s) |
