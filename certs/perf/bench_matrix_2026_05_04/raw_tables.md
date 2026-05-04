# Bench matrix — 2026-05-04T21:57:08 → 2026-05-04T22:47:55

max_tokens=256, concurrencies=[1, 2, 4, 8]

## qwen35_9B_q4_1 / pp2tp2

prompt_tokens=3313, completion_tokens_per_stream=256

| N | path | prefill_ms (mean) | per-stream tps (mean) | aggregate tps | n_ok | n_err |
|---|------|-------------------|------------------------|---------------|------|-------|
| 1 | no_batched | 4334 | 49.7 | 49.7 | 1 | 0 |
| 1 | batched | 4352 | 49.5 | 49.5 | 1 | 0 |
| 2 | no_batched | 8579 | 44.5 | 89.0 | 2 | 0 |
| 2 | batched | 8553 | 44.2 | 88.2 | 2 | 0 |
| 4 | no_batched | 13674 | 20.4 | 80.0 | 4 | 0 |
| 4 | batched | 13355 | 19.9 | 72.9 | 4 | 0 |
| 8 | no_batched | 20543 | 9.8 | 75.4 | 8 | 0 |
| 8 | batched | 22502 | 9.8 | 75.1 | 8 | 0 |

| N | aggregate tps no_batched | aggregate tps batched | batched / no_batched |
|---|--------------------------|-----------------------|----------------------|
| 1 | 49.7 | 49.5 | 1.00× |
| 2 | 89.0 | 88.2 | 0.99× |
| 4 | 80.0 | 72.9 | 0.91× |
| 8 | 75.4 | 75.1 | 1.00× |

## qwen35_9B_q4_1 / pp4

prompt_tokens=3313, completion_tokens_per_stream=256

| N | path | prefill_ms (mean) | per-stream tps (mean) | aggregate tps | n_ok | n_err |
|---|------|-------------------|------------------------|---------------|------|-------|
| 1 | no_batched | 6183 | 41.6 | 41.6 | 1 | 0 |
| 1 | batched | 6149 | 42.0 | 42.0 | 1 | 0 |
| 2 | no_batched | 6707 | 42.0 | 83.9 | 2 | 0 |
| 2 | batched | 6489 | 42.0 | 84.0 | 2 | 0 |
| 4 | no_batched | 9969 | 27.7 | 110.7 | 4 | 0 |
| 4 | batched | 10654 | 28.2 | 112.4 | 4 | 0 |
| 8 | no_batched | 16627 | 15.4 | 119.6 | 8 | 0 |
| 8 | batched | 16462 | 16.7 | 115.3 | 8 | 0 |

| N | aggregate tps no_batched | aggregate tps batched | batched / no_batched |
|---|--------------------------|-----------------------|----------------------|
| 1 | 41.6 | 42.0 | 1.01× |
| 2 | 83.9 | 84.0 | 1.00× |
| 4 | 110.7 | 112.4 | 1.02× |
| 8 | 119.6 | 115.3 | 0.96× |

## qwen35_9B_q4_1 / single

prompt_tokens=3313, completion_tokens_per_stream=256

| N | path | prefill_ms (mean) | per-stream tps (mean) | aggregate tps | n_ok | n_err |
|---|------|-------------------|------------------------|---------------|------|-------|
| 1 | no_batched | 6244 | 40.5 | 40.5 | 1 | 0 |
| 1 | batched | 6537 | 38.5 | 38.5 | 1 | 0 |
| 2 | no_batched | 12629 | 20.2 | 40.2 | 2 | 0 |
| 2 | batched | 12642 | 18.7 | 36.0 | 2 | 0 |
| 4 | no_batched | 24138 | 9.5 | 36.0 | 4 | 0 |
| 4 | batched | 25150 | 9.1 | 33.9 | 4 | 0 |
| 8 | no_batched | 50041 | 4.7 | 35.7 | 8 | 0 |
| 8 | batched | 51408 | 4.5 | 34.2 | 8 | 0 |

| N | aggregate tps no_batched | aggregate tps batched | batched / no_batched |
|---|--------------------------|-----------------------|----------------------|
| 1 | 40.5 | 38.5 | 0.95× |
| 2 | 40.2 | 36.0 | 0.90× |
| 4 | 36.0 | 33.9 | 0.94× |
| 8 | 35.7 | 34.2 | 0.96× |

## qwen35_9B_q4_1 / tp2

prompt_tokens=3313, completion_tokens_per_stream=256

| N | path | prefill_ms (mean) | per-stream tps (mean) | aggregate tps | n_ok | n_err |
|---|------|-------------------|------------------------|---------------|------|-------|
| 1 | no_batched | 4298 | 48.5 | 48.5 | 1 | 0 |
| 1 | batched | 4329 | 47.7 | 47.7 | 1 | 0 |
| 2 | no_batched | 8535 | 22.1 | 43.6 | 2 | 0 |
| 2 | batched | 8703 | 21.4 | 42.8 | 2 | 0 |
| 4 | no_batched | 16985 | 10.6 | 40.8 | 4 | 0 |
| 4 | batched | 17018 | 10.5 | 40.6 | 4 | 0 |
| 8 | no_batched | 33435 | 5.3 | 42.0 | 8 | 0 |
| 8 | batched | 34418 | 5.2 | 40.1 | 8 | 0 |

| N | aggregate tps no_batched | aggregate tps batched | batched / no_batched |
|---|--------------------------|-----------------------|----------------------|
| 1 | 48.5 | 47.7 | 0.98× |
| 2 | 43.6 | 42.8 | 0.98× |
| 4 | 40.8 | 40.6 | 0.99× |
| 8 | 42.0 | 40.1 | 0.95× |

## qwen36_27B_q4_1 / pp2tp2

prompt_tokens=3313, completion_tokens_per_stream=256

| N | path | prefill_ms (mean) | per-stream tps (mean) | aggregate tps | n_ok | n_err |
|---|------|-------------------|------------------------|---------------|------|-------|
| 1 | no_batched | 12031 | 22.8 | 22.8 | 1 | 0 |
| 1 | batched | 12057 | 22.8 | 22.8 | 1 | 0 |
| 2 | no_batched | 23637 | 19.7 | 39.3 | 2 | 0 |
| 2 | batched | 23894 | 19.5 | 38.9 | 2 | 0 |
| 4 | no_batched | 33490 | 8.8 | 34.3 | 4 | 0 |
| 4 | batched | 29315 | 8.6 | 33.8 | 4 | 0 |
| 8 | no_batched | 57920 | 4.1 | 32.2 | 8 | 0 |
| 8 | batched | 57857 | 4.0 | 31.0 | 8 | 0 |

| N | aggregate tps no_batched | aggregate tps batched | batched / no_batched |
|---|--------------------------|-----------------------|----------------------|
| 1 | 22.8 | 22.8 | 1.00× |
| 2 | 39.3 | 38.9 | 0.99× |
| 4 | 34.3 | 33.8 | 0.98× |
| 8 | 32.2 | 31.0 | 0.96× |

## qwen36_27B_q4_1 / pp4

prompt_tokens=3313, completion_tokens_per_stream=256

| N | path | prefill_ms (mean) | per-stream tps (mean) | aggregate tps | n_ok | n_err |
|---|------|-------------------|------------------------|---------------|------|-------|
| 1 | no_batched | 20134 | 17.1 | 17.1 | 1 | 0 |
| 1 | batched | 20315 | 17.3 | 17.3 | 1 | 0 |
| 2 | no_batched | 20653 | 16.1 | 31.3 | 2 | 0 |
| 2 | batched | 20746 | 16.0 | 31.1 | 2 | 0 |
| 4 | no_batched | 28211 | 11.6 | 40.0 | 4 | 0 |
| 4 | batched | 27703 | 11.3 | 41.7 | 4 | 0 |
| 8 | no_batched | 50936 | 5.6 | 40.1 | 8 | 0 |
| 8 | batched | 48685 | 5.3 | 38.0 | 8 | 0 |

| N | aggregate tps no_batched | aggregate tps batched | batched / no_batched |
|---|--------------------------|-----------------------|----------------------|
| 1 | 17.1 | 17.3 | 1.01× |
| 2 | 31.3 | 31.1 | 0.99× |
| 4 | 40.0 | 41.7 | 1.04× |
| 8 | 40.1 | 38.0 | 0.95× |

## qwen36_27B_q4_1 / tp2

prompt_tokens=3313, completion_tokens_per_stream=256

| N | path | prefill_ms (mean) | per-stream tps (mean) | aggregate tps | n_ok | n_err |
|---|------|-------------------|------------------------|---------------|------|-------|
| 1 | no_batched | 11903 | 20.5 | 20.5 | 1 | 0 |
| 1 | batched | 12083 | 19.9 | 19.9 | 1 | 0 |
| 2 | no_batched | 24079 | 9.2 | 18.3 | 2 | 0 |
| 2 | batched | 24797 | 9.0 | 17.9 | 2 | 0 |
| 4 | no_batched | 12381 | 19.9 | 19.9 | 1 | 3 |
| 4 | batched | 25585 | 8.9 | 17.7 | 2 | 2 |
| 8 | no_batched | 12675 | 19.7 | 19.7 | 1 | 7 |
| 8 | batched | 26338 | 8.8 | 17.6 | 2 | 6 |

| N | aggregate tps no_batched | aggregate tps batched | batched / no_batched |
|---|--------------------------|-----------------------|----------------------|
| 1 | 20.5 | 19.9 | 0.97× |
| 2 | 18.3 | 17.9 | 0.98× |
| 4 | 19.9 | 17.7 | 0.89× |
| 8 | 19.7 | 17.6 | 0.89× |

## qwen36_35B_a3b_q4_0 / pp2tp2

prompt_tokens=3313, completion_tokens_per_stream=256

| N | path | prefill_ms (mean) | per-stream tps (mean) | aggregate tps | n_ok | n_err |
|---|------|-------------------|------------------------|---------------|------|-------|
| 1 | no_batched | 3958 | 44.2 | 44.2 | 1 | 0 |
| 1 | batched | 3962 | 44.2 | 44.2 | 1 | 0 |
| 2 | no_batched | 7903 | 46.1 | 92.2 | 2 | 0 |
| 2 | batched | 7980 | 46.5 | 93.0 | 2 | 0 |
| 4 | no_batched | 11656 | 19.7 | 77.3 | 4 | 0 |
| 4 | batched | 9194 | 19.5 | 76.1 | 4 | 0 |
| 8 | no_batched | 19050 | 9.4 | 72.8 | 8 | 0 |
| 8 | batched | 19239 | 9.4 | 73.9 | 8 | 0 |

| N | aggregate tps no_batched | aggregate tps batched | batched / no_batched |
|---|--------------------------|-----------------------|----------------------|
| 1 | 44.2 | 44.2 | 1.00× |
| 2 | 92.2 | 93.0 | 1.01× |
| 4 | 77.3 | 76.1 | 0.98× |
| 8 | 72.8 | 73.9 | 1.01× |

## qwen36_35B_a3b_q4_0 / pp4

prompt_tokens=3313, completion_tokens_per_stream=256

| N | path | prefill_ms (mean) | per-stream tps (mean) | aggregate tps | n_ok | n_err |
|---|------|-------------------|------------------------|---------------|------|-------|
| 1 | no_batched | 5565 | 41.2 | 41.2 | 1 | 0 |
| 1 | batched | 5527 | 41.5 | 41.5 | 1 | 0 |
| 2 | no_batched | 5885 | 40.3 | 79.1 | 2 | 0 |
| 2 | batched | 5876 | 40.2 | 78.9 | 2 | 0 |
| 4 | no_batched | 8583 | 35.3 | 132.9 | 4 | 0 |
| 4 | batched | 9608 | 33.6 | 124.9 | 4 | 0 |
| 8 | no_batched | 16596 | 17.1 | 126.1 | 8 | 0 |
| 8 | batched | 15521 | 17.2 | 125.6 | 8 | 0 |

| N | aggregate tps no_batched | aggregate tps batched | batched / no_batched |
|---|--------------------------|-----------------------|----------------------|
| 1 | 41.2 | 41.5 | 1.01× |
| 2 | 79.1 | 78.9 | 1.00× |
| 4 | 132.9 | 124.9 | 0.94× |
| 8 | 126.1 | 125.6 | 1.00× |

## Skipped (infeasible)

- qwen36_27B_q4_1 / single: infeasible
- qwen36_35B_a3b_q4_0 / single: infeasible
- qwen36_35B_a3b_q4_0 / tp2: infeasible
