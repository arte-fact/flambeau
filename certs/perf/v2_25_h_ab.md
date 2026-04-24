# V2.25.h — async PP refinement iter 2: 1F1B interleaved dispatch

Rewrote `forward_prefill_pp_async` from serial-across-ubatches
(`for ub { for rank {...} }`) to 1F1B interleaved dispatch
(`for t { for rank: if 0 <= t-r < K: dispatch(t-r, r) }`). At steady
state, all N ranks dispatch different ubatches in the same time step —
real pipeline fill.

## Results (9B Q4_1 Mesh<4>, L=1024)

| ubatch | V2.25.d serial | **V2.25.h 1F1B** | Δ iter | Δ sync |
|---:|---:|---:|---:|---:|
| 128 | 517.57 | **553.13** | +6.9 % | −27.9 % |
| 256 | 632.28 | **678.23** | +7.3 % | −11.6 % |
| **512** | 696.15 | **711.24** | **+2.2 %** | **−7.3 %** |

Sync reference: 767.20 tok/s. Parity bit-exact across all configs
(`last_id=220`).

**1F1B helps** — every ubatch size improves 2-7 % over V2.25.d serial
dispatch. **Sync still wins** by 7 % at the best-tuned config.

## Why async doesn't net-win at this scale

Pipeline depth analysis for 4 ranks × 2 ubatches (the L=1024 ubatch=512
case):

- **Fill** (t=0..N-1 = 0..3): each step adds one rank to the pipeline.
  During fill, only (t+1) ranks are active. Steady-state efficiency
  during fill = (t+1)/N per step.
- **Steady** (t=N-1..K-1 = 3..1): all N ranks active. Only 1 step of
  steady state here (t=3).
- **Drain** (t=K..K+N-2 = 2..4): each step removes one rank. Drain
  efficiency mirrors fill.

Total active "rank-steps" = 1+2+3+4+4+3+2+1 = 20. Best-case serial (no
overlap) = 2 ubatches × 4 ranks = 8 rank-timesteps. Ideal speedup =
(4 × 2) / 5 timesteps = 1.6×. Measured: 711/696 = 1.02× over V2.25.d,
but 0.93× vs sync — so we're not even recovering to serial.

The infrastructure overhead (Mutex/Event/closure per dispatch) exceeds
the overlap benefit at this scale. Real async-PP wins require
**pipeline depth × ubatches >> overhead per dispatch**. For N=4, that
means K ≥ ~16 ubatches — i.e. L ≥ 16 × ubatch_size. At 9B Q4_1 that
would be L ≥ 8192 — way beyond our current prefill sweep.

## Would 1F1B win at larger L?

Projection (assumes ideal pipeline speedup):
- L=4096 ubatch=256 (K=16, N=4): steady = 13 steps of 4 ranks, fill+drain = 6 steps. Ideal 16/(16+3) = 0.84× per-rank efficiency — but 4× speedup over sequential. Sync at this L would saturate other serial bits (embed, output head).
- L=8192 ubatch=256 (K=32): ideal 32/(32+3) = 0.91×, 4× speedup.

Turbo's 1569 tok/s at Mesh<4> pp=1024 corresponds to ~1 ms/layer/token
at full parallelism — similar ballpark to what our async could reach
at K ≥ 16.

## Deferred — V2.26 prefill PP at higher L

Proper win on async-PP prefill needs:
- Test harness extended to L ∈ {2048, 4096, 8192} (memory permitting)
- Tune ubatch_size + u_lanes for each L
- Compare to sync at same L (which also gains from larger batch sizes,
  independent of PP)

Filed for V2.26 cycle. For V2.25 the infrastructure + 1F1B scheduling
is validated correct; perf regression is documented.

## Gate

- UD-Q4_K_S 8-token parity bit-exact (async path not triggered at L=1)
- last_id=220 bit-exact on every ubatch size — async path produces
  same output as sync
- Build clean, opt-in only (default path unchanged)

## Commit summary — V2.25 complete

| step | commit | delta to final |
|---|---|---|
| a: aux streams | `7e05545` | infra (dormant) |
| b: hipEvent + async peer-copy | `b88fa62` | infra |
| c: u_lanes scratch | `711949d` | infra |
| d: async ubatch loop | `0b452d3` | serial dispatch, −9 % |
| e: output head epilogue | `3879ba9` | cleanup |
| f: ubatch × u_lanes sweep | `0ae0d96` | measurement |
| g: per-lane bounces + non-blocking streams | `9c3b16e` | null, disproved bounce hypothesis |
| **h: 1F1B dispatch** | (this) | **+7 % over d, −7 % vs sync** |

Net V2.25 status:
- **Infrastructure complete**: `HipCluster.aux_streams`,
  `lane_bounces`, `HipEvent`, `peer_copy_via_host_async_laned`,
  `new_non_blocking` streams, `u_lanes`-parameterised scratch.
- **Opt-in async PP runs correctly** — parity bit-exact.
- **Net perf regression at L=1024** (best async 711 vs sync 767).
- **Real win deferred to V2.26** (longer L, more ubatches per rank
  to amortise overhead).
