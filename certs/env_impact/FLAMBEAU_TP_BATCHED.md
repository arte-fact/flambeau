# Env-impact cert: `FLAMBEAU_TP_BATCHED`

- **Class:** B
- **Default state:** `on`
- **Generated:** 2026-05-05T22:00:00+00:00 (manual cert from partial run)
- **Spec runs/cell:** 5 measured / 0 warmup observed; harness killed mid-run
- **Run note:** Sweep stopped after 5 calls of the `=0` opt-out cell on
  qwen36-27b-q4_0/tp2 because each call took ~150 s (vs ~14 s for the
  default). Continuing would have spent ~50 min just on this one var.
  Data below is the observed median over the calls that completed.

**Description.** Batched TP forward (default ON; opt-out gate via `=0`).

## Measured cells

### qwen36-27b-q4_0 / tp2

| value | n | ttft median (ms) | total wall (ms) | decode tok/s | Δ ttft vs default | correct? |
|---|---:|---:|---:|---:|---:|---|
| `unset` (default) | 5 | 11,438 | 14,499 | ~20.6 | — | — |
| `0` (opt-out) | 3 | 149,396 | 152,547 | ~23.7 | **+13.06×** worse | match |

### qwen36-27b-q4_0 / pp2tp2 — NOT MEASURED (sweep killed)
### qwen36-35b-a3b-q4_0 / tp2 — NOT MEASURED
### qwen36-35b-a3b-q4_0 / pp2tp2 — NOT MEASURED

## Triage

- qwen36-27b-q4_0/tp2 `0` → **HALT-DEAD-PATH-SLOW** (ttft +13× regression,
  decode unchanged, output bit-identical)

## Disposition

- **KEEP-DEFAULT** — TP_BATCHED=0 is a 13× prefill regression with no
  compensating benefit (decode rate unchanged, output text identical).
  The opt-out path is dead. Two follow-ups:
  1. **Migrate the gate to dispatch-table-internal-only**: every code
     path keyed off `TP_BATCHED` should be either default-baked or
     deleted; the env-flag is the wrong layer.
  2. **File a separate issue on the `=0` slow path itself** if the
     non-batched TP code is meant to remain functional (e.g. for
     correctness fall-back). At the current 13× regression on a
     long prompt, it's not a usable fall-back; either fix or delete.

## Methodological note (HARNESS BUG)

The harness's `triage_decision()` uses **decode tok/s only** to triage.
On this var, decode is unchanged but prefill collapses 13×, so a naive
`tg64` triage would have classified `=0` as `null` or even `win`
(decode 23.7 > 20.6).

The cert must also report and triage on **prefill_ms** delta. Patch the
harness disposition to escalate prefill regressions ≥ 5× to
`HALT-DEAD-PATH-SLOW`. (Filed: harness has `prefill_ms_median` already
captured; the disposition function just needs to consult it.)

## Notes from spec

> Read in models/tp.rs:797,860. Opt-out cost not re-measured since
> landing. If opt-out is materially slower, delete the gate (no reason
> to keep dead path).

Confirmed: opt-out is materially slower. Delete the gate.
