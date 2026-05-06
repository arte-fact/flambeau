# 35B-A3B-Q4_0 / pp2tp2 — per-layer state-dump localisation

- **Generated:** 2026-05-06T15:39:21+00:00
- **Model:** qwen36-35b-a3b-q4_0 on 0,2,1,3 (pp+tp)
- **Slots:** 8 | **Concurrent:** 2 | **ctx_cap:** 4096 | **tg_len:** 64

## Method

Two boots with `FLAMBEAU_LAYER_STATE_DUMP=1`, 2 concurrent
greedy chat calls per boot, parse `[STATE-DUMP]` lines from
server log, position-diff the two dump streams.

## Top-level outcome

- Boot v1 captured **0** dump lines (errs: none)
- Boot v2 captured **0** dump lines (errs: none)
- Generated text seed=0 match: `True`
- Generated text seed=1 match: `True`

- **No layer-state divergence detected** — both boots' dump streams identical line-for-line. 
  If the generated text *did* diverge (seed=1 match=False above), the race is downstream of the dumped state 
  (e.g. in the output head, sampler, or in a buffer not currently dumped).

## Context window around first divergence

_no divergence_

## Generated text per boot

### Boot v1

seed=0:

```
The Linux kernel scheduler is a complex subsystem designed to manage the execution of threads across multiple CPU cores efficiently. It operates through a combination of data structures, trigger events, and algorithms that balance fairness, throughput, and real-time requirements.

### Data Structures and Run Queues

At the heart of the scheduler is the `
```

seed=1:

```
The Linux kernel scheduler is a complex subsystem designed to manage the execution of threads across multiple CPU cores efficiently. It operates through a combination of data structures, trigger events, and algorithms that balance fairness, throughput, and real-time requirements.

### Data Structures and Run Queues

At the heart of the scheduler is the `
```

### Boot v2

seed=0:

```
The Linux kernel scheduler is a complex subsystem designed to manage the execution of threads across multiple CPU cores efficiently. It operates through a combination of data structures, trigger events, and algorithms that balance fairness, throughput, and real-time requirements.

### Data Structures and Run Queues

At the heart of the scheduler is the `
```

seed=1:

```
The Linux kernel scheduler is a complex subsystem designed to manage the execution of threads across multiple CPU cores efficiently. It operates through a combination of data structures, trigger events, and algorithms that balance fairness, throughput, and real-time requirements.

### Data Structures and Run Queues

At the heart of the scheduler is the `
```
