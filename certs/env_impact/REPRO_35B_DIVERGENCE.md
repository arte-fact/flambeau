# 35B-A3B-Q4_0 / pp2tp2 multi-slot divergence — reproduction

- **Generated:** 2026-05-07T10:12:06+00:00
- **Model:** qwen36-35b-a3b-q4_0 on 0,2,1,3 (pp+tp)
- **Slots:** 2 | **Concurrent:** 2 | **ctx_cap:** 4096 | **tg_len:** 64 | seed=0

## Verdict

- **H1-RACE confirmed** — same env, two boots produce different text. There is a true within-env race in the multi-slot decode path.

## H1 test (same env, two boots)

| run | seed | sha256[:16] | err |
|---|---|---|---|
| A baseline v1 | 0 | `02a16e1143bcea1f` |  |
| A baseline v1 | 1 | `209d8f811a9deccc` |  |
| B baseline v2 | 0 | `02a16e1143bcea1f` |  |
| B baseline v2 | 1 | `02a16e1143bcea1f` |  |

- **A vs B (seed 0) text equal:** `True`
- **A vs B (seed 1) text equal:** `False`

## H2 test (default batched-GDN vs per-token GDN)

| run | sha256[:16] | err |
|---|---|---|
| A baseline (batched) | `02a16e1143bcea1f` |  |
| C per-token (=1)     | `02a16e1143bcea1f` |  |

- **A vs C text equal:** `True`
- **A vs C common prefix chars:** 356
- **A vs C Levenshtein distance:** 0
- **A length:** 356
- **C length:** 356

## Captured texts

### A_baseline_v1  env_extras={}

**seed=0:**

```
The Linux kernel scheduler is a complex subsystem designed to manage the execution of threads across multiple CPU cores efficiently. It operates through a combination of data structures, trigger events, and algorithms that balance fairness, throughput, and real-time requirements.

### Data Structures and Run Queues

At the heart of the scheduler is the `
```

**seed=1:**

```
The Linux kernel scheduler is a complex subsystem designed to manage the execution of threads across multiple CPU cores efficiently. It operates through a combination of data structures, trigger events, and algorithms that balance fairness, throughput, and power consumption.

### Data Structures and Run Queues

At the heart of the scheduler is the `kernel
```

### B_baseline_v2  env_extras={}

**seed=0:**

```
The Linux kernel scheduler is a complex subsystem designed to manage the execution of threads across multiple CPU cores efficiently. It operates through a combination of data structures, trigger events, and algorithms that balance fairness, throughput, and real-time requirements.

### Data Structures and Run Queues

At the heart of the scheduler is the `
```

**seed=1:**

```
The Linux kernel scheduler is a complex subsystem designed to manage the execution of threads across multiple CPU cores efficiently. It operates through a combination of data structures, trigger events, and algorithms that balance fairness, throughput, and real-time requirements.

### Data Structures and Run Queues

At the heart of the scheduler is the `
```

### C_per_token_gdn  env_extras={'FLAMBEAU_GDN_NO_BATCHED': '1'}

**seed=0:**

```
The Linux kernel scheduler is a complex subsystem designed to manage the execution of threads across multiple CPU cores efficiently. It operates through a combination of data structures, trigger events, and algorithms that balance fairness, throughput, and real-time requirements.

### Data Structures and Run Queues

At the heart of the scheduler is the `
```

**seed=1:**

```
The Linux kernel scheduler is a complex subsystem designed to manage the execution of threads across multiple CPU cores efficiently. It operates through a combination of data structures, trigger events, and algorithms that balance fairness, throughput, and real-time requirements.

### Data Structures and Run Queues

At the heart of the scheduler is the `
```
