#!/usr/bin/env bash
# Poor-man's sampling profiler using gdb-batch. Pauses the target,
# dumps all-thread backtraces, resumes. Repeat N times. Aggregate
# the leaf frames to find where the host wall is spent.
#
# Usage: ./phase3_gdb_sample.sh <pid> [n_samples=80] [interval_ms=80]
set -euo pipefail
PID="${1:?usage: phase3_gdb_sample.sh <pid> [n] [ms]}"
N="${2:-80}"
INTERVAL_MS="${3:-80}"
OUT="/tmp/gdb_sample_${PID}.txt"
: >"${OUT}"
for i in $(seq 1 "${N}"); do
  sudo -n -- gdb -batch -p "${PID}" \
       -ex "set pagination off" \
       -ex "set print pretty off" \
       -ex "set print frame-arguments none" \
       -ex "set logging file ${OUT}" \
       -ex "set logging redirect on" \
       -ex "set logging enabled on" \
       -ex "thread apply all bt 30" \
       -ex "set logging enabled off" \
       2>/dev/null
  echo "----SAMPLE ${i}----" >> "${OUT}"
  sleep "$(awk "BEGIN{print ${INTERVAL_MS}/1000}")"
done
echo "wrote ${OUT}"
