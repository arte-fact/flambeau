#!/usr/bin/env bash
# v2 serve smoke matrix — runs `v2_smoke.sh` over a sequence of
# (model, mesh, devices) cases and reports pass/fail. Each case
# spawns/kills its own server so GPU memory is fresh per case.

set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SMOKE="$ROOT/scripts/serve/v2_smoke.sh"

# (label, gguf, mesh, devices)
CASES=(
  "qwen35-9B-Q4_1 SD                  | /artefact/models/Qwen3.5-9B-Q4_1.gguf            | pp     | 0"
  "qwen35-9B-Q4_1 TP=2                | /artefact/models/Qwen3.5-9B-Q4_1.gguf            | tp     | 0,1"
  "qwen35-9B-Q4_1 pp2tp2              | /artefact/models/Qwen3.5-9B-Q4_1.gguf            | pp+tp  | 0,2,1,3"
  "qwen35-9B-Q3_K_S SD                | /artefact/models/Qwen3.5-9B-Q3_K_S.gguf          | pp     | 0"
  "qwen35-27B-Q4_0 TP=2               | /artefact/models/Qwen3.5-27B-Q4_0.gguf           | tp     | 0,1"
  "qwen35-27B-Q4_1 TP=2               | /artefact/models/Qwen3.5-27B-Q4_1.gguf           | tp     | 0,1"
  "qwen36-27B-Q4_0 PP=2               | /artefact/models/Qwen3.6-27B-Q4_0.gguf           | pp     | 0,1"
  "qwen36-27B-UD-Q3_K_XL PP=2         | /artefact/models/Qwen3.6-27B-UD-Q3_K_XL.gguf     | pp     | 0,1"
  "qwen36-35B-A3B-Q3_K_S PP=2         | /artefact/models/Qwen3.6-35B-A3B-Q3_K_S.gguf     | pp     | 0,1"
)

PASS=()
FAIL=()
SKIP=()
START=$(date +%s)

for raw in "${CASES[@]}"; do
  IFS='|' read -r label gguf mesh devices <<<"$raw"
  label="$(echo "$label" | sed -e 's/[[:space:]]*$//' -e 's/^[[:space:]]*//')"
  gguf="$(echo "$gguf" | sed -e 's/[[:space:]]*$//' -e 's/^[[:space:]]*//')"
  mesh="$(echo "$mesh" | sed -e 's/[[:space:]]*$//' -e 's/^[[:space:]]*//')"
  devices="$(echo "$devices" | sed -e 's/[[:space:]]*$//' -e 's/^[[:space:]]*//')"
  echo
  echo "================================================================"
  echo " CASE: $label"
  echo "  gguf=$gguf"
  echo "  mesh=$mesh devices=$devices"
  echo "================================================================"
  if [[ ! -f "$gguf" ]]; then
    echo "SKIP (model file missing)"
    SKIP+=("$label")
    continue
  fi
  if bash "$SMOKE" "$gguf" "$mesh" "$devices"; then
    PASS+=("$label")
  else
    FAIL+=("$label")
  fi
  pkill -f 'target/release/flambeau serve' 2>/dev/null
  sleep 2
done

END=$(date +%s)
echo
echo "================================================================"
echo " MATRIX SUMMARY ($((END - START))s)"
echo "================================================================"
echo "  PASS (${#PASS[@]}):"
for c in "${PASS[@]}"; do echo "    ✓ $c"; done
echo "  FAIL (${#FAIL[@]}):"
for c in "${FAIL[@]}"; do echo "    ✗ $c"; done
echo "  SKIP (${#SKIP[@]}):"
for c in "${SKIP[@]}"; do echo "    - $c"; done

if [[ ${#FAIL[@]} -gt 0 ]]; then
  exit 1
fi
