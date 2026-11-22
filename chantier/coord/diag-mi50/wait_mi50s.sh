#!/bin/bash
# Wait until both MI50s (GPU 0 and 1) have junction <= 56 C and VRAM use < 5 %.
reading() {
  local d=$1 t v
  t=$(rocm-smi -d "$d" --showtemp 2>/dev/null | tr -d '\r' | grep -i junction | grep -o "[0-9.]*$" | head -1)
  v=$(rocm-smi -d "$d" --showmemuse 2>/dev/null | tr -d '\r' | grep -a "VRAM%" | grep -o "[0-9]\+$" | head -1)
  echo "${t:-99} ${v:-99}"
}
while true; do
  read t0 v0 < <(reading 0)
  read t1 v1 < <(reading 1)
  awk -v a="$t0" -v b="$v0" -v c="$t1" -v d="$v1" 'BEGIN{exit !(a<=56 && b<5 && c<=56 && d<5)}' && break
  sleep 10
done
echo "GUARD MI50s: gpu0 junction=${t0} vram=${v0}% gpu1 junction=${t1} vram=${v1}%"
