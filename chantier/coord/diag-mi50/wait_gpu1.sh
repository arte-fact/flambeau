#!/bin/bash
# Wait until MI50 GPU 1 junction <= 54 C and its VRAM use < 5 %.
while true; do
  t=$(rocm-smi -d 1 --showtemp 2>/dev/null | tr -d '\r' | grep -i junction | grep -o "[0-9.]*$" | head -1)
  v=$(rocm-smi -d 1 --showmemuse 2>/dev/null | tr -d '\r' | grep -a "VRAM%" | grep -o "[0-9]\+$" | head -1)
  awk -v x="${t:-99}" -v y="${v:-99}" 'BEGIN{exit !(x<=54 && y<5)}' && break
  sleep 10
done
echo "GUARD GPU1: junction=${t} vram=${v}%"
