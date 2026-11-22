#!/bin/bash
while true; do
  read t m < <(nvidia-smi -i 0 --query-gpu=temperature.gpu,memory.used --format=csv,noheader,nounits | tr -d ',')
  [ "$t" -le 50 ] && [ "$m" -lt 1024 ] && break
  sleep 10
done
echo "GUARD RTX: temp=$t mem=${m}MiB"
