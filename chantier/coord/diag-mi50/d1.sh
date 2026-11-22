#!/bin/bash
# D1: rocprofv3 kernel trace, MTP-4 TP-2, V2/V3 alternated. Usage: d1.sh tag bin
set -u
D=/tmp/claude-1000/-home-artefact-Workspace-flambeau/c651d041-9a81-4a77-af90-27f026cd6d36/scratchpad/diag; tag=$1; bin=$2
M=$HOME/models/Qwen3.8-27B-UD-Q4_K_M.gguf
bash /tmp/claude-1000/-home-artefact-Workspace-flambeau/c651d041-9a81-4a77-af90-27f026cd6d36/scratchpad/wait_mi50s.sh
rocm-smi --showtemp | grep -i junction | tr -s ' ' | cut -d' ' -f1,2,8- > $D/$tag.temps
RUST_LOG="info,mtp.cycle=debug" setsid rocprofv3 --kernel-trace --output-format csv -d $D/$tag -o run -- $bin serve --model $M --ctx-cap 8192 --inflight-slots 1 --kv q8 --prefill-ubatch 512 --devices hip:0,1 --mesh-mode tp --tp-size 2 --mtp 4 --port 8097 > $D/$tag.log 2>&1 &
pid=$!
for i in $(seq 1 300); do curl -sf http://127.0.0.1:8097/health >/dev/null && break; sleep 2; done
curl -sf http://127.0.0.1:8097/health >/dev/null || { echo "$tag NOT UP"; tail -20 $D/$tag.log; }
python3 -I $D/req.py 8097 $D/$tag.json $HOME/Workspace/flambeau-exp/v3/lp_g.py || echo "$tag REQ FAIL"
spid=$(pgrep -f "$bin serve --model" | head -1)
kill -INT $spid 2>/dev/null
for i in $(seq 1 60); do kill -0 $pid 2>/dev/null || break; sleep 2; done
kill -0 $pid 2>/dev/null && { echo "$tag still alive, TERM"; pkill -TERM -f "$bin serve --model"; sleep 5; pkill -KILL -f "$bin serve --model"; sleep 3; }
pgrep -f "serve --model" >/dev/null && echo "WARN server alive" || echo "$tag done"
find $D/$tag -name '*kernel_trace.csv' | xargs -r wc -l
