#!/usr/bin/env bash
# 4-slot concurrent probe for llama-server (batched aggregate throughput).
set -uo pipefail
PORT="${1:-18021}"
PJ=$(python3 -c 'import json;print(json.dumps("Write a detailed essay about the history of computing."))')
rm -f /tmp/lcpp-par-*.json
T0=$(date +%s.%N)
for i in 1 2 3 4; do
  curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"qwen3.8\",\"messages\":[{\"role\":\"user\",\"content\":$PJ}],\"max_tokens\":160,\"temperature\":0,\"ignore_eos\":true}" \
    > "/tmp/lcpp-par-$i.json" &
done
wait
T1=$(date +%s.%N)
python3 - <<'EOF'
import json, glob
n = 0; tps = []
for f in sorted(glob.glob('/tmp/lcpp-par-*.json')):
    try:
        d = json.load(open(f)); t = d.get('timings') or {}
        n += t.get('predicted_n') or d['usage']['completion_tokens']
        if t.get('predicted_per_second'): tps.append(round(t['predicted_per_second'], 1))
    except Exception as e:
        print(f, 'ERR', e)
print('gen tokens total:', n)
print('per-request tok/s:', tps)
EOF
awk -v a="$T0" -v b="$T1" 'BEGIN{printf "WALL: %.2fs
", b-a}'
