#!/usr/bin/env bash
# Single-stream probe against llama-server (OpenAI-compatible API).
# Usage: probe_llamacpp.sh [PORT] [MAX_TOKENS] [N_REQ]
# Same protocol as the vLLM probes: temp 0, ignore_eos, 4 x 160-token essays
# (MAX_TOKENS/N_REQ overridable), sequential (one at a time).
set -euo pipefail
PORT="${1:-18021}"
MAXT="${2:-160}"
N="${3:-4}"
PROMPT_JSON=$(python3 -c 'import json;print(json.dumps("Write a detailed essay about the history of computing, covering mechanical calculators, early electronic computers, transistors, microprocessors, personal computers, the internet, and mobile computing. Be thorough and technical."))')

TOTAL_T0=$(date +%s.%N)
for i in $(seq 1 "$N"); do
  curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"qwen3.8\",\"messages\":[{\"role\":\"user\",\"content\":$PROMPT_JSON}],\"max_tokens\":$MAXT,\"temperature\":0,\"ignore_eos\":true}" >/tmp/lcpp-probe-$i.json
done
TOTAL_T1=$(date +%s.%N)

python3 - <<'EOF'
import json, glob
tot_n=0; tot_s=0.0
for f in sorted(glob.glob('/tmp/lcpp-probe-*.json')):
    d=json.load(open(f))
    t=d.get('timings') or d.get('usage') or {}
    pn=t.get('prompt_n') or d['usage']['prompt_tokens']
    gn=t.get('predicted_n') or d['usage']['completion_tokens']
    ms=t.get('predicted_ms') or 0
    tps=t.get('predicted_per_second') or (gn/(ms/1000) if ms else 0)
    tot_n+=gn; tot_s+=ms/1000 if ms else 0
    print(f"req {f}: gen={gn} tok/s={tps:.1f} prompt={pn}")
EOF
echo "TOTAL wall: $(echo "$TOTAL_T1 $TOTAL_T0" | awk '{printf "%.2f", $1-$2}')s"
