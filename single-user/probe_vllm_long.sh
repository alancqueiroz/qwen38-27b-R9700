#!/usr/bin/env bash
# Long-context probe for the vLLM stack: ~N_K_FILLERk tokens of filler + a real
# question, then MAXT generated tokens at that depth.
set -euo pipefail
PORT="${1:-18020}"
MAXT="${2:-160}"
FILLER_K="${3:-96}"

python3 - "$FILLER_K" "$MAXT" <<'EOF'
import json, sys
filler_k = int(sys.argv[1]); maxt = int(sys.argv[2])
para = ("The history of computing includes mechanical calculators, early electronic "
        "machines, vacuum tubes, transistors, integrated circuits, microprocessors, "
        "personal computers, networks, and mobile devices. ")  # ~41 tokens
reps = filler_k * 1000 // 41
text = para * reps
payload = {
    "model": "qwen3.8-27b",
    "messages": [
        {"role": "user", "content": text + "\n\nIgnoring the filler above, write a technical summary of Moore's law."}
    ],
    "max_tokens": maxt,
    "temperature": 0,
    "ignore_eos": True,
}
json.dump(payload, open("/tmp/longprobe-payload.json", "w"))
print(f"filler repetitions: {reps}; chars: {len(text)} (~{len(text)//4} tokens)")
EOF

T0=$(date +%s%N)
curl -s http://127.0.0.1:"$PORT"/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/longprobe-payload.json > /tmp/longprobe-out.json
T1=$(date +%s%N)

python3 - <<'EOF'
import json
d = json.load(open('/tmp/longprobe-out.json'))
if "usage" not in d:
    print("RAW:", json.dumps(d)[:400])
else:
    u = d["usage"]
    print("prompt_tokens:", u.get("prompt_tokens"), "completion_tokens:", u.get("completion_tokens"))
EOF
python3 -c "print('WALL: %.2fs' % (($T1-$T0)/1e9))"
