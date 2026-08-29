#!/usr/bin/env python3
"""Verbatim-reproduction probe: the condition the n-gram chain targets.

A long repeating document is given as context; the model must continue it
verbatim. While the request reproduces its own context, the DFlash2 chain
proposes whole verify blocks from history alone (drafter skipped). Compare
wall/tok-s with VLLM_DFLASH2_CHAIN on/off; outputs must be identical (greedy).

Usage: probe_verbatim.py [PORT] [MAXT] [TAG]
"""
import json, sys, time, urllib.request

port = sys.argv[1] if len(sys.argv) > 1 else "18020"
maxt = int(sys.argv[2]) if len(sys.argv) > 2 else 400
tag = sys.argv[3] if len(sys.argv) > 3 else "run"

para = ("The quarterly report lists the following items: alpha gradient buffer, "
        "beta calibration offset, gamma sampler warmup, delta eviction policy, "
        "epsilon scheduler slot, zeta routing table. ")
doc = para * 120  # ~4900 tokens of self-similar context
prompt = (doc + "\n\n/no_think\n\nAbove is the quarterly report. Copy the report's full text "
          "one more time, exactly as written, without any commentary.\n\nReport:\n")

payload = json.dumps({
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": maxt,
    "temperature": 0,
    "ignore_eos": True,
    "chat_template_kwargs": {"enable_thinking": False},
}).encode()

t0 = time.time()
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=payload, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=1800) as r:
    d = json.loads(r.read())
dt = time.time() - t0
u = d.get("usage", {})
gen = u.get("completion_tokens", 0)
text = d["choices"][0]["message"]["content"] or ""
text = ((d["choices"][0]["message"].get("reasoning") or "") + "\n[CONTENT]\n" + text)
open(f"/tmp/verbatim-{tag}.txt", "w").write(text)
print(f"[{tag}] wall={dt:.1f}s gen={gen} tok/s={gen/dt:.1f} "
      f"prompt={u.get('prompt_tokens')}")
