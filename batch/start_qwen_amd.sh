#!/bin/bash
# Qwen3.8-27B on one AMD Radeon AI PRO R9700 (RDNA4 / gfx1201, 32 GB) --
# BATCH / THROUGHPUT mode. ROCm port of batch/start_qwen.sh; that script has
# the full rationale for every shared knob. What changes on AMD
# (docs/amd.md has the complete map):
#
#  - KV=fp8 (the NVIDIA default) is FlashInfer's fp8 attention -- no
#    FlashInfer on RDNA4, so it is refused. The default here is instead:
#  - KV=int8pth: int8-per-token-head KV on vLLM's Triton attention backend.
#    Same 'half-width KV' idea as fp8 with a portable kernel; ~131k context by
#    default. KV=bf16 is the quality-neutral fallback (~64k).
#  - INT8_ACT (W4A8 Marlin int8 tensor-core GEMMs): Marlin does not exist on
#    ROCm, so this knob is INERT here and a warning prints if you set it.
#    The other batch wins all survive: W4A16 weights via the ROCm fallback GEMM
#    path, --mamba-ssm-cache-dtype float16, prefix caching, tuned batching.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Stale offload regions -- unchanged from the NVIDIA script.
if [ "${VLLM_OFFLOAD_KEEP_SHM:-0}" != 1 ]; then
  for f in /dev/shm/vllm_offload_*.mmap; do
    [ -e "$f" ] || continue
    grep -lqs "$f" /proc/[0-9]*/maps 2>/dev/null || { echo "[start_qwen_amd] removing stale offload region $f"; rm -f "$f"; }
  done
fi
REPO="$(dirname "$DIR")"
cd "$REPO"

MODEL=${MODEL:-$REPO/models/Qwen3.8-27B-W4A16-AutoRound}
PORT=${PORT:-18020}
MAX_SEQS=${MAX_SEQS:-64}
API_SERVERS=${API_SERVERS:-1}
# KV=int8pth (default on AMD): int8 per-token-head KV + Triton attention.
# KV=bf16: full-precision KV, smallest context, zero quantization risk.
# KV=kvarn: the KVarN 4/2-bit cache (bash kvarn/install.sh once) -- EXPERIMENTAL
#   on RDNA4 until its Triton kernels are numerically validated there.
# KV=fp8 is REFUSED: that path needs FlashInfer.
KV=${KV:-int8pth}
if [ "$KV" = "int8pth" ]; then
  MAX_LEN=${MAX_LEN:-131072}
  GPU_UTIL=${GPU_UTIL:-0.93}
  KV_ARGS="--kv-cache-dtype int8_per_token_head --attention-backend TRITON_ATTN"
elif [ "$KV" = "bf16" ]; then
  MAX_LEN=${MAX_LEN:-65536}
  GPU_UTIL=${GPU_UTIL:-0.972}
  KV_ARGS="--kv-cache-dtype bfloat16"
elif [ "$KV" = "kvarn" ]; then
  MAX_LEN=${MAX_LEN:-262144}
  GPU_UTIL=${GPU_UTIL:-0.93}
  KV_ARGS="--kv-cache-dtype kvarn_k4v2_g128 --block-size 128"
  export KVARN_POOL_MEM_FRAC=${KVARN_POOL_MEM_FRAC:-0.25}
elif [ "$KV" = "fp8" ]; then
  echo "KV=fp8 needs FlashInfer, which does not build for RDNA4." >&2
  echo "  Use KV=int8pth (default), KV=bf16, or KV=kvarn (experimental)." >&2
  exit 1
else
  echo "KV=$KV unknown (int8pth | bf16 | kvarn)" >&2
  exit 1
fi
# int8 activations: accepted for .env compatibility with the NVIDIA stack but
# inert here -- without Marlin kernels nothing reads VLLM_MARLIN_INPUT_DTYPE,
# and exporting the empty string to *disable* it kills the engine at startup
# (env_with_choices rejects '', issue #20). So: warn and skip entirely.
if [ -n "${INT8_ACT+x}" ] && [ -n "$INT8_ACT" ]; then
  echo "[start_qwen_amd] note: INT8_ACT=$INT8_ACT ignored -- W4A8 Marlin does not exist on ROCm;" \
       "batch runs pure W4A16 through vLLM's fallback GEMM path."
fi
INT8_ACT=

# PREFIX_CACHE -- unchanged logic and numbers.
if [ "${PREFIX_CACHE:-0}" = "1" ]; then
  EXTRA_ARGS="--enable-prefix-caching --mamba-cache-mode align ${EXTRA_ARGS}"
fi

# Tool calling -- qwen3_coder reads the XML format this chat template emits.
TOOL_PARSER=${TOOL_PARSER:-qwen3_coder}
TOOL_ARGS=$([ "${TOOLS:-1}" = 1 ] && echo --enable-auto-tool-choice --tool-call-parser $TOOL_PARSER)

# Vision tower offload -- torch ops, platform-neutral.
if [ "${VISION:-0}" = 1 ]; then
  VISION_ARGS='--limit-mm-per-prompt {"image":{"count":1}} --mm-processor-kwargs {"size":{"shortest_edge":65536,"longest_edge":2097152}}'
  [ "${VISION_OFFLOAD:-1}" = 1 ] && export VLLM_VISION_CPU_OFFLOAD_GB=${VLLM_VISION_CPU_OFFLOAD_GB:-1}
else
  VISION_ARGS="--language-model-only"
fi

export PATH="$REPO/venv/bin:$PATH"
# No expandable_segments default on ROCm (CUDA-VMM feature); see start_qwen_amd
# in single-user/. No flashinfer sampler either.
export VLLM_USE_FLASHINFER_SAMPLER=0

# API key: api_key.txt in the repo root, or export VLLM_API_KEY.
if [ -z "$VLLM_API_KEY" ] && [ -f "$REPO/api_key.txt" ]; then
  export VLLM_API_KEY="$(cat "$REPO/api_key.txt")"
fi

exec venv/bin/vllm serve "$MODEL" \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port $PORT \
  --gpu-memory-utilization $GPU_UTIL \
  --max-model-len $MAX_LEN \
  --max-num-seqs $MAX_SEQS \
  --api-server-count $API_SERVERS \
  ${VISION_ARGS} \
  $KV_ARGS \
  --mamba-ssm-cache-dtype float16 \
  --async-scheduling \
  --max-num-batched-tokens 2048 \
  --compilation-config '{"max_cudagraph_capture_size":64,"custom_ops":["+rms_norm","+silu_and_mul"]}' \
  --reasoning-parser qwen3 \
  ${TOOL_ARGS} \
  ${EXTRA_ARGS}