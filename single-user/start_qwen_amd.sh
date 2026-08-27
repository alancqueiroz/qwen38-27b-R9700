#!/bin/bash
# Qwen3.8-27B on one AMD Radeon AI PRO R9700 (RDNA4 / gfx1201, 32 GB) --
# SINGLE USER / LOW LATENCY mode. ROCm port of single-user/start_qwen.sh;
# read that script for the full rationale of every shared knob. This file
# documents only what CHANGES for AMD (docs/amd.md has the complete map):
#
#  - Same speculative stack, unchanged: MTP with the own-output draft vocab
#    and the requantized drafter, or SPEC=dflash2; LOOKUP drafting;
#    PREFIX_CACHE. All of it is Python + Triton, which runs on HIP.
#  - No FlashInfer on RDNA4: no fp8 KV cache, so CTX=fast is bf16 KV and
#    CTX=long is the Triton int8-per-token-head path instead of FlashInfer's
#    fp8. The DFlash2 selector falls back from flashinfer.topk to torch.topk
#    (~half the selector's speed; a fraction of a ms per step at batch 1).
#  - No Marlin: W4A16 GEMMs run vLLM's ROCm fallback path. Batch-1 decode is
#    bandwidth-bound, so most of the W4A16 win survives; measure, don't assume.
#  - The split-KV verify attention (patches/spec-decode-attn.patch) is a
#    Triton kernel but its hook lives in the FLASH_ATTN backend class. It is
#    left ON here; whether it fires depends on which backend this build picks
#    -- check the 'Using ... attention backend' log line at boot.
#  - KV pool pins scaled x4/3 from the 24 GB card's values: same proportions,
#    8 GiB more VRAM to spend.
#
# Everything else -- flags, defaults, env names, port -- matches the NVIDIA
# script so bench results are comparable across cards.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# A dead engine leaves its OffloadingConnector region behind as
# /dev/shm/vllm_offload_*.mmap ... (unchanged from the NVIDIA script).
if [ "${VLLM_OFFLOAD_KEEP_SHM:-0}" != 1 ]; then
  for f in /dev/shm/vllm_offload_*.mmap; do
    [ -e "$f" ] || continue
    grep -lqs "$f" /proc/[0-9]*/maps 2>/dev/null || { echo "[start_qwen_amd] removing stale offload region $f"; rm -f "$f"; }
  done
fi
REPO="$(dirname "$DIR")"
cd "$REPO"

if [ -z "$MODEL" ] && [ -d "$REPO/models/Qwen3.8-27B-W4A16-AutoRound-fast" ]; then
  MODEL=$REPO/models/Qwen3.8-27B-W4A16-AutoRound-fast
fi
MODEL=${MODEL:-$REPO/models/Qwen3.8-27B-W4A16-AutoRound}
PORT=${PORT:-18020}
MAX_SEQS=${MAX_SEQS:-}
# 0.93 here, NOT batch mode's higher value: the DeltaNet workspace in the MTP
# decode path allocates beyond the startup memory profile (gotcha 4). On 32 GB
# there is more absolute headroom behind this than on the 3090; keep the
# fraction identical so boot geometry stays comparable.
GPU_UTIL=${GPU_UTIL:-0.93}
API_SERVERS=${API_SERVERS:-1}
# CTX=fast (default): bf16 KV, 4 drafts, 64k context -- the configuration whose
# numbers transfer most directly to this card.
# CTX=long: int8-per-token-head KV on vLLM's Triton attention backend (the
#   same combination SPEC=dflash2 uses at CTX=long upstream of this fork),
#   ~128k context. EXPERIMENTAL on ROCm until measured.
# CTX=huge: KVarN 4/2-bit KV cache (kvarn/), 240k context. The kernels are
#   Triton and should run on HIP unmodified, but nothing here has been
#   numerically validated on RDNA4 yet: EXPERIMENTAL=1 required.
CTX=${CTX:-fast}
SPEC=${SPEC:-mtp}
# SPEC_ATTN=1: split-KV Triton attention for the multi-query verify step;
# bf16 KV only. See the header caveat about which backend it hooks into.
if [ "$CTX" = "fast" ]; then
  MAX_LEN=${MAX_LEN:-65536}
  DRAFT_TOKENS=${DRAFT_TOKENS:-4}
  # No explicit --attention-backend: let the ROCm build pick its default.
  # Forcing FLASH_ATTN would name a CUDA-backend enum that may not exist here.
  ATTN_ARGS="--kv-cache-dtype bfloat16"
  export VLLM_SPEC_DECODE_ATTN=${SPEC_ATTN:-1}
elif [ "$CTX" = "huge" ]; then
  MAX_LEN=${MAX_LEN:-200000}
  DRAFT_TOKENS=${DRAFT_TOKENS:-3}
  ATTN_ARGS="--kv-cache-dtype kvarn_k4v2_g128 --block-size 128"
  export KVARN_POOL_MEM_FRAC=${KVARN_POOL_MEM_FRAC:-0.15}
else
  MAX_LEN=${MAX_LEN:-131072}
  DRAFT_TOKENS=${DRAFT_TOKENS:-3}
  ATTN_ARGS="--attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head"
fi
if [ "$CTX" != "fast" ] && [ "${EXPERIMENTAL:-0}" != "1" ]; then
  echo "CTX=$CTX is experimental on ROCm (unmeasured kernels/geometry)." >&2
  echo "  Re-run with EXPERIMENTAL=1 to accept that." >&2
  exit 1
fi
if [ "$SPEC" = "dflash2" ] && [ "$CTX" = "huge" ]; then
  # KVarN brings its own dequant path for the verify step; keep the split-KV
  # kernel out, exactly as the NVIDIA script does.
  export VLLM_SPEC_DECODE_ATTN=0
fi
# SPEC=dflash2 is BROKEN on RDNA4 as measured (2026-08-26, real 22.7k-token
# corpus): draft acceptance is ~ZERO across all six labd tasks (copy/code at
# 1.00 tok/step, total 1.14) versus MTP's 3.57 on the same tasks and corpus,
# making DFlash2 ~2.6x slower than MTP and slower than no-spec. Generation
# itself is correct (no early EOS; every task runs to the token cap). The
# mechanism: this build selects the ROCM_ATTN attention backend (auto-overridden
# at boot), while the spec-decode-attn split-KV verify hooks and the lookup
# path patch against the FLASH_ATTN backend class used on NVIDIA -- so drafts
# are verified by a fallback that rejects everything. The short-prompt
# benchmarks agree (C1 default below MTP, acceptance collapsing at C4+).
# MTP is healthy on this card and stays the default.
if [ "$SPEC" = "dflash2" ] && [ "${EXPERIMENTAL:-0}" != "1" ]; then
  echo "SPEC=dflash2 measured broken on RDNA4: zero draft acceptance under ROCM_ATTN" >&2
  echo "  (1.14 tok/step vs MTP 3.57 on identical workloads) -- see docs/amd.md." >&2
  echo "  MTP is the healthy default here. EXPERIMENTAL=1 forces it on." >&2
  exit 1
fi
if [ "$SPEC" = "dflash2" ]; then
  if [ -z "$DRAFT" ]; then
    for d in Qwen3.8-27B-DFlash2-W4A16 Qwen3.8-27B-DFlash2; do
      [ -f "$REPO/models/$d/model.safetensors" ] && DRAFT=$REPO/models/$d && break
    done
  fi
  [ -n "$DRAFT" ] || { echo "SPEC=dflash2 needs the drafter: python prepare/fetch_dflash2.py" >&2; exit 1; }
  export VLLM_DFLASH2_LOOKUP=${LOOKUP:-1}
  DRAFT_TOKENS=${DFLASH_TOKENS:-7}
  SPEC_CFG="{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":$DRAFT_TOKENS}"
  export VLLM_SPEC_DECODE_ATTN_QMAX=${VLLM_SPEC_DECODE_ATTN_QMAX:-$((DRAFT_TOKENS + 1))}
  # Adaptive verify length corrupts a prefix-cache hit under KVarN (see the
  # NVIDIA script's long note); pin it under the same conditions.
  if [ -z "${LOOKUP_ADAPTIVE:-}" ] && [ "$VLLM_DFLASH2_LOOKUP" = "1" ] && [ "$DRAFT_TOKENS" -gt 7 ] \
     && [ "$CTX" = "huge" ] && [ "${PREFIX_CACHE:-0}" = "1" ]; then
    echo "DFLASH_TOKENS>7 + CTX=huge + PREFIX_CACHE=1: pinning the verify block to" >&2
    echo "$((DRAFT_TOKENS + 1)) tokens (VLLM_DFLASH2_LOOKUP_ADAPTIVE=0)." >&2
    export VLLM_DFLASH2_LOOKUP_ADAPTIVE=0
  fi
  if [ "$VLLM_DFLASH2_LOOKUP" = "1" ] && [ "$DRAFT_TOKENS" -gt 7 ]; then
    ASYNC_SCHED=${ASYNC_SCHED:-0}
  fi
  # Memory pins. The 24 GB card's constants, scaled x4/3 for the extra 8 GiB:
  # same shape of budget, bigger absolute pool. Override with KV_MEM as usual.
  SCALE_NUM=4; SCALE_DEN=3
  kv_scaled() { echo $(( ($1 * SCALE_NUM + SCALE_DEN - 1) / SCALE_DEN )); }
  if [ "$CTX" = "huge" ]; then
    MAX_SEQS=${MAX_SEQS:-2}
    if [ "$DRAFT_TOKENS" -gt 7 ]; then
      MAX_LEN=${DFLASH_MAX_LEN:-221184}
    else
      MAX_LEN=${DFLASH_MAX_LEN:-245760}
    fi
    KV_MEM=${KV_MEM-$(kv_scaled 5261334938)}
    if [ "$DRAFT_TOKENS" -gt 7 ]; then
      export VLLM_V2_CUDAGRAPH_MEM_MIB=${VLLM_V2_CUDAGRAPH_MEM_MIB:-1900}
    else
      export VLLM_V2_CUDAGRAPH_MEM_MIB=${VLLM_V2_CUDAGRAPH_MEM_MIB:-1400}
    fi
  elif [ "$CTX" = "long" ]; then
    MAX_SEQS=${MAX_SEQS:-4}
    MAX_LEN=${DFLASH_MAX_LEN:-131072}
    KV_MEM=${KV_MEM-$(kv_scaled 5583457484)}
    if [ "$DRAFT_TOKENS" -gt 7 ]; then
      export VLLM_V2_CUDAGRAPH_MEM_MIB=${VLLM_V2_CUDAGRAPH_MEM_MIB:-1900}
    else
      export VLLM_V2_CUDAGRAPH_MEM_MIB=${VLLM_V2_CUDAGRAPH_MEM_MIB:-1400}
    fi
  elif [ "$DRAFT_TOKENS" -gt 7 ]; then
    MAX_SEQS=${MAX_SEQS:-4}
    MAX_LEN=${DFLASH_MAX_LEN:-57344}
    KV_MEM=${KV_MEM-$(kv_scaled 5583457484)}
    export VLLM_V2_CUDAGRAPH_MEM_MIB=${VLLM_V2_CUDAGRAPH_MEM_MIB:-1900}
  else
    MAX_LEN=${DFLASH_MAX_LEN:-65536}
    KV_MEM=${KV_MEM-$(kv_scaled 5583457484)}
    export VLLM_V2_CUDAGRAPH_MEM_MIB=${VLLM_V2_CUDAGRAPH_MEM_MIB:-1400}
  fi
  MAX_SEQS=${MAX_SEQS:-8}
  CG=${CG:-$((MAX_SEQS * (DRAFT_TOKENS + 1) > 64 ? 64 : MAX_SEQS * (DRAFT_TOKENS + 1)))}
  # Seats are admissions, not residency (unchanged arithmetic; the state page
  # is a model property, not a GPU property).
  if [ -n "$KV_MEM" ]; then
    RESIDENT=$(( KV_MEM / ((DRAFT_TOKENS + 2) * 104988089) ))
    [ "$RESIDENT" -lt 1 ] && RESIDENT=1
    if [ "$MAX_SEQS" -gt "$RESIDENT" ]; then
      echo "[start_qwen_amd] note: the pinned pool holds about $RESIDENT resident requests at" \
           "DFLASH_TOKENS=$DRAFT_TOKENS (state pages), fewer with long prompts;" \
           "MAX_SEQS=$MAX_SEQS admits more than that and the rest queue."
    fi
  fi
  # The seat-count cliff was measured on a 24 GiB card (gotcha 39). This card
  # has ~8 GiB more transient headroom, so the sharp floor moves up -- but the
  # mechanism (per-seat allocations eating prefill headroom) is identical.
  # Keep the warning; interpret the exact seat count as 24 GiB-era data.
  if [ "$CTX" = "huge" ] && [ "$MAX_SEQS" -gt 12 ]; then
    echo "[start_qwen_amd] WARNING: MAX_SEQS=$MAX_SEQS at CTX=huge killed a 24 GiB card" \
         "on the first prompt (boots, serves /health, then OutOfMemoryError)." \
         "This 32 GB card has more room; lower MAX_SEQS/KV_MEM if the first prompt OOMs." >&2
  fi
  [ -n "$KV_MEM" ] && EXTRA_ARGS="--kv-cache-memory=$KV_MEM ${EXTRA_ARGS}"
else
  MAX_SEQS=${MAX_SEQS:-8}
  SPEC_CFG="{\"method\":\"mtp\",\"num_speculative_tokens\":$DRAFT_TOKENS,\"draft_sample_method\":\"${DRAFT_SAMPLE:-probabilistic}\"}"
  CG=${CG:-32}
fi

# PREFIX_CACHE=1 -- unchanged logic, same trade as the NVIDIA script.
if [ "${PREFIX_CACHE:-0}" = "1" ]; then
  EXTRA_ARGS="--enable-prefix-caching --mamba-cache-mode align ${EXTRA_ARGS}"
  [ "$CTX" = "huge" ] && EXTRA_ARGS="--prefix-match-unit 128 ${EXTRA_ARGS}"
  if [ "$CTX" = "huge" ] && [ "$SPEC" != "dflash2" ]; then
    CG_MODE=",\"cudagraph_mode\":\"${CUDAGRAPH_MODE:-PIECEWISE}\""
  fi
fi
if [ -n "${CUDAGRAPH_MODE:-}" ] && [ -z "${CG_MODE:-}" ]; then
  CG_MODE=",\"cudagraph_mode\":\"$CUDAGRAPH_MODE\""
fi

# ASYNC_SCHED -- unchanged rationale (worker-chosen verify length needs the
# synchronous scheduler).
ASYNC_ARGS=$([ "${ASYNC_SCHED:-1}" = 1 ] && echo --async-scheduling || echo --no-async-scheduling)

# Tool calling -- qwen3_coder is the format this chat template emits (XML), not
# hermes JSON. Unchanged.
TOOL_PARSER=${TOOL_PARSER:-qwen3_coder}
TOOL_ARGS=$([ "${TOOLS:-1}" = 1 ] && echo --enable-auto-tool-choice --tool-call-parser $TOOL_PARSER)

# Vision tower offload -- torch ops, platform-neutral. Unchanged.
if [ "${VISION:-0}" = 1 ]; then
  VISION_ARGS='--limit-mm-per-prompt {"image":{"count":1}} --mm-processor-kwargs {"size":{"shortest_edge":65536,"longest_edge":2097152}}'
  [ "${VISION_OFFLOAD:-1}" = 1 ] && export VLLM_VISION_CPU_OFFLOAD_GB=${VLLM_VISION_CPU_OFFLOAD_GB:-1}
else
  VISION_ARGS="--language-model-only"
fi

# fp16 activations stay refused with a speculator: the split-KV verify kernel
# hardcodes tl.bfloat16 (patches/spec-decode-attn.patch). Unchanged.
case " ${EXTRA_ARGS:-} " in
  *" --dtype float16 "*|*" --dtype=float16 "*|*" --dtype fp16 "*|*" --dtype=fp16 "*|*" --dtype half "*|*" --dtype=half "*)
    if [ "${SPEC:-mtp}" != "none" ]; then
      echo "--dtype float16 needs SPEC=none: this repo's speculative path is bf16-only." >&2
      exit 1
    fi ;;
esac

export PATH="$REPO/venv/bin:$PATH"
# expandable_segments is a CUDA-VMM allocator feature; torch-on-HIP does not
# provide it, so unlike the NVIDIA script nothing sets PYTORCH_CUDA_ALLOC_CONF
# here. Pass your own value through the environment if you ever need one.

# No flashinfer sampler anywhere near this stack (there is no flashinfer).
export VLLM_USE_FLASHINFER_SAMPLER=0

if [ -z "$VLLM_API_KEY" ] && [ -f "$REPO/api_key.txt" ]; then
  export VLLM_API_KEY="$(cat "$REPO/api_key.txt")"
fi

# The compilation config is built with printf so the double quotes survive any
# transport mangling -- this line once shipped with its escaping stripped and
# argparse rejected the JSON at boot.
printf -v COMP_CFG '{"max_cudagraph_capture_size":%d,"custom_ops":["+rms_norm","+silu_and_mul"]%s}' "$CG" "${CG_MODE:-}"

exec venv/bin/vllm serve "$MODEL" \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port $PORT \
  --gpu-memory-utilization $GPU_UTIL \
  --max-model-len $MAX_LEN \
  --max-num-seqs $MAX_SEQS \
  --api-server-count $API_SERVERS \
  ${VISION_ARGS} \
  $ATTN_ARGS \
  --mamba-ssm-cache-dtype float16 \
  ${ASYNC_ARGS} \
  --max-num-batched-tokens 2048 \
  --speculative-config "$SPEC_CFG" \
  --compilation-config "$COMP_CFG" \
  --reasoning-parser qwen3 \
  ${TOOL_ARGS} \
  ${EXTRA_ARGS}