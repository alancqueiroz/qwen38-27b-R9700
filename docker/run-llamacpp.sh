#!/usr/bin/env bash
# llama.cpp server launcher for the R9700 (single GPU).
# Mirrors the knob-style of start_qwen_amd.sh: SPEC / CTX / VISION / PORT.
# Defaults are the max-safe-context profile: 128k ctx with q8_0 K/V cache
# (~24-25 GiB total incl. drafter, measured headroom ~7 GiB on the 32 GB card).
set -euo pipefail

MODEL="${MODEL:-/app/gguf/Qwen3.8-27B-Q5_K_M.gguf}"
SPEC="${SPEC:-none}"
DRAFT="${DRAFT:-}"
CTX="${CTX:-131072}"
KV_DTYPE="${KV_DTYPE:-q8_0}"          # q8_0 (default, 2x context per GiB) | bf16
NGL="${NGL:-999}"
PORT="${PORT:-18021}"
PARALLEL="${PARALLEL:-1}"
VISION="${VISION:-0}"
MMPROJ="${MMPROJ:-/app/gguf/mmproj-F16.gguf}"

ARGS=(
  --model "$MODEL"
  --ctx-size "$CTX"
  --n-gpu-layers "$NGL"
  --port "$PORT"
  --host 0.0.0.0
  --parallel "$PARALLEL"
  --jinja
)

# -fa is default-on for HIP builds in recent llama.cpp; keep it explicit so the
# rocWMMA gfx12 path is what actually runs (built via GGML_HIP_ROCWMMA_FATTN_GFX12).
ARGS+=(--flash-attn on)

# K/V cache quantization: q8_0 halves KV bytes (131k ctx fits comfortably);
# KV_DTYPE=bf16 restores full fidelity at half the context budget.
if [ "$KV_DTYPE" != bf16 ]; then
  ARGS+=(--cache-type-k "$KV_DTYPE" --cache-type-v "$KV_DTYPE")
fi

# Vision: attach the F16 multimodal projector (~0.9 GiB). Images then consume
# context tokens, so keep prompt+images+output within CTX.
if [ "$VISION" = 1 ]; then
  if [ -f "$MMPROJ" ]; then
    ARGS+=(--mmproj "$MMPROJ")
    echo "[run-llamacpp] vision ON via $MMPROJ"
  else
    echo "[run-llamacpp] VISION=1 but mmproj not found at $MMPROJ — starting text-only" >&2
  fi
fi

case "$SPEC" in
  mtp)
    # The unsloth main GGUF carries the native nextn (MTP) layer, so the
    # sidecar is optional: DRAFT empty => in-model MTP head.
    if [ -n "$DRAFT" ]; then ARGS+=(--model-draft "$DRAFT"); fi
    ARGS+=(--spec-type draft-mtp --spec-draft-ngl 999)
    ;;
  dflash)
    [ -n "$DRAFT" ] || { echo "[run-llamacpp] SPEC=dflash requires DRAFT=<dflash sidecar gguf>" >&2; exit 2; }
    ARGS+=(--model-draft "$DRAFT" --spec-type draft-dflash --spec-draft-ngl 999)
    ;;
  none) : ;;
  *) echo "[run-llamacpp] unknown SPEC='$SPEC' (none|mtp|dflash)" >&2; exit 2 ;;
esac

# Optional extras appended verbatim last (override anything above).
if [ -n "${EXTRA_ARGS:-}" ]; then read -r -a EXTRA <<< "$EXTRA_ARGS"; ARGS+=("${EXTRA[@]}"); fi

echo "[run-llamacpp] llama.cpp $(cat /opt/llama.cpp-commit.txt 2>/dev/null || echo '?')"
echo "[run-llamacpp] exec llama-server ${ARGS[*]}"
exec llama-server "${ARGS[@]}"
