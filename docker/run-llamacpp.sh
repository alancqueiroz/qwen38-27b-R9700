#!/usr/bin/env bash
# llama.cpp server launcher for the R9700 (single GPU).
# Mirrors the knob-style of start_qwen_amd.sh: SPEC / CTX / DFLASH_* / PORT.
set -euo pipefail

MODEL="${MODEL:-/app/gguf/Qwen3.8-27B-Q5_K_M.gguf}"
SPEC="${SPEC:-none}"
DRAFT="${DRAFT:-}"
CTX="${CTX:-65536}"
NGL="${NGL:-999}"
PORT="${PORT:-18021}"
PARALLEL="${PARALLEL:-1}"

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

# Optional extras (e.g. --cache-type-k q8_0 --cache-type-v q8_0 for long ctx)
if [ -n "${EXTRA_ARGS:-}" ]; then read -r -a EXTRA <<< "$EXTRA_ARGS"; ARGS+=("${EXTRA[@]}"); fi

echo "[run-llamacpp] llama.cpp $(cat /opt/llama.cpp-commit.txt 2>/dev/null || echo '?')"
echo "[run-llamacpp] exec llama-server ${ARGS[*]}"
exec llama-server "${ARGS[@]}"
