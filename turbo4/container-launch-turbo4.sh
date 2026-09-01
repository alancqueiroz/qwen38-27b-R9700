#!/usr/bin/env bash
# Launcher TurboQuant (fork AtomicBot): Qwen3.8-27B + visao + 96K + KV turbo4 + NextN MTP
set -euo pipefail
LL="$(command -v llama-server)"
ARGS=(-m "${MODEL:?MODEL nao definido}")
ARGS+=(-c "${CTX:-98304}" -ngl "${NGL:-999}" -fa on)
ARGS+=(-ctk "${CTK_DTYPE:-turbo4}" -ctv "${CTV_DTYPE:-turbo4}")
ARGS+=(--jinja -t "${THREADS:-8}" -tb "${THREADS_BATCH:-16}")
ARGS+=(-b "${BATCH:-1024}" -ub "${UBATCH:-1024}" -np "${PARALLEL:-1}")
ARGS+=(--host 0.0.0.0 --port "${PORT:-8091}")
if [ "${VISION:-1}" = "1" ] && [ -n "${MMPROJ:-}" ]; then
  ARGS+=(--mmproj "${MMPROJ}")
  echo "[turbo4] visao ON via ${MMPROJ}"
fi
SPEC="${SPEC:-nextn}"
case "$SPEC" in
  nextn)
    # caminho shared-model: -md = mesmo arquivo do modelo (usa os nextn.* embutidos no GGUF)
    ARGS+=(--spec-type nextn -ngld "${DRAFT_NGL:-999}")
    if [ -n "${MODEL_DRAFT:-}" ]; then ARGS+=(-md "${MODEL_DRAFT}"); else ARGS+=(-md "${MODEL}"); fi
    [ -n "${DRAFT_MAX:-}" ] && ARGS+=(--spec-draft-n-max "${DRAFT_MAX}")
    [ -n "${DRAFT_MIN:-}" ] && ARGS+=(--spec-draft-n-min "${DRAFT_MIN}")
    ;;
  none) : ;;
  *) echo "[turbo4] SPEC='$SPEC' invalido (nextn|none)" >&2; exit 2 ;;
esac
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=(${EXTRA_ARGS})
echo "[turbo4] exec: $LL ${ARGS[*]}"
exec "$LL" "${ARGS[@]}"
