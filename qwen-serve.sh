#!/usr/bin/env bash
# qwen-serve.sh — menu para subir/parar os stacks testados no R9700 (32 GB).
#
#   ./qwen-serve.sh              # menu interativo
#   ./qwen-serve.sh status       # status direto (sem menu)
#   ./qwen-serve.sh stop         # para tudo
#   ./qwen-serve.sh up-128k      # vLLM DFlash2 128K sem menu
#
# Combinações medidas (single-stream, 4x160 tok, temp 0, ignore_eos):
#   vLLM 0.28.0 DFlash2 T=4  64k bf16      ~52.1 tok/s   (a mais rápida)
#   vLLM DFlash2 T=4 · 128k int8           ~43.8 tok/s   (CTX=long)
#   vLLM MTP · 64k                         ~50.7 tok/s
#   llama.cpp DFlash2 z-lab Q8_0           ~41.2 tok/s
#   llama.cpp MTP (nextn in-model)         ~38.4 tok/s
#   llama.cpp plain                        ~24.8 tok/s
set -euo pipefail
cd "$(dirname "$0")"

PORT_VLLM="${PORT_VLLM:-18020}"
PORT_LCPP="${PORT_LCPP:-18021}"
LCPP_DFLASH_DRAFT="${LCPP_DFLASH_DRAFT:-/app/gguf/Qwen3.8-27B-DFlash2-zlab-Q8_0.gguf}"

env_set() { # env_set KEY VALUE — grava no .env sem duplicar
  local k="$1" v="$2"
  touch .env
  if grep -q "^\${k}=" .env; then sed -i "s/^\${k}=.*/${k}=${v}/" .env; else printf '%s=%s\n' "$k" "$v" >> .env; fi
}
env_del() { sed -i "/^${1}=/d" .env 2>/dev/null || true; }

down_vllm() { docker compose -f docker-compose.amd.yml -f docker-compose.port.yml --profile single down >/dev/null 2>&1 || true; }
down_lcpp()  { docker compose -f docker-compose.llamacpp.yml --profile single down >/dev/null 2>&1 || true; }
stop_all() {
  echo "[*] Parando vLLM e llama.cpp..."
  down_lcpp
  down_vllm
}

wait_health() { # wait_health URL LABEL MAX_TRIES(15s cada)
  local url="$1" label="$2" max="${3:-24}" i
  echo -n "[*] Aguardando $label"
  for i in $(seq 1 "$max"); do
    if curl -sf -m 4 "$url" >/dev/null 2>&1; then echo " OK"; return 0; fi
    echo -n "."; sleep 15
  done
  echo " TIMEOUT (logs: docker logs qwen38-27b-rtx3090-amd-single-1 --tail 30)"
  return 1
}

up_vllm() { # up_vllm fast|long
  local ctx="$1"
  stop_all
  if [ "$ctx" = long ]; then env_set CTX long; else env_del CTX; fi
  echo "[*] Subindo vLLM (DFlash2/MTP conforme .env, CTX=$ctx)"
  echo "    primeiro boot frio pode levar ~15 min; com cache de compilação, ~2-4 min"
  docker compose -f docker-compose.amd.yml -f docker-compose.port.yml --profile single up -d --force-recreate
  wait_health "http://127.0.0.1:$PORT_VLLM/health" "vLLM ($ctx)" 720 \
    && echo "    API: http://127.0.0.1:$PORT_VLLM/v1  (modelo servido: qwen3.8-27b)"
}

up_lcpp() { # up_lcpp none|mtp|dflash
  local spec="$1"
  stop_all
  echo "[*] Subindo llama.cpp (SPEC=$spec) na porta $PORT_LCPP"
  if [ "$spec" = dflash ]; then
    SPEC="$spec" DRAFT="$LCPP_DFLASH_DRAFT" docker compose -f docker-compose.llamacpp.yml --profile single up -d --force-recreate
  else
    SPEC="$spec" docker compose -f docker-compose.llamacpp.yml --profile single up -d --force-recreate
  fi
  wait_health "http://127.0.0.1:$PORT_LCPP/health" "llama.cpp" 40 \
    && echo "    API: http://127.0.0.1:$PORT_LCPP/v1  (Qwen3.8-27B-Q5_K_M)"
}

status() {
  echo "=== .env (knobs ativos) ==="
  grep -vE '^\s*(#|$)' .env 2>/dev/null || echo "(sem .env)"
  echo "=== containers ==="
  docker ps --format 'table {{.Names}}\t{{.Status}}' | grep -E 'NAMES|qwen38|llamacpp' || echo "(nenhum)"
  echo "=== health ==="
  curl -s -m 3 "http://127.0.0.1:$PORT_VLLM/health" >/dev/null 2>&1 \
    && echo "vLLM      :$PORT_VLLM UP" || echo "vLLM      : down"
  curl -s -m 3 "http://127.0.0.1:$PORT_LCPP/health" >/dev/null 2>&1 \
    && echo "llama.cpp :$PORT_LCPP UP" || echo "llama.cpp : down"
  echo "=== VRAM (GPU0) ==="
  rocm-smi --showmemuse 2>/dev/null | grep 'GPU\[0\]' || true
}

menu() {
  while true; do
    echo
    echo "=========== Qwen3.8-27B · R9700 — stacks de serving ==========="
    echo " 1) vLLM    DFlash2 T=4 · 64k      ~52 tok/s  (mais rápido)"
    echo " 2) vLLM    DFlash2 T=4 · 128K     ~44 tok/s  [perfil DSH atual]"
    echo " 3) vLLM    MTP · 64k              ~51 tok/s"
    echo " 4) llama.cpp plain                ~25 tok/s"
    echo " 5) llama.cpp MTP (nextn)          ~38 tok/s"
    echo " 6) llama.cpp DFlash2 z-lab Q8_0   ~41 tok/s"
    echo " 7) status (containers / health / VRAM)"
    echo " 8) PARAR tudo"
    echo " 0) sair"
    read -rp "Escolha: " op
    case "$op" in
      1) env_set SPEC dflash2; env_set DFLASH_TOKENS 4; env_set PREFIX_CACHE 1; env_del CTX; up_vllm fast ;;
      2) env_set SPEC dflash2; env_set DFLASH_TOKENS 4; env_set PREFIX_CACHE 1; env_set CTX long; up_vllm long ;;
      3) env_set SPEC mtp; env_del DFLASH_TOKENS; env_del CTX; up_vllm fast ;;
      4) up_lcpp none ;;
      5) up_lcpp mtp ;;
      6) up_lcpp dflash ;;
      7) status ;;
      8) stop_all; echo "[OK] parado." ;;
      0) exit 0 ;;
      *) echo "opção inválida" ;;
    esac
    echo
    read -rp "<Enter> volta ao menu... " _ </dev/tty 2>/dev/null || true
  done
}

case "${1:-}" in
  status) status ;;
  stop) stop_all ;;
  up-64k) env_set SPEC dflash2; env_set DFLASH_TOKENS 4; env_set PREFIX_CACHE 1; env_del CTX; up_vllm fast ;;
  up-128k) env_set SPEC dflash2; env_set DFLASH_TOKENS 4; env_set PREFIX_CACHE 1; env_set CTX long; up_vllm long ;;
  up-mtp) env_set SPEC mtp; env_del DFLASH_TOKENS; env_del CTX; up_vllm fast ;;
  up-lcpp-plain) up_lcpp none ;;
  up-lcpp-mtp) up_lcpp mtp ;;
  up-lcpp-dflash) up_lcpp dflash ;;
  *) menu ;;
esac
