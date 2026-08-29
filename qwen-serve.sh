#!/usr/bin/env bash
# qwen-serve.sh — menu para subir/parar os stacks testados no R9700 (32 GB).
#
#   ./qwen-serve.sh              # menu interativo
#   ./qwen-serve.sh status       # status direto (sem menu)
#   ./qwen-serve.sh stop         # para tudo
#   ./qwen-serve.sh up-128k      # vLLM DFlash2 128K texto, sem menu
#   VISION=1 ./qwen-serve.sh up-128k   # idem, com visão
#
# Todo perfil sobe no MÁXIMO de contexto seguro anti-OOM (pool de KV pinado):
#   texto : 81920 (fast) / 163840 (long)   visão: 65536 (fast) / 131072 (long)
# Output por rodada: teto do servidor = contexto - prompt; o cliente (DSH)
# limita com max_tokens (16.384 no settings.yaml).
#
# VRAM medida (32 GB card):
#   vLLM DFlash2 64k bf16 texto      27.3 GiB (85%)
#   vLLM DFlash2 160k int8 texto     ~26.5 GiB (83%)
#   vLLM + visão (torre+offload)     +~1 GiB, KV reduzido p/ 6 GiB
#   llama.cpp plain 64k bf16         ~22 GiB (69%)
#   llama.cpp DFlash2 + 4 slots      ~25.2 GiB (79%)
#   llama.cpp 128k q8_0 (+/- mmproj) ~24-25 GiB (~77%)
set -euo pipefail
cd "$(dirname "$0")"

PORT_VLLM="${PORT_VLLM:-18020}"
PORT_LCPP="${PORT_LCPP:-18021}"
LCPP_DFLASH_DRAFT="${LCPP_DFLASH_DRAFT:-/app/gguf/Qwen3.8-27B-DFlash2-zlab-Q8_0.gguf}"

# Pools de KV pinados (bytes) — contexto máximo seguro por perfil:
KV_TEXT=7444609979        # 6.93 GiB -> 96.6k tokens bf16 / 192k int8
KV_VISION=6442450944      # 6.00 GiB -> 78k bf16 / 155k int8 (folga p/ torre visual)
MAX_FAST_TEXT=81920;   MAX_LONG_TEXT=163840
MAX_FAST_VISION=65536; MAX_LONG_VISION=131072

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

ask_vision() { # imprime 1 se o usuário quiser visão
  local a=""
  read -rp "  Com visão (mmproj/torre visual, contexto reduzido)? [s/N]: " a </dev/tty 2>/dev/null || a="${VISION:-0}"
  case "$a" in s|S|1) echo 1 ;; *) echo 0 ;; esac
}

apply_vllm_vision() { # apply_vllm_vision 0|1 fast|long
  local vis="$1" ctx="$2" max
  if [ "$vis" = 1 ]; then
    env_set VISION 1
    env_set KV_MEM "$KV_VISION"
    [ "$ctx" = long ] && max=$MAX_LONG_VISION || max=$MAX_FAST_VISION
  else
    env_set VISION 0
    env_set KV_MEM "$KV_TEXT"
    [ "$ctx" = long ] && max=$MAX_LONG_TEXT || max=$MAX_FAST_TEXT
  fi
  env_set MAX_LEN "$max"
  env_set DFLASH_MAX_LEN "$max"   # dflash2 lê DFLASH_MAX_LEN; MTP lê MAX_LEN
}

up_vllm() { # up_vllm fast|long 0|1
  local ctx="$1" vis="$2"
  stop_all
  if [ "$ctx" = long ]; then env_set CTX long; else env_del CTX; fi
  apply_vllm_vision "$vis" "$ctx"
  echo "[*] Subindo vLLM (CTX=$ctx, visão=$vis, MAX_LEN=$(grep '^MAX_LEN=' .env | cut -d= -f2))"
  echo "    primeiro boot frio pode levar ~15 min; com cache de compilação, ~2-4 min"
  docker compose -f docker-compose.amd.yml -f docker-compose.port.yml --profile single up -d --force-recreate
  wait_health "http://127.0.0.1:$PORT_VLLM/health" "vLLM ($ctx, visão=$vis)" 720 \
    && echo "    API: http://127.0.0.1:$PORT_VLLM/v1  (modelo servido: qwen3.8-27b)"
}

up_lcpp() { # up_lcpp none|mtp|dflash 0|1
  local spec="$1" vis="$2"
  stop_all
  echo "[*] Subindo llama.cpp (SPEC=$spec, visão=$vis, CTX=${LLCPP_CTX:-131072} ${LLCPP_KV_DTYPE:-q8_0})"
  if [ "$spec" = dflash ]; then
    SPEC="$spec" DRAFT="$LCPP_DFLASH_DRAFT" VISION="$vis" docker compose -f docker-compose.llamacpp.yml --profile single up -d --force-recreate
  else
    SPEC="$spec" VISION="$vis" docker compose -f docker-compose.llamacpp.yml --profile single up -d --force-recreate
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
    echo " 1) vLLM    DFlash2 T=4 · 64k      ~52 tok/s   27.3 GiB"
    echo " 2) vLLM    DFlash2 T=4 · 160k     ~44 tok/s   26.5 GiB  [perfil DSH]"
    echo " 3) vLLM    MTP · 96k              ~51 tok/s   ~27 GiB"
    echo " 4) llama.cpp plain · 128k         ~25 tok/s   ~24 GiB"
    echo " 5) llama.cpp MTP (nextn) · 128k   ~38 tok/s   ~25 GiB"
    echo " 6) llama.cpp DFlash2 · 128k       ~41 tok/s   ~25 GiB"
    echo " 7) status (containers / health / VRAM)"
    echo " 8) PARAR tudo"
    echo " 0) sair"
    echo "(todas as opções de serve perguntam: com ou sem visão)"
    read -rp "Escolha: " op
    local vis=0
    case "$op" in
      1) vis=$(ask_vision); env_set SPEC dflash2; env_set DFLASH_TOKENS 4; env_set PREFIX_CACHE 1; env_del CTX; up_vllm fast "$vis" ;;
      2) vis=$(ask_vision); env_set SPEC dflash2; env_set DFLASH_TOKENS 4; env_set PREFIX_CACHE 1; env_set CTX long; up_vllm long "$vis" ;;
      3) vis=$(ask_vision); env_set SPEC mtp; env_del DFLASH_TOKENS; env_del CTX; up_vllm fast "$vis" ;;
      4) vis=$(ask_vision); up_lcpp none "$vis" ;;
      5) vis=$(ask_vision); up_lcpp mtp "$vis" ;;
      6) vis=$(ask_vision); up_lcpp dflash "$vis" ;;
      7) status ;;
      8) stop_all; echo "[OK] parado." ;;
      0) exit 0 ;;
      *) echo "opção inválida" ;;
    esac
    echo
    read -rp "<Enter> volta ao menu... " _ </dev/tty 2>/dev/null || true
  done
}

VIS_DEFAULT=${VISION:-0}
case "${1:-}" in
  status) status ;;
  stop) stop_all ;;
  up-64k)  env_set SPEC dflash2; env_set DFLASH_TOKENS 4; env_set PREFIX_CACHE 1; env_del CTX; up_vllm fast "$VIS_DEFAULT" ;;
  up-128k) env_set SPEC dflash2; env_set DFLASH_TOKENS 4; env_set PREFIX_CACHE 1; env_set CTX long; up_vllm long "$VIS_DEFAULT" ;;
  up-mtp)  env_set SPEC mtp; env_del DFLASH_TOKENS; env_del CTX; up_vllm fast "$VIS_DEFAULT" ;;
  up-lcpp-plain) up_lcpp none "$VIS_DEFAULT" ;;
  up-lcpp-mtp) up_lcpp mtp "$VIS_DEFAULT" ;;
  up-lcpp-dflash) up_lcpp dflash "$VIS_DEFAULT" ;;
  *) menu ;;
esac
