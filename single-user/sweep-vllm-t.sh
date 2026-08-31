#!/usr/bin/env bash
# Sweep de DFlash2 T (num_speculative_tokens) no vLLM 0.28.0 ROCm 7.2:
# perfil visao + contexto longo (131072), mesmo protocolo 4x160 do llama.cpp.
# Uso: bash single-user/sweep-vllm-t.sh [2 3 4 5]
set -uo pipefail
cd "$(dirname "$0")/.."
TS=("$@"); [ ${#TS[@]} -eq 0 ] && TS=(2 3 4 5)

env_set() { k=$1; v=$2; if grep -q "^${k}=" .env; then sed -i "s|^${k}=.*|${k}=${v}|" .env; else echo "${k}=${v}" >> .env; fi; }

# Perfil visao + contexto longo (fixo para todo o sweep)
env_set SPEC dflash2
env_set PREFIX_CACHE 1
env_set CTX long
env_set VISION 1
env_set KV_MEM 6442450944
env_set MAX_LEN 131072
env_set DFLASH_MAX_LEN 131072
env_set LOOKUP 1
env_set VLLM_DFLASH2_CHAIN 0

for T in "${TS[@]}"; do
  echo "=== T=$T ==="
  env_set DFLASH_TOKENS "$T"
  docker compose -f docker-compose.amd.yml -f docker-compose.port.yml --profile single up -d --force-recreate > /dev/null 2>&1
  ok=0
  for i in $(seq 1 90); do
    code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:18020/health || true)
    [ "$code" = "200" ] && { ok=1; break; }
    sleep 3
  done
  [ "$ok" = 1 ] || { echo "[FAIL] T=$T nao subiu"; continue; }
  sleep 2
  python3 probe_gen.py 2>&1 | tail -5
  docker logs qwen38-27b-rtx3090-amd-single-1 --since 6m 2>&1 | grep -E 'Mean acceptance length' | tail -1 | sed 's/^/  [aceitacao] /'
  docker logs qwen38-27b-rtx3090-amd-single-1 --since 6m 2>&1 | grep -E 'GPU KV cache size' | tail -1 | sed 's/^/  [pool] /'
done

# Restaura o diario T=4
env_set DFLASH_TOKENS 4
docker compose -f docker-compose.amd.yml -f docker-compose.port.yml --profile single up -d --force-recreate > /dev/null 2>&1
echo SWEEP_VLLM_DONE
