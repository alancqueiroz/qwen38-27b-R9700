#!/usr/bin/env bash
# probe_cycle.sh <DFLASH_TOKENS|-> <SPEC> : set knobs, recreate server, wait, report
set -u
T=$1; SPEC=${2:-dflash2}
cd /home/alan/projetos/qwen38-27b-rtx3090
# normalize .env: drop old knobs, set new
sed -i '/^DFLASH_TOKENS=/d;/^SPEC=/d' .env
printf 'SPEC=%s\n' "$SPEC" >> .env
[ "$T" != - ] && printf 'DFLASH_TOKENS=%s\n' "$T" >> .env
printf 'PREFIX_CACHE=1\n' >> .env
sort -u .env -o .env
docker compose -f docker-compose.amd.yml -f docker-compose.port.yml --profile single up -d >/dev/null 2>&1
code=000
for i in $(seq 1 40); do
  sleep 20
  code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:18020/health)
  [ "$code" = 200 ] && break
done
echo "T=$T SPEC=$SPEC health=$code after $((i*20))s"
