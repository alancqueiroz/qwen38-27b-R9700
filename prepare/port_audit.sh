#!/usr/bin/env bash
# Drift audit: would every patch in patches/ still apply to a given vLLM tree?
# Usage:
#   prepare/port_audit.sh /path/to/vllm-checkout [patches/upstream/NAME.diff]
# The optional second argument is applied to the COPY first, mirroring what
# docker/Dockerfile.rocm does with VLLM_UPSTREAM_PATCH in the git variant.
# Nothing outside /tmp is modified. Exit status = number of failing patches.
set -u
TREE=${1:?usage: port_audit.sh /path/to/vllm-checkout [upstream.diff]}
UP=${2:-none}
SRC=$(cd "$(dirname "$0")/.." && pwd)
TMP=$(mktemp -d /tmp/port-audit.XXXXXX)
mkdir "$TMP/tree"
cp -r "$TREE/." "$TMP/tree/"
# NOTE the two roots differ ON PURPOSE:
#   - patches/upstream/*.diff are repo-root relative (a/vllm/..., a/tests/...)
#     and apply at the checkout root -- same as VLLM_UPSTREAM_PATCH;
#   - patches/*.patch are PACKAGE-root relative (a/v1/...) because the image
#     applies them to site-packages/vllm, so they dry-run against tree/vllm.
if [ "$UP" != none ]; then
  echo "== pre-applying $UP (repo root)"
  if ! patch -p1 -s -d "$TMP/tree" < "$SRC/$UP"; then
    echo "upstream patch FAILED against this tree"; rm -rf "$TMP"; exit 125
  fi
fi
fail=0
for p in "$SRC"/patches/*.patch; do
  name=$(basename "$p")
  tag="    "
  if grep -Fxq "$name" "$SRC/patches/port-skip.lst" 2>/dev/null; then tag="skip"; fi
  if patch -p1 --dry-run -s -d "$TMP/tree/vllm" < "$p" >/dev/null 2>&1; then
    res=OK
  else
    res=FAIL; fail=$((fail+1))
  fi
  printf '%s %-4s %s\n' "$tag" "$res" "$name"
done
echo "--"
echo "failing: $fail of $(ls "$SRC"/patches/*.patch | wc -l)"
rm -rf "$TMP"
exit "$fail"
