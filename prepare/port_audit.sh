#!/usr/bin/env bash
# Build-order patch drift audit.
#   port_audit.sh [-m] /path/to/vllm-checkout [patches/upstream/NAME.diff]
#
# Default mode simulates EXACTLY what docker/Dockerfile.rocm does in the git
# variant: pre-apply the optional upstream diff at the repo root, then apply
# every non-skipped patches/*.patch IN GLOB ORDER to the same cumulative copy
# (package root). This is the honest question: with stacking, which patches
# still fail?  Patches that only modify files created by earlier patches are
# correctly reported as APPLIED here (they look broken against a clean tree).
#
# -m additionally classifies failures by retrying each with
# 'patch --merge=diff3', leaving a repairable workspace under
# /tmp/port-audit-last/<patch-name>/vllm/... (conflict markers inline) plus a
# per-patch log. Exit status = number of FAILED patches.
set -u
MERGE=0
if [ "${1:-}" = -m ]; then MERGE=1; shift; fi
TREE=${1:?usage: port_audit.sh [-m] /path/to/vllm-checkout [upstream.diff]}
UP=${2:-none}
SRC=$(cd "$(dirname "$0")"/.. && pwd)
BASE=$(mktemp -d /tmp/port-audit.XXXXXX)
rm -rf /tmp/port-audit-last && mkdir -p /tmp/port-audit-last
VP=$BASE/tree/vllm
mkdir "$BASE/tree"
cp -r "$TREE/." "$BASE/tree/"
if [ "$UP" != none ]; then
  echo "== upstream diff $UP (repo root)"
  if ! patch -p1 -s -d "$BASE/tree" < "$SRC/$UP"; then
    echo "upstream diff FAILED"; rm -rf "$BASE"; exit 125
  fi
fi
fail=0
cp_skipped=0
for p in "$SRC"/patches/*.patch; do
  name=$(basename "$p")
  if grep -Fxq "$name" "$SRC/patches/port-skip.lst" 2>/dev/null; then
    printf 'SKIPPED %s\n' "$name"; cp_skipped=$((cp_skipped+1)); continue
  fi
  log=/tmp/port-audit-last/$name.log
  if patch -p1 -s --no-backup-if-mismatch -d "$VP" < "$p" >"$log" 2>&1; then
    printf 'APPLIED %s\n' "$name"
    rm -f "$log"
  else
    fail=$((fail+1))
    first=$(head -1 "$log")
    printf 'FAILED  %s   (%s)\n' "$name" "$first"
    if [ "$MERGE" = 1 ]; then
      W=/tmp/port-audit-last/$name/vllm
      grep '^--- a/' "$p" | sed 's|^--- a/||' | while read -r f; do
        mkdir -p "$W/$(dirname "$f")"; cp "$VP/$f" "$W/$f" 2>/dev/null || true
      done
      cd "$W" 2>/dev/null && patch -p1 --merge=diff3 < "$p" \
        >/tmp/port-audit-last/$name.merge.log 2>&1
    fi
  fi
done
echo "--"
echo "failed: $fail | skipped(known-dropped): $cp_skipped"
[ "$MERGE" = 0 ] && echo "tip: rerun with -m for repairable conflict workspaces"
rm -rf "$BASE"
exit "$fail"
