#!/usr/bin/env bash
# Run a command under a hard memory ceiling.
#
#     bash scripts/capped.sh [--mem 14G] -- <command...>
#
# Why this exists: two stages in this repo touch multi-GB matrices, and an
# unbounded one does not fail politely — the kernel OOM killer picks a victim,
# which has twice been the whole session rather than the offending process.
# A cgroup ceiling turns that into a clean MemoryError in the child only.
#
# Default leaves ~8 GB of a 22 GB machine free, which is what "keep a few GB
# free" costs in practice. Raise with --mem only if you know the headroom is there.
set -euo pipefail
MEM="14G"
while [ $# -gt 0 ]; do
  case "$1" in
    --mem) MEM="$2"; shift 2 ;;
    --)    shift; break ;;
    *)     break ;;
  esac
done
[ $# -gt 0 ] || { echo "usage: capped.sh [--mem 14G] -- <command...>" >&2; exit 2; }
echo "[capped] MemoryMax=$MEM  free now: $(free -g | awk 'NR==2{print $7"G"}')" >&2
systemd-run --user --scope --quiet \
  -p MemoryMax="$MEM" -p MemorySwapMax=0 \
  -- "$@"
