#!/usr/bin/env bash
# Push a Kaggle kernel, wait for it, fetch logs and output.
#
#     bash scripts/kaggle/run_kernel.sh scripts/kaggle/fetch_sources [--no-wait]
#
# Kaggle is this project's "run-big" tier, replacing Aqua. It wins on the two
# axes that actually bottleneck us -- 32 GB RAM (vs 22 here) and datacentre
# bandwidth (vs ~2-4 MB/s) -- which are exactly what killed `vcc prep` locally
# and made an 8.2 GB download take an hour.
#
# The directory must contain kernel-metadata.json plus the script it names.
set -euo pipefail
DIR="${1:?usage: run_kernel.sh <kernel-dir> [--no-wait]}"
WAIT="${2:-}"
KAGGLE="${KAGGLE_BIN:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.venv/bin/kaggle}"

[ -f "$DIR/kernel-metadata.json" ] || { echo "FATAL: no kernel-metadata.json in $DIR"; exit 1; }
SLUG=$("$KAGGLE" kernels push -p "$DIR" 2>&1 | tee /dev/stderr | grep -oE '[a-z0-9-]+/[a-z0-9-]+' | tail -1)
[ -n "$SLUG" ] || { echo "FATAL: could not determine kernel slug from push output"; exit 1; }
echo "pushed: $SLUG"
[ "$WAIT" = "--no-wait" ] && exit 0

echo "polling (Kaggle sessions cap at 12h; this loop just watches)..."
while true; do
  S=$("$KAGGLE" kernels status "$SLUG" 2>&1 | tr -d '\r')
  echo "  $(date +%H:%M:%S) $S"
  case "$S" in
    *complete*|*COMPLETE*) break ;;
    *error*|*ERROR*|*cancel*|*CANCEL*)
      echo "=== FAILED — last log lines ==="
      "$KAGGLE" kernels logs "$SLUG" -p /tmp 2>/dev/null | tail -40 || true
      exit 1 ;;
  esac
  sleep 30
done
echo "=== output ==="
"$KAGGLE" kernels output "$SLUG" -p "${OUT_DIR:-./kaggle_out}"
