#!/usr/bin/env bash
# Fetch every source in config/fetch_manifest.tsv. LOGIN NODE ONLY.
#
#     bash scripts/aqua/02_fetch_login.sh [manifest.tsv]
#
# Aqua's compute nodes have no outbound internet, so acquisition is a separate
# interactive step from processing. Downloads are resumable (`curl -C -`), so a
# dropped login session costs only the remaining bytes -- re-run the same
# command. Nothing here touches the scheduler.
set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MANIFEST="${1:-$PROJECT_DIR/config/fetch_manifest.tsv}"

# $HOME is 50 GB and permanent; scratch is 500 GB and purged after a week of no
# access. Raw source data is large and re-downloadable, so it belongs on
# scratch; code and the small derived artifacts stay in $HOME. Writing a 14 GB
# h5ad into $HOME would blow the quota and, per Aqua policy, block job
# submission until it is cleaned up.
DATA_DIR="${VCC_DATA_DIR:-${SCRATCH:-$HOME/scratch}/vcc-data}"

[ -f "$MANIFEST" ] || { echo "FATAL: no manifest at $MANIFEST"; exit 1; }
mkdir -p "$DATA_DIR" || { echo "FATAL: cannot create $DATA_DIR — set VCC_DATA_DIR to your scratch path"; exit 1; }
cd "$DATA_DIR"

echo "manifest: $MANIFEST"
echo "data dir: $DATA_DIR  ($(df -h "$DATA_DIR" 2>/dev/null | awk 'NR==2{print $4" free"}'))"
echo "NOTE: scratch is purged after ~1 week of no access. Re-run this script if"
echo "      a later job reports missing inputs; nothing here is irreplaceable."
echo
# Aqua kills login-node processes after 5 minutes of CPU time. curl on a large
# download is I/O-bound and uses very little CPU, so this is normally fine --
# but if a multi-hour fetch is killed, just re-run: every transfer resumes.
echo

fail=0; done_n=0; skipped=0
while IFS=$'\t' read -r dest url sha || [ -n "${dest:-}" ]; do
  case "${dest:-}" in ''|\#*) continue ;; esac
  if [ "$url" = "RESOLVE" ]; then
    echo "[SKIP] $dest — url still marked RESOLVE; fill the accession first"
    skipped=$((skipped+1)); continue
  fi

  mkdir -p "$(dirname "$dest")"
  # Checksums may be sha256 (bare hex) or md5 (prefixed "md5:") — Zenodo publishes
  # md5, Figshare sha256, and a manifest that could only express one would force
  # us to drop verification on whichever source lost.
  verify() {  # verify <file> <checksum-spec> -> 0 ok, 1 bad
    case "$2" in
      md5:*)  echo "${2#md5:}  $1" | md5sum -c --status 2>/dev/null ;;
      *)      echo "$2  $1" | sha256sum -c --status 2>/dev/null ;;
    esac
  }
  if [ -f "$dest" ] && [ "$sha" != "SKIP" ] && [ "$sha" != "RESOLVE" ]; then
    if verify "$dest" "$sha"; then
      echo "[ok]   $dest — already present, checksum matches"
      done_n=$((done_n+1)); continue
    fi
    echo "[warn] $dest — present but checksum differs; re-fetching"
  fi

  echo "[get]  $dest"
  if ! curl -fL --retry 5 --retry-delay 10 -C - --progress-bar -o "$dest" "$url"; then
    echo "[FAIL] $dest — download error"; fail=$((fail+1)); continue
  fi

  if [ "$sha" != "SKIP" ] && [ "$sha" != "RESOLVE" ]; then
    if verify "$dest" "$sha"; then
      echo "[ok]   $dest — checksum verified"
    else
      echo "[FAIL] $dest — CHECKSUM MISMATCH"
      fail=$((fail+1)); continue
    fi
  else
    echo "[ok]   $dest — $(sha256sum "$dest" | cut -d' ' -f1)  <- record this in the manifest"
  fi
  done_n=$((done_n+1))
done < "$MANIFEST"

echo
echo "fetched/verified: $done_n   unresolved: $skipped   failed: $fail"
[ "$fail" -eq 0 ] || { echo "FAIL — fix the above before submitting any job"; exit 1; }
[ "$skipped" -eq 0 ] || echo "NOTE — $skipped entries still need accessions resolved."
echo "OK — data staged. Submit work with: qsub scripts/aqua/phase1_report.cmd"
