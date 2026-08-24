#!/usr/bin/env bash
# Read-only survey of the Aqua environment. Run this ON THE LOGIN NODE first:
#
#     bash scripts/aqua/00_probe_login.sh 2>&1 | tee aqua_probe.txt
#
# It changes nothing. Send the output back and the remaining scripts get pinned
# to real module names and paths instead of placeholders -- the two things this
# repo cannot guess from outside the cluster.
set -uo pipefail

hr() { printf '\n=== %s ===\n' "$1"; }

hr "host / os"
hostname; uname -a; cat /etc/os-release 2>/dev/null | head -3

hr "PBS Pro"
qstat --version 2>&1 | head -2
command -v qsub pbsnodes qmgr 2>&1
echo "-- queues (name, max cores, walltime) --"
qmgr -c "list queue @default" 2>/dev/null | grep -E 'Queue |resources_max|resources_default|enabled|started' | head -60 \
  || qstat -Q 2>&1 | head -20

hr "array job support"
qmgr -c "list server" 2>/dev/null | grep -iE 'max_array|array' | head -5 || echo "(could not query server)"

hr "module system"
command -v module >/dev/null 2>&1 && echo "modules: present" || echo "modules: ABSENT"
echo "-- python modules --"
( module avail 2>&1 | tr ' ' '\n' | grep -iE '^python|^anaconda|^miniconda|^conda|^py3' | sort -u | head -30 ) 2>/dev/null
echo "-- hdf5 / compiler modules (may matter for h5py) --"
( module avail 2>&1 | tr ' ' '\n' | grep -iE '^hdf5|^gcc' | sort -u | head -15 ) 2>/dev/null

hr "python already on PATH"
for p in python3 python conda mamba uv; do
  command -v "$p" >/dev/null 2>&1 && echo "$p -> $($p --version 2>&1 | head -1)" || echo "$p -> absent"
done

hr "outbound internet FROM LOGIN NODE"
# Fetch REAL objects, not bucket roots: https://storage.googleapis.com with no
# object returns HTTP 400 even when connectivity is perfect, which reads as a
# failure and is not one.
curl -sI --max-time 15 https://pypi.org/simple/ -o /dev/null -w "  pypi:    HTTP %{http_code}\n" || echo "  pypi: FAILED"
curl -sI --max-time 20 "https://storage.googleapis.com/arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/gene_names.csv" \
  -o /dev/null -w "  gcs:     HTTP %{http_code}  (200 = Arc bucket reachable)\n" || echo "  gcs: FAILED"
curl -sI --max-time 15 https://ftp.ebi.ac.uk -o /dev/null -w "  ebi:     HTTP %{http_code}\n" || echo "  ebi: FAILED"
echo "  (proxy vars: ${http_proxy:-unset} / ${https_proxy:-unset})"

hr "scratch candidates (HOME has a 50 GB quota — data must NOT live there)"
echo "SCRATCH=${SCRATCH:-unset}  USER=$USER"
for base in /lfs1 /lfs /scratch; do
  [ -d "$base" ] || continue
  echo "-- $base --"
  ls -1d "$base"/*scratch* "$base"/*/*scratch* 2>/dev/null | head -5
  for cand in "$base/usrscratch/$USER" "$base/scratch/$USER" "$base/$USER"; do
    [ -d "$cand" ] && printf "   WRITABLE? %-40s %s\n" "$cand" "$([ -w "$cand" ] && echo yes || echo no)"
  done
done

hr "filesystems and quota"
echo "HOME=$HOME"
for d in "$HOME" /scratch "/scratch/$USER" /lfs /lfs1 "${SCRATCH:-}"; do
  [ -n "$d" ] && [ -d "$d" ] && printf "  %-24s %s\n" "$d" "$(df -h "$d" 2>/dev/null | awk 'NR==2{print $2" total, "$4" avail"}')"
done
command -v lfs >/dev/null 2>&1 && lfs quota -h "$HOME" 2>/dev/null | head -5
quota -s 2>/dev/null | head -5

hr "cpu / memory of this (login) node"
nproc; free -h 2>/dev/null | head -2

echo
echo "=== done. send this output back ==="
