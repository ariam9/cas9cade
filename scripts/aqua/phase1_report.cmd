#!/usr/bin/env bash
#PBS -N vcc_phase1_report
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=01:30:00
#PBS -j oe
#PBS -o logs/phase1_report.log
#
# Phase 1 acceptance on Aqua: profile every landed source, write data_report.md.
#
#     qsub -q <your_queue> scripts/aqua/phase1_report.cmd
#
# ⚠️ NO `#PBS -q` HERE, ON PURPOSE. Aqua's live queue list is per-PI
# (sumesh_q, parag_q, nccrd_q, ...); `workq` exists but is `started = False` and
# `routeq` is disabled, so there is no usable default. The small8/small20/medium
# names in Aqua's public docs did not appear in the live `qmgr` listing. Pass the
# queue you are actually entitled to on the command line -- a hardcoded guess
# just gets the job rejected at submit time.
#
# Walltime must fit the queue's ceiling: PBS Pro rejects an over-limit walltime
# rather than clamping it.
#
# Prerequisites, both LOGIN-NODE steps (compute nodes have no internet):
#     bash scripts/aqua/01_setup_env_login.sh
#     bash scripts/aqua/02_fetch_login.sh
set -euo pipefail

cd "${PBS_O_WORKDIR:-$PWD}"
mkdir -p logs

# Pin this to what 00_probe_login.sh reported.
[ -n "${PYTHON_MODULE:-}" ] && module load "$PYTHON_MODULE"
source "${VENV_DIR:-$PWD/.venv-aqua}/bin/activate"

echo "host:     $(hostname)"
echo "job:      ${PBS_JOBID:-interactive}"
echo "python:   $(python --version 2>&1)"
echo "workdir:  $PWD"
echo "started:  $(date -Is)"

python scripts/phase1_data_report.py --root . --out data/data_report.md
status=$?

echo "finished: $(date -Is)  exit=$status"
exit $status
