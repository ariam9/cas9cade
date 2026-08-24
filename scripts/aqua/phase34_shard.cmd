#!/usr/bin/env bash
#PBS -N vcc_p34_shard
#PBS -l select=1:ncpus=8:mem=64gb
#PBS -l walltime=24:00:00
#PBS -J 0-9
#PBS -j oe
#PBS -o logs/p34_shard_^array_index^.log
#
# Phases 3-4 as a PBS Pro ARRAY JOB: render each (cell line, perturbation) into
# the challenge regime and precompute its delta + DE table.
#
#     qsub -q small40 -J 0-1 scripts/aqua/phase34_shard.cmd   # smoke test
#     qsub -q small40        scripts/aqua/phase34_shard.cmd   # all 10 shards
#
# ⚠️ `-J 0-9` x `ncpus=8` = 80 cores, which is EXACTLY Aqua's per-user cap for
# the small and medium groups (max 10 running jobs, max 10 queued, max 80
# cores). Widening the array or the cpu count does not go faster -- it queues,
# and may be rejected outright against the 10-queued limit. To do more work,
# give each shard more perturbations (NSHARDS below), not more subjobs.
#
# ⚠️ No `#PBS -q`: Aqua's published queue tables contradict each other (the
# summary lists small20/small40/verylong, the detail section does not) and the
# live `qmgr` listing showed only per-PI queues. Pass the queue you are actually
# entitled to. small40 is 1--40 cores and the longest small-group option; check
# its real walltime ceiling before raising -l walltime, since PBS Pro rejects an
# over-limit request rather than clamping it.
#
# ⚠️ PLAN.md says "SLURM array jobs"; Aqua is PBS Pro, so it is `-J 0-N` and
# $PBS_ARRAY_INDEX, not --array/$SLURM_ARRAY_TASK_ID. The ^array_index^ token in
# the -o path is PBS Pro's substitution and is what keeps per-shard logs apart.
#
# Resumability (CLAUDE.md: heavy stages must survive a killed batch job) is a
# property of the STAGE, not this wrapper: each shard writes a checkpoint and
# skips work already checkpointed, so re-submitting the same array after a
# walltime kill costs only the unfinished shards. --force overrides.
#
# If shards are the bottleneck, raise walltime (up to the queue's ceiling)
# rather than the shard count -- the 80-core cap means extra subjobs only queue.
set -euo pipefail

cd "${PBS_O_WORKDIR:-$PWD}"
mkdir -p logs data/regime artifacts/shards

[ -n "${PYTHON_MODULE:-}" ] && module load "$PYTHON_MODULE"
source "${VENV_DIR:-$PWD/.venv-aqua}/bin/activate"

SHARD="${PBS_ARRAY_INDEX:-0}"
NSHARDS="${NSHARDS:-10}"   # must match the -J range above
SEED="${SEED:-0}"

# Raw data lives on scratch (500 GB, purged after a week); code and the small
# derived artifacts live in $HOME (50 GB, backed up). See scripts/aqua/README.md.
DATA_DIR="${VCC_DATA_DIR:-${SCRATCH:-$HOME/scratch}/vcc-data}"
echo "data:     $DATA_DIR"
[ -d "$DATA_DIR" ] || echo "WARNING: $DATA_DIR missing — scratch may have been purged; re-run 02_fetch_login.sh"

echo "host:     $(hostname)"
echo "job:      ${PBS_JOBID:-interactive}  shard ${SHARD}/${NSHARDS}"
echo "python:   $(python --version 2>&1)"
echo "threads:  ${NCPUS:-8}"
echo "started:  $(date -Is)"

# Keep BLAS from oversubscribing the cores PBS actually granted.
export OMP_NUM_THREADS="${NCPUS:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

# NOTE: scripts/phase34_precompute.py is Phase 3-4 work and does not exist yet.
# This wrapper is the submission contract it will be written against -- shard
# index in, checkpointed artifacts out, deterministic under --seed.
python scripts/phase34_precompute.py \
    --shard "$SHARD" --n-shards "$NSHARDS" \
    --seed "$SEED" \
    --out artifacts/shards \
    --checkpoint-dir artifacts/shards/.ckpt
status=$?

echo "finished: $(date -Is)  exit=$status"
exit $status
