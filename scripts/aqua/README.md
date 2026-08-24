# Running this repo on Aqua (PBS Pro)

Aqua's compute nodes have **no outbound internet**. That single fact sets the
shape of everything here: anything that downloads — `pip install`, dataset
fetches — happens interactively on a **login node**, and the submitted jobs only
ever touch files that are already on disk. A `pip install` inside a `.cmd` fails
*after* the scheduler has spent your queue time, which is the expensive way to
learn this.

## Order of operations

```bash
# --- login node ---
bash scripts/aqua/00_probe_login.sh 2>&1 | tee aqua_probe.txt   # read-only survey
export VCC_DATA_DIR=/path/to/your/scratch/vcc-data              # from the probe
bash scripts/aqua/01_setup_env_login.sh                         # builds .venv-aqua
bash scripts/aqua/02_fetch_login.sh                             # downloads sources

# --- scheduler ---
qsub -q small8 scripts/aqua/phase1_report.cmd
qstat -u "$USER"
```

The probe is read-only and submits nothing. It exists because one thing still
cannot be determined from outside the cluster: **your scratch path**. The python
module is now pinned (`anaconda3_2024.10` — see below).

## RHEL 7 is the constraint that shapes everything

Aqua is **RHEL 7.6 → glibc 2.17**, and the login shell has only Python 2.6.6
with no conda. Two consequences:

- **`module load anaconda3_2024.10`** (newest available, ships Python 3.12,
  matching the repo's pin). Already the default in `01_setup_env_login.sh`.
- **`requirements.lock.txt` cannot be used here.** Every compiled pin in it
  needs glibc 2.24–2.28. Aqua installs from **`requirements-aqua.txt`**, whose
  ceilings are the newest releases still shipping `manylinux2014` wheels —
  numpy 2.2.6, scipy 1.16.3, pyarrow 20.0.0, h5py 3.14.0, pandas 2.3.2.
  `anndata` is byte-identical to the laptop's. The install uses
  `--only-binary=:all:` so a missing wheel fails immediately instead of
  attempting an hour-long source build that would fail anyway.

`01_setup_env_login.sh` writes `requirements-aqua.lock.txt` (a `pip freeze`)
after installing, because a glibc constraint cannot be expressed in a
requirements file and so the Aqua lock can only be captured on Aqua.

## Storage: where things must live

| | Quota | Lifetime | Put here |
|---|---|---|---|
| `$HOME` | **50 GB** | permanent, backed up | this repo, `artifacts/`, job scripts, logs |
| scratch | **500 GB** | **purged after ~1 week of no access**, no backup | raw `.h5ad` sources, intermediates |

Set `VCC_DATA_DIR` (default `${SCRATCH:-$HOME/scratch}/vcc-data`); `02_fetch_login.sh`
and the job scripts both honour it. Writing a 14 GB source into `$HOME` blows the
quota, and Aqua policy then **blocks job submission** until you clean it up.

This is also why the architecture works: PLAN.md touches raw cells exactly once
(Phases 1–4) and reduces them to a few MB of derived artifacts. The raw data is
disposable and re-downloadable; the artifacts go to `$HOME` where they are
backed up. Nothing irreplaceable ever lives on scratch — and per Aqua policy,
"users are strictly advised against putting source code, scripts, libraries,
executables in /scratch", nor symlinking scratch dirs into `$HOME`.

## Queues and the per-user caps

⚠️ **Aqua's own docs contradict themselves** — the summary table lists
`small20`/`small40`/`verylong`, the detail table does not, and the live `qmgr`
listing showed only per-PI queues (`workq` is `started = False`). So no job
script here hardcodes `-q`; pass the queue you are entitled to:

```bash
qsub -q small40 scripts/aqua/phase34_shard.cmd
```

| Queue | Cores | Walltime | Use for |
|---|---|---|---|
| `small8` | 1–8 | 2 h | Phase 1 report, smoke tests |
| `small40` | 1–40 | 24 h – 1 week (tables disagree) | **Phase 3–4 array shards** |
| `medium` / `long` | 21–40 | 48 h – 1 week | wide single jobs |
| `large` | 41–520 | 48 h | not needed by this plan |
| `gpuq` | — | 48 h | Phase 7 only |

**Per-user caps drive the array size** (small and medium groups):

| | limit |
|---|---|
| running jobs | 10 |
| queued jobs | 10 |
| cores | **80** |

`phase34_shard.cmd` therefore defaults to `-J 0-9` × `ncpus=8` = **exactly 80
cores**. Widening the array does not go faster: it queues, and may be rejected
against the 10-queued cap. To do more work, give each shard more perturbations
(`NSHARDS`), not more subjobs. Each compute node has 40 cores / 192 GB, so
`ncpus=8:mem=64gb` packs several shards per node.

PBS Pro **rejects** a walltime above the queue's ceiling rather than clamping
it, so change `-q` and `-l walltime` together.

## Login-node etiquette

Aqua kills login-node processes after **5 minutes of CPU time**. The two
login-node scripts stay under it: `curl` on a large download is I/O-bound, and
`pip install --only-binary` unpacks wheels rather than compiling. If a long
fetch is killed anyway, just re-run it — every transfer resumes. Policy also
asks for no more than two or three concurrent transfers, so `02_fetch_login.sh`
downloads strictly serially.

## PBS Pro, not SLURM

`PLAN.md` says "SLURM array jobs". Aqua is PBS Pro. The translation:

| SLURM | PBS Pro |
|---|---|
| `sbatch` | `qsub` |
| `--array=0-63` | `-J 0-63` |
| `$SLURM_ARRAY_TASK_ID` | `$PBS_ARRAY_INDEX` |
| `%a` in `-o` | `^array_index^` |
| `-c 8 --mem=64G` | `-l select=1:ncpus=8:mem=64gb` |
| `$SLURM_SUBMIT_DIR` | `$PBS_O_WORKDIR` |

## Files

| File | Where it runs | What it does |
|---|---|---|
| `00_probe_login.sh` | login | Read-only survey: PBS version, queues, modules, python, internet, scratch. Changes nothing. |
| `01_setup_env_login.sh` | login | Builds `.venv-aqua` from **`requirements-aqua.txt`** (not the laptop lock) + editable install. **Needs internet.** |
| `02_fetch_login.sh` | login | Downloads `config/fetch_manifest.tsv` into `$VCC_DATA_DIR` on scratch. Resumable, checksum-verified, serial. |
| `phase1_report.cmd` | `qsub` | Phase 1 acceptance — profiles landed sources, writes `data/data_report.md`. |
| `phase34_shard.cmd` | `qsub` | Phase 3–4 array job, one shard per `(line, perturbation)` group. |

## Conventions the job scripts rely on

- **`PYTHON_MODULE`** — defaults to `anaconda3_2024.10`; override only if the
  module list changes.
- **`VENV_DIR`** — defaults to `$PWD/.venv-aqua`, i.e. in `$HOME` alongside the
  repo. Keep it there: it must be visible to compute nodes (node-local `/tmp` is
  not), and it must survive scratch's weekly purge.
- **`VCC_DATA_DIR`** — raw data on scratch. Default
  `${SCRATCH:-$HOME/scratch}/vcc-data`.
- **`logs/`** — created by each job before PBS writes into it. PBS Pro fails the
  job outright if the `-o` directory does not exist.
- **Thread pinning** — array shards export `OMP_NUM_THREADS=$NCPUS` so BLAS does
  not oversubscribe the cores PBS granted. Without it, 10 shards each spawning
  40 threads on a 40-core node will thrash.

## Resumability

`CLAUDE.md` requires heavy stages to survive a killed batch job. That lives in
the stage, not the wrapper: each shard checkpoints and skips completed work, so
re-submitting the same array after a walltime kill only redoes the unfinished
shards. Re-running a finished array is therefore cheap and safe.

## Not yet written

`phase34_shard.cmd` calls `scripts/phase34_precompute.py`, which is Phase 3–4
work and does not exist yet. The `.cmd` is the submission contract that script
will be written against: shard index in, checkpointed artifacts out,
deterministic under `--seed`. Submitting it today will fail on the missing
script, by design — the wrapper is here so the interface is fixed before the
implementation.
