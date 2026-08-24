"""Build AND prep the submission on Kaggle, where there is enough RAM.

`vcc prep` needs ~20-30 GB to hold a realistic 360,000-cell submission
(2.05e9 stored values = 15.3 GB as CSR, plus a copy while encoding). The dev
laptop has 22 GB and was OOM-killed at both 16 GB and 19 GB ceilings. Kaggle
gives 32 GB, so build->prep runs in one session and the multi-GB intermediate
never crosses a network.

Output is a .vcc of a few hundred MB, which IS worth pulling back. Submission
happens from the laptop, so no VCC token needs to exist here.
"""
import glob, os, subprocess, sys, time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = "https://github.com/ariam9/cas9cade"

def sh(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)

# Locate the mounted dataset rather than assuming its path. An earlier run
# built symlinks to a guessed path; os.symlink does NOT verify its target, so
# they dangled silently and surfaced as a confusing FileNotFoundError much
# later. Resolve it explicitly and fail loudly here if it is missing.
print("=== /kaggle/input tree ===", flush=True)
for root, dirs, files in os.walk("/kaggle/input"):
    if files:
        print(" ", root, sorted(files)[:9], flush=True)
# Search RECURSIVELY: Kaggle mounted this at /kaggle/input/datasets/<user>/<slug>/,
# not the /kaggle/input/<slug>/ a one-level glob assumed. Anchor on a file we
# know must be there rather than on any assumed layout.
hits = glob.glob("/kaggle/input/**/gene_names.csv", recursive=True)
if not hits:
    sys.exit("FATAL: gene_names.csv not found anywhere under /kaggle/input — check dataset_sources")
IN = Path(hits[0]).parent
print(f"using inputs from: {IN}", flush=True)

t0 = time.time()
sh(f"pip -q install 'git+{REPO}' vcc-cli")
os.chdir(WORK)
sh(f"git clone -q {REPO} repo")
(WORK / "artifacts").mkdir(exist_ok=True)
(WORK / "sub").mkdir(exist_ok=True)

# Axis: symbols + order only. ensembl_id is irrelevant to the submission (the
# emitter indexes by symbol), so no GENCODE download is needed here.
sh(f"python repo/scripts/build_gene_axis.py {IN}/gene_names.csv -o artifacts/gene_axis.parquet")

# gzip: uncompressed this is 17 GB and /kaggle/working caps at ~20 GB.
sh(f"python repo/scripts/make_submission.py -o sub/submission.h5ad --compress gzip "
   f"--axis artifacts/gene_axis.parquet --perts {IN}/pert_counts.csv "
   f"--bulk {IN}/K562_gwps_raw_bulk_01.h5ad --controls {IN}")

sh(f"python -m vccjudge.contract sub/submission.h5ad --axis artifacts/gene_axis.parquet "
   f"--perts {IN}/pert_counts.csv --contexts A,B,C")

sh(f"vcc prep sub/submission.h5ad -g {IN}/gene_names.csv --perts {IN}/pert_counts.csv "
   f"-o {WORK}/prediction.vcc")

# Keep only the .vcc as kernel output; the rest is large and disposable.
sh("rm -rf /kaggle/working/sub /kaggle/working/repo /kaggle/working/artifacts")
print(f"\nDONE in {time.time()-t0:.0f}s")
for p in sorted(WORK.iterdir()):
    print(f"  {p.stat().st_size/2**20:10,.1f} MB  {p.name}")
