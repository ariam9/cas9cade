"""Build AND prep the submission on Kaggle, where there is enough RAM.

`vcc prep` needs ~20-30 GB to hold a realistic 360,000-cell submission
(2.05e9 stored values = 15.3 GB as CSR, plus a copy while encoding). The dev
laptop has 22 GB and was OOM-killed at both 16 GB and 19 GB ceilings. Kaggle
gives 32 GB, so the whole build->prep path runs in one session and the 17 GB
intermediate never crosses a network.

Output is a .vcc of a few hundred MB, which IS worth pulling back — submission
happens from the laptop, so no VCC token is needed here.
"""
import os, subprocess, sys, time
from pathlib import Path

IN = Path("/kaggle/input/vcc2026-inputs")
WORK = Path("/kaggle/working")
REPO = "https://github.com/ariam9/cas9cade"

def sh(cmd, **kw):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kw)

t0 = time.time()
sh(f"pip -q install 'git+{REPO}' anndata scipy pandas pyarrow")
sh("pip -q install vcc-cli")

# Lay the repo's expected paths out under /kaggle/working.
(WORK / "data/bundle").mkdir(parents=True, exist_ok=True)
(WORK / "data/raw/replogle2022").mkdir(parents=True, exist_ok=True)
(WORK / "artifacts").mkdir(exist_ok=True)
for n in ("context_A.h5ad", "context_B.h5ad", "context_C.h5ad",
          "gene_names.csv", "pert_counts.csv", "manifest.json"):
    os.symlink(IN / n, WORK / "data/bundle" / n)
os.symlink(IN / "K562_gwps_raw_bulk_01.h5ad",
           WORK / "data/raw/replogle2022/K562_gwps_raw_bulk_01.h5ad")

os.chdir(WORK)
sh(f"git clone -q {REPO} repo")

# The axis: symbols + order only. ensembl_id is irrelevant to the submission
# (the emitter indexes by symbol), so no GENCODE download is needed here.
sh("python repo/scripts/build_gene_axis.py data/bundle/gene_names.csv "
   "-o artifacts/gene_axis.parquet")

# gzip the h5ad: uncompressed it is 17 GB and /kaggle/working caps at ~20 GB.
sh("python repo/scripts/make_submission.py -o data/submission/submission.h5ad "
   "--compress gzip")

sh("python -m vccjudge.contract data/submission/submission.h5ad "
   "--axis artifacts/gene_axis.parquet --perts data/bundle/pert_counts.csv "
   "--contexts A,B,C")

sh("vcc prep data/submission/submission.h5ad -g data/bundle/gene_names.csv "
   "--perts data/bundle/pert_counts.csv -o /kaggle/working/prediction.vcc")

# Keep only the .vcc as kernel output; everything else is huge and disposable.
for p in ("data", "repo", "artifacts"):
    sh(f"rm -rf /kaggle/working/{p}")
print(f"\nDONE in {time.time()-t0:.0f}s")
for p in sorted(WORK.iterdir()):
    print(f"  {p.stat().st_size/2**20:10,.1f} MB  {p.name}")
