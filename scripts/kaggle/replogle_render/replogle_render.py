"""Fetch Replogle K562, subset to what the judge needs, harmonize, render.

Runs on Kaggle because the 8.2 GB download takes minutes there and never
finished on the dev connection, and because the source is a DENSE
1,989,578 x 8,248 float32 matrix -- 61 GB decompressed -- which has to be
streamed rather than loaded.

Output is small on purpose: only the 395 perturbations the judge actually needs
(136 shared with H1 = the leave-one-line-out test set, plus 272 on the 2026
panel) and 18,400 controls. That is ~4% of the screen, so the result comes back
over the wire instead of living on Kaggle forever.

⚠️ Replogle is a DEPTH AND CELL DEFICIT, the opposite of H1: ~166 cells/pert at
~11.4k UMI against the challenge's 400 at ~20k. Nothing can fix that -- you
cannot invent cells or reads -- so most groups render short and shallow, and are
flagged. Their DE will be lower-powered than the competition's, which is a real
property of this reference and must not be papered over.
"""
import glob, json, os, subprocess, sys, time
from pathlib import Path

WORK = Path("/kaggle/working")
# Scratch: /kaggle/working is capped at ~20 GB, far too small for an 8.2 GB
# download plus a 61 GB decompressed read. Pick the first location that actually
# EXISTS and has room -- probe_ram.py reported /kaggle/temp absent and /tmp with
# ~1 TB free, and hardcoding /kaggle/temp anyway cost a run.
def _scratch(need_gb=30):
    import shutil
    for c in ("/kaggle/temp", "/tmp", "/kaggle/working"):
        d = Path(c)
        try:
            d.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(d).free / 2**30 >= need_gb:
                return d
        except Exception:
            continue
    sys.exit("FATAL: no scratch dir with >=30 GB free")
REPO = "https://github.com/ariam9/cas9cade"
URL = ("https://zenodo.org/api/records/13350497/files/"
       "ReplogleWeissman2022_K562_gwps.h5ad/content")
MD5 = "13db594f8f1d2ccb88fec44a13e414dc"
NTC_CELLS = 18_400
TARGET_UMI = 20_000


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


t0 = time.time()
# Clone FIRST, then install the repo's own requirements.txt. Our pyproject
# deliberately declares dependencies = [] so requirements.txt stays the single
# source of truth -- which means `pip install git+<repo>` installs NO deps and
# the kernel dies on `import anndata`. Installing from requirements.txt keeps
# that single source of truth instead of duplicating a package list here.
sh(f"git clone -q {REPO} {WORK}/repo")
sh(f"pip -q install -r {WORK}/repo/requirements.txt")
sh(f"pip -q install -e {WORK}/repo --no-deps")
sys.path.insert(0, f"{WORK}/repo/src")

import numpy as np, pandas as pd, h5py, anndata as ad
from scipy import sparse

TMP = _scratch()
print(f"scratch: {TMP} ({__import__('shutil').disk_usage(TMP).free/2**30:,.0f} GB free)", flush=True)

# ---- inputs, verified before any expensive work ---------------------------
hits = sorted(glob.glob("/kaggle/input/**/gene_names.csv", recursive=True))
if not hits:
    sys.exit("FATAL: gene_names.csv not found under /kaggle/input")
IN = Path(hits[0]).parent
keep_perts = set(json.loads(Path(f"{WORK}/repo/config/replogle_keep_perts.json").read_text()))
emap = pd.read_csv(f"{WORK}/repo/artifacts/ensembl_map.csv")
axis_sym = pd.read_csv(IN / "gene_names.csv").iloc[:, 0].astype(str).str.strip().tolist()
sym2ens = dict(zip(emap.symbol, emap.ensembl_id))
axis_ens = [sym2ens.get(s) for s in axis_sym]
if any(e is None for e in axis_ens):
    sys.exit("FATAL: ensembl_map.csv does not cover the axis")
print(f"axis {len(axis_sym):,} genes | want {len(keep_perts)} perturbations", flush=True)

# ---- fetch ----------------------------------------------------------------
src = TMP / "replogle_k562_gwps.h5ad"
if not src.exists():
    t = time.time()
    sh(f"curl -fL --retry 5 -o {src} '{URL}'")
    mb = src.stat().st_size / 2**20
    print(f"  {mb:,.0f} MB in {time.time()-t:.0f}s ({mb/max(time.time()-t,1):.0f} MB/s)")
    import hashlib
    h = hashlib.md5()
    with open(src, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    if h.hexdigest() != MD5:
        sys.exit(f"FATAL: md5 {h.hexdigest()} != {MD5}")
    print("  md5 OK", flush=True)

# ---- inspect, and assert every assumption ---------------------------------
a = ad.read_h5ad(src, backed="r")
print(f"source: {a.shape} | obs {list(a.obs.columns)[:8]}", flush=True)
for need in ("perturbation",):
    if need not in a.obs.columns:
        sys.exit(f"FATAL: obs has no '{need}' column")
if "ensembl_id" not in a.var.columns:
    sys.exit(f"FATAL: var has no ensembl_id (has {list(a.var.columns)})")

obs_p = a.obs["perturbation"].astype(str).to_numpy()
vc = pd.Series(obs_p).value_counts()
ctrl_label = next((c for c in ("control", "non-targeting", "NTC", "non_targeting")
                   if c in vc.index), None)
if ctrl_label is None:
    sys.exit(f"FATAL: no control label found; top labels: {vc.head(8).to_dict()}")
print(f"control label: '{ctrl_label}' ({vc[ctrl_label]:,} cells)", flush=True)

# ---- choose rows (sorted -> one sequential pass) --------------------------
sys.path.insert(0, f"{WORK}/repo/src")
from vccjudge.regime import group_rng, thin_counts, thinning_probability

sel, meta = [], []
rng = group_rng("replogle2022", "K562", "__ntc__", 0)
ntc_idx = np.flatnonzero(obs_p == ctrl_label)
ntc_pick = np.sort(rng.choice(ntc_idx, min(NTC_CELLS, ntc_idx.size), replace=False))
sel.append(ntc_pick); meta.append(("non-targeting", ntc_pick.size, ntc_idx.size))
for p in sorted(keep_perts):
    idx = np.flatnonzero(obs_p == p)
    if idx.size == 0:
        continue
    if idx.size > 400:
        idx = np.sort(group_rng("replogle2022", "K562", p, 0).choice(idx, 400, replace=False))
    sel.append(idx); meta.append((p, idx.size, idx.size))
rows = np.sort(np.concatenate(sel))
n_short = sum(1 for _, got, _ in meta if got < 400 and _ != "non-targeting")
print(f"selected {rows.size:,} cells across {len(meta)} groups "
      f"({100*rows.size/a.n_obs:.1f}% of source); {n_short} groups short of 400", flush=True)

# ---- column map: source ensembl -> axis position --------------------------
src_ens = a.var["ensembl_id"].astype(str).str.split(".").str[0].to_numpy()
pos = {e: i for i, e in enumerate(axis_ens)}
col = np.array([pos.get(e, -1) for e in src_ens])
print(f"genes mapping onto the axis: {(col>=0).sum():,}/{len(col):,} "
      f"-> {len(axis_sym)-(col>=0).sum():,} axis genes become structural zeros", flush=True)

# ---- one sequential pass: extract -> harmonize -> thin --------------------
X = a.file["X"] if a.isbacked else None
is_dense = isinstance(X, h5py.Dataset)
print(f"source X is {'DENSE' if is_dense else 'sparse'}", flush=True)

want = np.zeros(a.n_obs, bool); want[rows] = True
CH = 4096
blocks, kept_umi = [], []
for start in range(0, a.n_obs, CH):
    stop = min(start + CH, a.n_obs)
    m = want[start:stop]
    if not m.any():
        continue
    blk = X[start:stop] if is_dense else a.X[start:stop]
    blk = np.asarray(blk)[m] if is_dense else blk[m]
    sp = sparse.csr_matrix(blk) if not sparse.issparse(blk) else blk.tocsr()
    keep = col[sp.indices] >= 0
    ni = col[sp.indices]
    if keep.all():
        out = sparse.csr_matrix((sp.data, ni, sp.indptr), shape=(sp.shape[0], len(axis_sym)))
    else:
        cnt = np.add.reduceat(keep.astype(np.int64), sp.indptr[:-1]) if sp.nnz else np.zeros(sp.shape[0], np.int64)
        cnt[np.diff(sp.indptr) == 0] = 0
        ip = np.zeros(sp.shape[0] + 1, np.int64); np.cumsum(cnt, out=ip[1:])
        out = sparse.csr_matrix((sp.data[keep], ni[keep], ip), shape=(sp.shape[0], len(axis_sym)))
    out.sort_indices()
    blocks.append(out.astype(np.float32))
    kept_umi.append(np.asarray(out.sum(1)).ravel())
    if (start // CH) % 50 == 0:
        print(f"  scanned {stop:,}/{a.n_obs:,}  kept {sum(b.shape[0] for b in blocks):,}  "
              f"{time.time()-t0:5.0f}s", flush=True)
a.file.close()

M = sparse.vstack(blocks).tocsr(); del blocks
med = float(np.median(np.concatenate(kept_umi)))
p, deficit = thinning_probability(med, TARGET_UMI)
print(f"depth: median {med:,.0f} UMI -> target {TARGET_UMI:,} | p={p:.4f} "
      f"{'⚠ DEFICIT — cannot thin upward, rendered as-is' if deficit else ''}", flush=True)
if not deficit:
    M = thin_counts(M, p, np.random.default_rng(0))

obs = pd.DataFrame({
    "perturbation": np.where(obs_p[rows] == ctrl_label, "non-targeting", obs_p[rows]),
    "cell_line": "K562", "dataset": "replogle2022", "source_row": rows,
})
out_ad = ad.AnnData(X=M, obs=obs, var=pd.DataFrame(index=pd.Index(axis_sym)))
# The rendered matrix stays on SCRATCH, not /kaggle/working. A 546 MB kernel
# output did not survive Kaggle's output mechanism (it came back as 889 bytes,
# and `kernels output` then stalled at 0 bytes), and hauling it over the wire
# was never the plan anyway: PLAN.md reduces raw cells to small artifacts on the
# big tier and ships only those. So Phase 4 runs HERE and only parquets go home.
dest = TMP / "replogle_K562__seed0.h5ad"
out_ad.write_h5ad(dest, compression="gzip")
achieved = float(np.median(np.asarray(M.sum(1)).ravel()))
json.dump({"n_cells": int(M.shape[0]), "n_groups": len(meta), "n_short": n_short,
           "median_umi_source": med, "median_umi_achieved": achieved,
           "thin_p": p, "depth_deficit": deficit,
           "genes_mapped": int((col >= 0).sum()), "nnz": int(M.nnz)},
          open(WORK / "replogle_K562__seed0.json", "w"), indent=2)
print(f"\nrendered -> {dest} ({dest.stat().st_size/2**20:,.0f} MB, on scratch)")
print(f"  {M.shape[0]:,} cells | nnz {M.nnz:,} | median UMI {achieved:,.0f}", flush=True)
del M, out_ad

# ---- Phase 4 here: reduce to the small artifacts the judge consumes -------
# K562 is a DEFICIT, so the render neither thinned (p=1.0) nor subsampled all
# but 10 groups -- the rendered file IS effectively full depth for these
# perturbations, so one file legitimately serves both the Role A (delta) and
# Role B (DE) sources. That is NOT true for H1, where they must differ.
sh(f"python {WORK}/repo/scripts/phase4_precompute.py "
   f"--dataset replogle2022 --cell-line K562 "
   f"--harmonized {dest} --rendered {dest} "
   f"--out {WORK}/reference --shard-size 20")
sh(f"rm -rf {WORK}/repo {WORK}/reference/de_shards__replogle2022__K562")
print(f"\nDONE in {time.time()-t0:.0f}s")
for f in sorted((WORK / "reference").glob("*")):
    print(f"  {f.stat().st_size/2**20:8,.1f} MB  {f.name}")
