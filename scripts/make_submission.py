#!/usr/bin/env python
"""Build a contract-valid submission by transferring Replogle K562 effects.

    python scripts/make_submission.py -o submission.h5ad

For each (context, perturbation): draw 400 of that context's own control cells
and scale them by the perturbation's CPM-space effect measured in Replogle
K562. Perturbations with no Replogle measurement get ratio 1 everywhere, i.e.
the control profile -- an honest "no information" prediction rather than a
fabricated one.

This is PLAN.md's floor predictor: enough to prove the pipeline emits something
Arc accepts, and to put one real number on the board to calibrate Phase 5
against. It is NOT a model.
"""
from __future__ import annotations
import argparse, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
from scipy import sparse
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vccjudge.gene_axis import load_gene_axis
from vccjudge.predictors import apply_effect, effect_ratios

PAT = re.compile(r'^(\d+)_(.+?)_(?:P1P2|P1|P2|ENST[\d.,ENST]*)_(ENSG\d+)$')

def build_ratio_table(bulk_path, axis, verbose=True):
    """symbol -> per-axis-gene ratio vector, from Replogle K562 pseudobulk."""
    a = ad.read_h5ad(bulk_path)
    X = np.asarray(a.X.todense() if sparse.issparse(a.X) else a.X, dtype=np.float64)
    idx = a.obs.index.astype(str).tolist()
    gene_names = a.var["gene_name"].astype(str).values
    is_ctrl = np.array(["non-targeting" in s.lower() for s in idx])
    ctrl = X[is_ctrl].mean(axis=0)
    # Replogle gene -> axis column
    pos = {s: i for i, s in enumerate(axis["symbol"])}
    col = np.array([pos.get(g, -1) for g in gene_names])
    keep = col >= 0
    if verbose:
        print(f"  controls: {is_ctrl.sum()} rows | genes mapped to axis: {keep.sum():,}/{len(col):,}")
    out = {}
    for i, s in enumerate(idx):
        m = PAT.match(s)
        if not m or is_ctrl[i]:
            continue
        r = effect_ratios(X[i], ctrl)
        full = np.ones(len(axis), dtype=np.float64)
        full[col[keep]] = r[keep]
        out[m.group(2)] = full.astype(np.float32)
    return out

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="data/submission/submission.h5ad")
    ap.add_argument("--axis", default="artifacts/gene_axis.parquet")
    ap.add_argument("--perts", default="data/bundle/pert_counts.csv")
    ap.add_argument("--bulk", default="data/raw/replogle2022/K562_gwps_raw_bulk_01.h5ad")
    ap.add_argument("--controls", default="data/bundle")
    ap.add_argument("--contexts", default="A,B,C")
    ap.add_argument("--cells-per-pert", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-perts", type=int, default=None, help="smoke test with N perturbations")
    ap.add_argument("--compress", default=None, choices=[None, "gzip", "lzf"],
                    help="h5ad compression. Uncompressed is ~8.3 B/nonzero (17 GB here); gzip lands ~4 GB, "
                         "which is what fits Kaggle's 20 GB working dir. Does NOT reduce prep's RAM need.")
    a = ap.parse_args()

    axis = load_gene_axis(a.axis)
    perts = pd.read_csv(a.perts).iloc[:, 0].astype(str).str.strip().tolist()
    if a.limit_perts:
        perts = perts[: a.limit_perts]
    contexts = a.contexts.split(",")
    print(f"axis {len(axis):,} genes | {len(perts)} perturbations | contexts {contexts}")

    print("building Replogle K562 effect table ...")
    ratios = build_ratio_table(a.bulk, axis)
    have = [p for p in perts if p in ratios]
    print(f"  effects available for {len(have)}/{len(perts)} panel perturbations "
          f"({len(perts)-len(have)} fall back to control profile)")

    rng = np.random.default_rng(a.seed)
    # Stream the write: 360k cells x ~5.7k stored values is ~2.05e9 entries
    # (15 GB, 30 GB peak through vstack) — well past this machine's RAM. Blocks
    # are appended as they are made, so peak memory is one 400-cell block.
    import h5py
    from anndata.io import sparse_dataset, write_elem

    obs = pd.DataFrame({
        "target_gene": [p for ctx in contexts for p in perts for _ in range(a.cells_per_pert)],
        "context":     [ctx for ctx in contexts for _ in perts for _ in range(a.cells_per_pert)],
    })
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(a.out) + ".partial")
    t0 = time.time(); nnz = 0; tot_max = 0.0; umis = []
    with h5py.File(tmp, "w") as f:
        f.attrs["encoding-type"] = "anndata"; f.attrs["encoding-version"] = "0.1.0"
        write_elem(f, "obs", obs)
        write_elem(f, "var", pd.DataFrame(index=pd.Index(axis["symbol"].tolist())))
        dk = {"compression": a.compress} if a.compress else {}
        write_elem(f, "X", sparse.csr_matrix((0, len(axis)), dtype=np.float32),
                   dataset_kwargs=dk)
        Xds = sparse_dataset(f["X"])
        for ctx in contexts:
            ctrl = ad.read_h5ad(f"{a.controls}/context_{ctx}.h5ad")
            assert list(map(str, ctrl.var_names)) == axis["symbol"].tolist(), f"context_{ctx} off-axis"
            Xc = ctrl.X.tocsr(); n = Xc.shape[0]
            for j, p in enumerate(perts):
                pick = rng.choice(n, size=a.cells_per_pert, replace=False)
                block = apply_effect(Xc[pick], ratios.get(p, np.ones(len(axis), np.float32)), rng)
                Xds.append(block); nnz += block.nnz
                t = np.asarray(block.sum(axis=1)).ravel()
                umis.append(np.median(t)); tot_max = max(tot_max, float(t.max()))
                if (j + 1) % 50 == 0:
                    print(f"  {ctx}: {j+1}/{len(perts)} perts  nnz {nnz:,}  {time.time()-t0:5.0f}s", flush=True)
            del ctrl, Xc
    tmp.replace(a.out)
    print(f"\nwrote {a.out}: {len(obs):,} x {len(axis):,} | stored {nnz:,} "
          f"| median UMI {np.median(umis):,.0f} | max/cell {tot_max:,.0f} | {time.time()-t0:.0f}s")
    return 0

def _unused():
    pass

if __name__ == "__main__":
    raise SystemExit(main())
