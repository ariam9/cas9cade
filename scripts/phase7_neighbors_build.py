#!/usr/bin/env python
"""Idea 5 build: co-expression neighbors for the 300-gene panel, one line.

    bash scripts/capped.sh --mem 14G -- .venv/bin/python \
        scripts/phase7_neighbors_build.py --dataset vcc2025_h1 --cell-line H1

Writes the donor-side artifact `TransportWeightedTransfer` needs beyond
`control_cpm` (already written by phase4_precompute.py): for each of the
300 panel genes, its top-k |correlation| neighbors computed over THIS line's
own control cells -- the network-free "functional neighborhood" the locked
scope decision calls for (see neighborhood.py's module docstring).
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vccjudge.neighborhood import coexpression_neighbors  # noqa: E402
from vccjudge.regime import NON_TARGETING  # noqa: E402


def main() -> int:
    import anndata as ad

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="vcc2025_h1")
    ap.add_argument("--cell-line", default="H1")
    ap.add_argument("--rendered", default=None)
    ap.add_argument("--panel", default="data/bundle/pert_counts.csv")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--out", default="artifacts/reference")
    a = ap.parse_args()

    ren = Path(a.rendered or f"data/regime/{a.dataset}/{a.cell_line}__seed0.h5ad")
    if not ren.exists():
        sys.exit(f"FATAL: {ren} not found — run scripts/phase3_render.py first")
    panel = pd.read_csv(a.panel)["target_gene"].astype(str).tolist()

    adata = ad.read_h5ad(ren)
    obs = adata.obs["perturbation"].astype(str).to_numpy()
    ctrl = adata.X.tocsr()[np.flatnonzero(obs == NON_TARGETING)]
    axis = list(map(str, adata.var_names))
    print(f"{ren}: {ctrl.shape[0]:,} control cells, {len(panel)} panel genes")

    neighbors = coexpression_neighbors(ctrl, panel, axis, k=a.k)
    rows = [{"gene": g, "neighbor_gene": axis[j], "rank": r}
            for g, idxs in neighbors.items() for r, j in enumerate(idxs)]
    df = pd.DataFrame(rows)
    out_path = Path(a.out) / f"{a.dataset}_coexpr_neighbors__panel.parquet"
    df.to_parquet(out_path, index=False)
    print(f"wrote {out_path}: {df['gene'].nunique()} panel genes covered, up to {a.k} neighbors each")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
