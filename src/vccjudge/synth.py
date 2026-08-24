"""Synthetic-data helpers for testing the contract without real bundles."""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def make_random_submission(
    symbols, perts, contexts, cells_per_pert=400,
    detected_frac=(0.25, 0.35), max_umi=20_000, seed=0,
):
    """A contract-VALID random submission over the given axis.

    Emits sparse integer counts with realistic per-cell sparsity and library
    size, no non-targeting rows, exactly `cells_per_pert` cells per group.
    """
    rng = np.random.default_rng(seed)
    n_genes = len(symbols)
    rows, obs_pert, obs_ctx = [], [], []
    for ctx in contexts:
        for p in perts:
            for _ in range(cells_per_pert):
                k = rng.integers(int(detected_frac[0] * n_genes), int(detected_frac[1] * n_genes) + 1)
                idx = rng.choice(n_genes, size=k, replace=False)
                counts = rng.integers(1, 8, size=k)
                # scale down if over the library cap (won't happen here, but be safe)
                if counts.sum() > max_umi:
                    counts = np.maximum(1, (counts * max_umi / counts.sum()).astype(int))
                row = sparse.csr_matrix((counts, (np.zeros(k, int), idx)), shape=(1, n_genes))
                rows.append(row)
                obs_pert.append(p)
                obs_ctx.append(ctx)
    X = sparse.vstack(rows).tocsr()
    X.eliminate_zeros()
    obs = pd.DataFrame({"target_gene": obs_pert, "context": obs_ctx})
    var = pd.DataFrame(index=pd.Index(list(symbols), name=None))
    return ad.AnnData(X=X.astype(np.float32), obs=obs, var=var)
