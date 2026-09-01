"""Idea 5 (vcc2026-architecture-ideas.pdf) -- transportability, not similarity.

`DeltaTransfer` applies a donor line's measured effect to a target line's
controls with a single fixed global clip (`predictors.RATIO_CLIP`) and no
similarity weighting at all. The field's standard fix -- weight donor LINES by
global transcriptome similarity -- has nothing to weight between here: the
corpus has exactly one usable donor per held-out line (leave-one-line-out
always leaves one; see `harness.py`'s own note on this). So this module
realizes the idea as a per-gene CONFIDENCE multiplier on that single donor's
effect instead: trust the transferred effect for gene g in proportion to how
much donor and target agree in g's own functional neighborhood, not in their
overall transcriptomes.

No pathway/interaction network exists anywhere in this repo (a locked scope
decision -- see the plan this module implements). The "neighborhood" here is
therefore built from co-expression over control cells already on disk: genes
that move together in this line's own unperturbed population are the cheapest
network-free proxy for "acts through the same mechanism" available without a
new data source. A real network (STRING/GO) is a noted fast-follow only if
this fails its own kill criterion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


def control_pseudobulk_cpm(control_X) -> np.ndarray:
    """Pseudobulk CPM of a block of control cells, full axis."""
    X = control_X.tocsr() if hasattr(control_X, "tocsr") else control_X
    total = np.asarray(X.sum(axis=0)).ravel()
    lib = total.sum()
    return total * (1e6 / lib) if lib > 0 else total


def coexpression_neighbors(control_X, panel_genes, axis_symbols, k: int = 50) -> dict[str, np.ndarray]:
    """Top-k |Pearson correlation| neighbor gene INDICES (into axis_symbols) for
    each panel gene, computed over one line's own control cells in
    CPM/log1p space.

    Restricted to `panel_genes` on the query side so only a
    (len(panel_genes) x n_genes) correlation slab is ever built -- never the
    full 18,533x18,533 matrix.
    """
    X = control_X.tocsr() if hasattr(control_X, "tocsr") else control_X
    n_cells = X.shape[0]
    axis = pd.Index(axis_symbols)
    present = [g for g in panel_genes if g in axis]
    panel_idx = np.array([axis.get_loc(g) for g in present])

    lib = np.asarray(X.sum(axis=1)).ravel()
    lib[lib == 0] = 1.0
    scale = sparse.diags(1e6 / lib)
    full = np.log1p((scale @ X).toarray()).astype(np.float32)   # (n_cells, n_genes)
    panel = full[:, panel_idx]                                   # (n_cells, n_panel)

    full_c = full - full.mean(axis=0, keepdims=True)
    panel_c = panel - panel.mean(axis=0, keepdims=True)
    full_std = full_c.std(axis=0) + 1e-6
    panel_std = panel_c.std(axis=0) + 1e-6
    cov = (panel_c.T @ full_c) / n_cells                         # (n_panel, n_genes)
    corr = cov / (panel_std[:, None] * full_std[None, :])

    out: dict[str, np.ndarray] = {}
    for i, g in enumerate(present):
        row = np.abs(corr[i]).copy()
        row[panel_idx[i]] = -1.0    # exclude self
        top = np.argpartition(row, -k)[-k:]
        out[g] = top[np.argsort(-row[top])]
    return out


def neighbourhood_confidence(
    donor_control_cpm: np.ndarray, target_control_cpm: np.ndarray,
    neighbor_idx: np.ndarray | None, floor: float = 0.15,
) -> float:
    """Correlation of log1p(CPM) restricted to a gene's neighborhood, mapped
    from [-1, 1] to [floor, 1].

    Mean-centered (i.e. Pearson correlation), not raw cosine similarity: two
    log1p(CPM) vectors are all-positive and sit at a similar overall scale, so
    UNcentered cosine similarity is dominated by that shared magnitude and
    barely responds to real differences in composition (verified: a
    deliberately inverted, noised profile still cosine-scored >0.99 similar to
    itself). Centering removes the shared "DC" component so the score actually
    reflects whether donor and target agree on the SHAPE of expression in this
    neighborhood, matching the correlation `coexpression_neighbors` already
    uses to define the neighborhood in the first place.

    `neighbor_idx=None` (or empty) falls back to the WHOLE axis -- this single
    function serves both the gene-specific arm and the global-similarity A/B
    baseline, controlled only by which dict a caller passes in.
    """
    d = np.log1p(np.asarray(donor_control_cpm, dtype=np.float64))
    t = np.log1p(np.asarray(target_control_cpm, dtype=np.float64))
    if neighbor_idx is not None and len(neighbor_idx) > 0:
        d, t = d[neighbor_idx], t[neighbor_idx]
    d, t = d - d.mean(), t - t.mean()
    dn, tn = np.linalg.norm(d), np.linalg.norm(t)
    corr = float(np.dot(d, t) / (dn * tn)) if dn > 0 and tn > 0 else 0.0
    return floor + (1.0 - floor) * (corr + 1.0) / 2.0
