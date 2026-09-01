"""Predictors: control cells + a gene list in, 400 cells per perturbation out.

The floor predictor here is `replogle_transfer`: borrow the perturbation's
pseudobulk effect measured in K562 and apply it to the target context's own
control cells.

Design note -- why we start from REAL control cells rather than synthesising
counts from a mean profile. The contract wants raw integer counts, realistic
library sizes, no explicitly-stored zeros, and >=1 stored value per gene we
claim. Real control cells already satisfy all of that, and they carry the
context's own expression state, which is the only information the challenge
gives us about a blinded cell line. So we perturb real cells rather than invent
new ones: multiply each cell's counts by a per-gene ratio, then round
stochastically so integrality is preserved without biasing the mean.

Everything is depth-agnostic: the ratio is computed in CPM space, so Replogle's
~11k UMI pseudobulk transfers onto the challenge's ~20k UMI cells unchanged.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

#: Ratios outside this band come from genes with almost no signal in the
#: reference, where a CPM ratio is noise amplified by division. Clipping is a
#: statement that we do not believe a 50x effect estimated off ~1 count.
RATIO_CLIP = (0.05, 20.0)
#: CPM floor in the denominator; ~1 count at the challenge's depth.
CPM_EPS = 1.0


def cpm(x: np.ndarray) -> np.ndarray:
    tot = float(x.sum())
    return x * (1e6 / tot) if tot > 0 else x


def effect_ratios(pert_profile, ctrl_profile, clip=RATIO_CLIP, eps=CPM_EPS) -> np.ndarray:
    """Per-gene multiplicative effect, in CPM space, clipped."""
    p, c = cpm(np.asarray(pert_profile, float)), cpm(np.asarray(ctrl_profile, float))
    r = (p + eps) / (c + eps)
    return np.clip(r, *clip)


def stochastic_round(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Round to integers without biasing the mean: E[round(x)] == x."""
    fl = np.floor(x)
    return fl + (rng.random(x.shape) < (x - fl))


def cap_library_size(X, max_counts: int = 1_000_000):
    """Contract item 7: <=max_counts stored per cell.

    Shared by every emitter that scales real cells' counts -- never hit at
    challenge depth, but cheap to guarantee.
    """
    totals = np.asarray(X.sum(axis=1)).ravel()
    over = totals > max_counts
    if over.any():
        for i in np.flatnonzero(over):
            s, e = X.indptr[i], X.indptr[i + 1]
            X.data[s:e] = np.floor(X.data[s:e] * (max_counts / totals[i]))
        X.eliminate_zeros()
    return X


def apply_effect(X, ratios: np.ndarray, rng: np.random.Generator, max_counts: int = 1_000_000):
    """Scale a CSR block of control cells by a per-gene ratio, keep counts integral."""
    X = X.tocsr().astype(np.float64)
    X.data *= ratios[X.indices]
    X.data = stochastic_round(X.data, rng)
    X.eliminate_zeros()
    return cap_library_size(X, max_counts).astype(np.float32)
