"""Idea 4 (vcc2026-architecture-ideas.pdf) -- test-aware emission: invert the scorer.

Four of six scored metrics are Wilcoxon tests run against the 400 cells a
predictor emits, not against a mean profile. `apply_effect` (predictors.py)
applies one deterministic per-gene ratio to every cell -- a near-zero-dispersion
decoder that reports a belief's DIRECTION and MAGNITUDE but throws away its
CONFIDENCE, so the scorer's own significance test sees an artificially tight
(or loose) sample regardless of how sure the belief actually was.

This module is a measurement-model component only: it claims no biology, and
composes with any belief -- whether that belief comes from a real DE table
(the kill-check this module exists to support) or a future biological model.
It never decides WHICH genes respond or BY HOW MUCH -- only how a per-gene
belief becomes 400 emitted cells. Keeping it architecturally separate from
biological components means a bad kill-check result implicates this module
alone, never the biology feeding it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .predictors import cap_library_size, stochastic_round


@dataclass
class Belief:
    """A per-gene belief for ONE perturbation, axis-aligned (same order and
    length as the control block's columns).

    `confidence` in [0, 1]: how sure the belief is. Untested and
    non-significant genes default to confidence=1.0 (confidently unchanged) --
    there is no reason to inject noise where nothing is claimed to move.
    """

    gene: np.ndarray
    log2fc: np.ndarray
    is_sig: np.ndarray
    confidence: np.ndarray

    def __post_init__(self):
        n = len(self.gene)
        for name in ("log2fc", "is_sig", "confidence"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"Belief.{name} must match Belief.gene in length")


def belief_from_de_table(
    de_df: pd.DataFrame, target: str, axis_symbols, alpha: float = 0.05, *,
    target_col: str = "target", feature_col: str = "feature",
    lfc_col: str = "log2_fold_change", padj_col: str = "p_adj",
) -> Belief:
    """Build a Belief from a reference_de__*.parquet-shaped table.

    Genes the DE table never tested (below cell-eval2's CPM floor, so absent
    from the table's rows for this target) get log2fc=0, is_sig=False,
    confidence=1.0 -- there was no expression to test in the first place.
    """
    axis = pd.Index(axis_symbols, dtype=object)
    sub = de_df.loc[de_df[target_col].astype(str) == str(target)].set_index(feature_col)
    lfc = sub[lfc_col].reindex(axis).fillna(0.0).to_numpy(dtype=np.float64)
    padj = sub[padj_col].reindex(axis).to_numpy(dtype=np.float64)
    is_sig = padj < alpha  # NaN < alpha is False: untested genes are non-significant
    conf = np.ones(len(axis), dtype=np.float64)
    conf[is_sig] = np.clip(1.0 - padj[is_sig] / alpha, 0.0, 1.0)
    return Belief(gene=axis.to_numpy(), log2fc=lfc, is_sig=is_sig, confidence=conf)


def emit_from_belief(
    control_block, belief: Belief, rng: np.random.Generator,
    dispersion_scale: float | np.ndarray = 0.3, max_counts: int = 1_000_000,
):
    """Sample 400 cells consistent with `belief`, starting from real control cells.

    Non-significant genes: unchanged (ratio 1.0, natural per-cell dispersion
    inherited from the real starting cells -- see predictors.py's design note
    on why we perturb real cells rather than invent new ones). Significant
    genes: the per-cell ratio is drawn around the belief's target fold change
    with log-normal spread controlled by `dispersion_scale`, widened where
    `confidence` is lower -- an unsure belief should produce a diffuse sample,
    not a confidently wrong tight one. This per-cell dispersion is the one
    mechanism `apply_effect` lacks: it applies exactly one ratio to every cell
    regardless of how sure the belief was.

    Cannot create expression in a cell where a gene's count is already zero
    (any ratio times zero is zero) -- the same limitation `apply_effect` has,
    for the same reason.
    """
    X = control_block.tocsr().astype(np.float64)
    n_genes = X.shape[1]
    if belief.gene.size != n_genes:
        raise ValueError("belief must be axis-aligned to control_block's columns")

    scale = np.broadcast_to(np.asarray(dispersion_scale, dtype=np.float64), (n_genes,))
    ratio_mean = np.where(belief.is_sig, np.exp2(belief.log2fc), 1.0)
    spread = np.clip(np.where(belief.is_sig, scale * (1.5 - belief.confidence), 0.0), 0.0, None)

    col = X.indices
    base_ratio = ratio_mean[col]
    jitter = np.ones_like(base_ratio)
    noisy = spread[col] > 0
    if noisy.any():
        sigma = spread[col][noisy]
        mu = -0.5 * sigma ** 2  # E[exp(N(mu, sigma^2))] == 1, so the mean ratio stays base_ratio
        jitter[noisy] = np.exp(rng.normal(mu, sigma))

    X.data *= base_ratio * jitter
    X.data = stochastic_round(X.data, rng)
    X.eliminate_zeros()
    return cap_library_size(X, max_counts).astype(np.float32)


def regenerate_from_real(
    de_table: pd.DataFrame, target: str, axis_symbols, control_block,
    rng: np.random.Generator, dispersion_scale: float = 0.3, alpha: float = 0.05,
):
    """The kill-test recipe: real DE table + real control cells -> 400 emitted cells.

    No perturbed cells are used -- this is exactly what a predictor would have
    if it perfectly knew the true significance/sign/magnitude/confidence but
    still had to synthesise cells from real controls, same as any real
    predictor does.
    """
    belief = belief_from_de_table(de_table, target, axis_symbols, alpha=alpha)
    return emit_from_belief(control_block, belief, rng, dispersion_scale=dispersion_scale)
