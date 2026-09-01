"""Phase 6: hide a cell line, predict it from the rest, score on a certified scale.

This is what the first five phases were for. A predictor sees ONLY the held-out
line's non-targeting controls plus the list of perturbations to predict -- never
that line's perturbed cells -- emits 400 cells per perturbation, and is scored
against the line's own anchors, so 0 still means "the context-mean baseline" and
1 still means "a split-half replicate".

⚠️ Leave-one-LINE-out and leave-one-DATASET-out are THE SAME SPLIT for the
current corpus: H1 is the whole of vcc2025_h1 and K562 is the whole of
replogle2022. PLAN.md wants the gap between them as an optimism correction, and
that gap is not measurable until a third line shares a dataset with one of these.
Reported numbers are therefore the pessimistic (leave-one-dataset-out) kind --
which is the honest one, but do not quote a LOLO-vs-LODO delta we cannot compute.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from .predictors import apply_effect, effect_ratios
from .regime import NON_TARGETING, group_rng

CELLS_PER_PERT = 400


class Predictor(Protocol):
    """What a model must implement to be scored by the harness.

    `controls` is the held-out line's non-targeting cells, on the canonical axis.
    `perturbations` is the list to predict. Nothing else about the held-out line
    is provided -- that is the whole point.
    """

    name: str

    def predict(self, controls, perturbations: list[str], rng) -> "list[tuple[str, object]]":
        ...


@dataclass
class ContextMean:
    """The floor: predict every perturbation as the unperturbed control profile.

    Defines what 0 means. If the harness scores this materially above 0, the
    scale is wrong, not the predictor.
    """

    name: str = "context_mean"

    def predict(self, controls, perturbations, rng):
        n = controls.shape[0]
        out = []
        for p in perturbations:
            pick = rng.choice(n, size=min(CELLS_PER_PERT, n), replace=False)
            out.append((p, controls[np.sort(pick)].copy()))
        return out


@dataclass
class DeltaTransfer:
    """Apply another line's measured effect to this line's control cells.

    The floor predictor from PLAN.md, and exactly what the real submission did:
    take a perturbation's CPM-space effect as measured in a DIFFERENT cell line
    and scale this line's controls by it. Its failure mode is the interesting
    part -- see the `fid` discussion in README.
    """

    effects: pd.DataFrame          # perturbation x gene, CPM-space delta
    control_cpm: np.ndarray        # source line's control profile, CPM
    name: str = "delta_transfer"

    def predict(self, controls, perturbations, rng):
        n = controls.shape[0]
        out = []
        for p in perturbations:
            pick = np.sort(rng.choice(n, size=min(CELLS_PER_PERT, n), replace=False))
            block = controls[pick]
            if p in self.effects.index:
                pert_cpm = self.control_cpm + self.effects.loc[p].to_numpy(dtype=np.float64)
                ratios = effect_ratios(np.clip(pert_cpm, 0, None), self.control_cpm)
                block = apply_effect(block, ratios.astype(np.float32), rng)
            out.append((p, block))
        return out


@dataclass
class TransportWeightedTransfer:
    """Idea 5 (vcc2026-architecture-ideas.pdf): DeltaTransfer, but the donor's
    effect for gene g is shrunk toward "no change" by how much donor and
    target agree in g's own functional neighborhood, not by a fixed global
    clip alone.

    `neighbor_idx`: {gene -> array of axis indices}. An empty dict forces
    every perturbation onto the whole-axis fallback -- this is what makes
    `transport_transfer_global` (the A/B baseline) and `transport_transfer`
    (the gene-specific arm) the same code path in phase6_harness.py.
    """

    effects: pd.DataFrame          # perturbation x gene, CPM-space delta (donor)
    control_cpm: np.ndarray        # TARGET line's own control profile, CPM
    donor_control_cpm: np.ndarray  # DONOR line's own control profile, CPM
    neighbor_idx: dict
    name: str = "transport_transfer"

    def predict(self, controls, perturbations, rng):
        from .neighborhood import neighbourhood_confidence

        n = controls.shape[0]
        out = []
        for p in perturbations:
            pick = np.sort(rng.choice(n, size=min(CELLS_PER_PERT, n), replace=False))
            block = controls[pick]
            if p in self.effects.index:
                pert_cpm = self.control_cpm + self.effects.loc[p].to_numpy(dtype=np.float64)
                raw_ratios = effect_ratios(np.clip(pert_cpm, 0, None), self.control_cpm)
                conf = neighbourhood_confidence(self.donor_control_cpm, self.control_cpm,
                                                 self.neighbor_idx.get(p))
                ratios = 1.0 + conf * (raw_ratios - 1.0)
                block = apply_effect(block, ratios.astype(np.float32), rng)
            out.append((p, block))
        return out


def assemble(predictions, axis_symbols) -> "object":
    """Stack a predictor's per-perturbation blocks into one scoreable AnnData."""
    import anndata as ad
    from scipy import sparse

    blocks, labels = [], []
    for name, blk in predictions:
        blocks.append(blk.tocsr() if hasattr(blk, "tocsr") else blk)
        labels += [name] * blk.shape[0]
    X = sparse.vstack(blocks).tocsr()
    X.eliminate_zeros()
    return ad.AnnData(
        X=X.astype(np.float32),
        obs=pd.DataFrame({"perturbation": labels}),
        var=pd.DataFrame(index=pd.Index(list(axis_symbols))),
    )


def leakage_check(held_out_line: str, effects_source_line: str) -> None:
    """Refuse to score a predictor that was handed the answer.

    CLAUDE.md: any feature must be built from sources EXCLUDING the held-out
    line. Transferring a line's own effects onto itself would score near the
    ceiling and mean nothing.
    """
    if held_out_line == effects_source_line:
        raise ValueError(
            f"LEAKAGE: predicting {held_out_line} from {effects_source_line}'s own "
            f"effects. The held-out line's measurements must not reach the predictor."
        )
