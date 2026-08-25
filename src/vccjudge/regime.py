"""Phase 3: render a harmonized line into the challenge's measurement regime.

`nreal` -- how many genes test significant -- is a function of cell count and
sequencing depth, not biology alone. Four of the six scored metrics are
DE-based, so a reference measured at a different regime calibrates them to a
world the leaderboard does not live in. That is what this closes.

The two corpora need opposite corrections, which is why the renderer must
handle both directions and say which one it applied:

    H1        1,045 cells/pert @ 53,912 UMI  -> SURPLUS (subsample + thin down)
    Replogle    166 cells/pert @ 11,431 UMI  -> DEFICIT (cannot be fixed)

You cannot invent cells or reads. A short group is rendered short and flagged;
`PLAN.md` is explicit that this is "a real limitation, not something to
fabricate around". Sampling with replacement would manufacture fake replicates
and inflate every DE statistic computed from them.

Determinism is per (dataset, cell_line, perturbation, seed) -- derived from a
stable hash, never from a shared global stream -- because Phase 3/4 shard per
group. A shard must produce identical output whether it runs alone, in a
resumed batch, or in a different order.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

CHALLENGE_CELLS_PER_PERT = 400
#: The challenge's control arm is 46x a perturbation group -- manifest.json says
#: ground_truth_cells 138,400 = 300 x 400 + 18,400. Rendering NTC down to 400
#: would make local DE far noisier than the competition's, miscalibrating nreal
#: in the opposite direction from the bug this module exists to fix.
CHALLENGE_NTC_CELLS = 18_400
#: Measured across all 55,200 control cells: 20,109 / 19,946 / 20,034 for A/B/C.
CHALLENGE_MEDIAN_UMI = 20_000
NON_TARGETING = "non-targeting"


def group_rng(dataset: str, cell_line: str, perturbation: str, seed: int) -> np.random.Generator:
    """A generator determined solely by its group identity and the seed."""
    key = f"{dataset}|{cell_line}|{perturbation}|{seed}".encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))


@dataclass
class GroupPlan:
    """What will happen to one (perturbation) group, decided before any I/O."""

    perturbation: str
    n_available: int
    n_target: int
    picked: np.ndarray                      # row indices into the source, sorted
    short: bool = False                     # fewer cells than the regime wants

    @property
    def n_picked(self) -> int:
        return int(self.picked.size)


@dataclass
class RenderPlan:
    """The whole render, decided up front so I/O is one sequential pass."""

    groups: list[GroupPlan] = field(default_factory=list)
    thin_p: float = 1.0                     # binomial keep-probability
    source_median_umi: float = 0.0
    depth_deficit: bool = False             # source shallower than the challenge

    @property
    def order(self) -> np.ndarray:
        """All picked rows, sorted -- random seeks are what make this slow."""
        return np.sort(np.concatenate([g.picked for g in self.groups]))

    def summary(self) -> dict:
        short = [g for g in self.groups if g.short]
        return {
            "n_groups": len(self.groups),
            "n_cells": sum(g.n_picked for g in self.groups),
            "n_short_groups": len(short),
            "shortest": min((g.n_picked for g in short), default=None),
            "thin_p": self.thin_p,
            "source_median_umi": self.source_median_umi,
            "depth_deficit": self.depth_deficit,
        }


def plan_render(
    perturbations,
    dataset: str,
    cell_line: str,
    seed: int,
    cells_per_pert: int = CHALLENGE_CELLS_PER_PERT,
    ntc_cells: int = CHALLENGE_NTC_CELLS,
) -> RenderPlan:
    """Choose which rows to keep. No matrix access, so this is cheap and testable."""
    perts = np.asarray(perturbations, dtype=object)
    plan = RenderPlan()
    for name in sorted(set(perts.tolist())):
        idx = np.flatnonzero(perts == name)
        target = ntc_cells if str(name).lower() == NON_TARGETING else cells_per_pert
        if idx.size <= target:
            picked, short = idx, idx.size < target
        else:
            rng = group_rng(dataset, cell_line, str(name), seed)
            picked, short = np.sort(rng.choice(idx, size=target, replace=False)), False
        plan.groups.append(GroupPlan(str(name), int(idx.size), target, picked, short))
    return plan


def thinning_probability(median_umi: float, target: float = CHALLENGE_MEDIAN_UMI) -> tuple[float, bool]:
    """One global keep-probability for the whole line, plus a deficit flag.

    Global rather than per-group on purpose: a per-group ratio would flatten the
    real depth differences between perturbations, which are part of what the
    metrics see. Scaling everything by one factor moves the median onto target
    and leaves the shape of the depth distribution intact.
    """
    if median_umi <= 0:
        return 1.0, True
    if median_umi < target:
        return 1.0, True          # cannot thin upward; render as-is and flag
    if median_umi == target:
        return 1.0, False         # already in regime: no thinning, and no deficit
    return float(target / median_umi), False


def thin_counts(X, p: float, rng: np.random.Generator):
    """Binomial thinning: each count c becomes Binomial(c, p).

    This is the exact generative model for shallower sequencing, not an
    approximation to it -- which is why the regime is simulated rather than
    corrected for analytically. Counts stay integral, so the result is still
    raw counts as the contract demands.
    """
    X = X.tocsr()
    if p >= 1.0:
        return X.astype(np.float32)
    out = X.astype(np.float64, copy=True)
    out.data = rng.binomial(out.data.astype(np.int64), p).astype(np.float64)
    out.eliminate_zeros()
    return out.astype(np.float32)
