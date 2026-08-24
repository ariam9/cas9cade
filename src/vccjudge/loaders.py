"""Phase 1: land every source dataset in one inspectable raw schema.

PLAN.md Phase 1 asks for `data/raw/<name>/<cell_line>.h5ad` in a fixed `.obs`
schema, and exposes `load_raw(dataset, cell_line) -> AnnData` (backed).

**Deviation, on purpose:** this module normalizes the schema *on read* rather
than rewriting each source matrix to a new file. Landing H1 alone would copy
14.4 GB to change nothing but four `.obs` columns -- `.X` is already raw counts
on the source axis, and Phase 2 (harmonize) and Phase 3 (render) both write new
files anyway. So the standardized thing is the *interface*, not a duplicated
matrix. Every downstream stage still sees one uniform schema, which is what the
phase is actually for; it just costs 0 bytes instead of 30+ GB across the
corpus. The per-source mapping that makes this work is declared in `SOURCES`
below, so it stays as inspectable as a landed file would be.

Everything here is backed/chunked: no stage may require a multi-GB matrix in
RAM, because the same code runs on a 24 GB laptop and on Aqua.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import sparse

#: The uniform `.obs` schema every raw source is presented in.
RAW_OBS_COLUMNS = ("perturbation", "cell_line", "dataset", "modality", "chemistry")

#: The canonical control label. Sources spell this many ways (see `ntc_labels`);
#: they are all normalized to this one so no stage has to know the dialect.
NON_TARGETING = "non-targeting"

#: Perturbation-column names seen across public perturb-seq datasets, in
#: priority order. Autodetection is a convenience for adding a source; every
#: registered source should pin `pert_col` explicitly once its schema is known.
_PERT_COL_CANDIDATES = (
    "target_gene", "gene", "gene_target", "perturbation", "guide_target",
    "gene_symbol", "knockdown", "condition",
)


@dataclass(frozen=True)
class RawSource:
    """One (dataset, cell_line) source file and how to read it uniformly.

    `modality` and `chemistry` are carried, never inferred: CLAUDE.md forbids
    silently pooling CRISPRi with CRISPRa/KO/chemical, and the 10x Flex vs
    non-Flex distinction is the gap Phase 3's renderer exists to measure.
    """

    dataset: str
    cell_line: str
    path: str
    modality: str
    chemistry: str
    pert_col: str | None = None
    ntc_labels: tuple[str, ...] = ("non-targeting", "NTC", "control", "non_targeting")
    notes: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (self.dataset, self.cell_line)


#: The registry. Provenance detail lives in config/datasets.yaml; this is the
#: machine-readable half that the loader actually dispatches on.
SOURCES: dict[tuple[str, str], RawSource] = {}


def register(src: RawSource) -> RawSource:
    SOURCES[src.key] = src
    return src


register(
    RawSource(
        dataset="vcc2025_h1",
        cell_line="H1",
        path="data/raw/vcc2025_h1/adata_Training.h5ad",
        modality="CRISPRi",
        chemistry="10x_flex",
        pert_col="target_gene",
        notes=(
            "VCC 2025 training split: 150 perturbations, only 13 of which are on the "
            "2026 panel. Matched chemistry + modality, but at ~54k median UMI and ~1k "
            "cells/pert it is far off the challenge regime -- Phase 3 must render it."
        ),
    )
)


def resolve_pert_col(obs_columns, pinned: str | None = None) -> str:
    """Find the perturbation column, preferring the source's pinned name."""
    cols = list(obs_columns)
    if pinned:
        if pinned not in cols:
            raise KeyError(f"pinned pert_col {pinned!r} not in .obs (has {cols})")
        return pinned
    for c in _PERT_COL_CANDIDATES:
        if c in cols:
            return c
    raise KeyError(f"no perturbation column found in .obs (has {cols})")


def load_raw(dataset: str, cell_line: str, backed: str | None = "r", root: str | Path = "."):
    """Open a registered source with `RAW_OBS_COLUMNS` populated.

    Backed by default -- a 14 GB source must not land in RAM just to be counted.
    Pass `backed=None` only when the caller genuinely needs the matrix.
    """
    import anndata as ad

    try:
        src = SOURCES[(dataset, cell_line)]
    except KeyError:
        raise KeyError(
            f"({dataset}, {cell_line}) is not registered; known: {sorted(SOURCES)}"
        ) from None

    path = Path(root) / src.path
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- download it before loading {src.key}")

    adata = ad.read_h5ad(path, backed=backed)
    pert_col = resolve_pert_col(adata.obs.columns, src.pert_col)

    perturbation = adata.obs[pert_col].astype(str)
    lowered = {l.lower() for l in src.ntc_labels}
    is_ntc = perturbation.str.lower().isin(lowered)
    adata.obs["perturbation"] = perturbation.where(~is_ntc, NON_TARGETING).values
    adata.obs["cell_line"] = src.cell_line
    adata.obs["dataset"] = src.dataset
    adata.obs["modality"] = src.modality
    adata.obs["chemistry"] = src.chemistry
    adata.uns["vccjudge_source"] = {
        "dataset": src.dataset, "cell_line": src.cell_line,
        "modality": src.modality, "chemistry": src.chemistry,
        "source_pert_col": pert_col, "path": str(src.path),
    }
    return adata


def _chunks(adata, chunk_size: int = 20_000):
    """Yield row blocks of `.X`, working in backed or in-memory mode alike."""
    n = adata.n_obs
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        X = adata.X[start:stop] if adata.isbacked else adata[start:stop].X
        yield X


def profile_raw(adata, chunk_size: int = 20_000) -> dict:
    """Per-file stats + a raw-counts verdict, computed in one chunked pass.

    PLAN.md wants n_cells / n_genes / median UMI / n perturbations / n NTC, and
    an assertion that the matrix is raw counts with a non-empty control group.
    Median UMI is exact (all per-cell totals are kept; they are 8 bytes each),
    while integrality/negativity are checked on every chunk as it goes by.
    """
    totals = np.empty(adata.n_obs, dtype=np.float64)
    detected = np.empty(adata.n_obs, dtype=np.int64)
    non_integer = negative = False
    at = 0
    for X in _chunks(adata, chunk_size):
        if sparse.issparse(X):
            Xc = X.tocsr()
            t = np.asarray(Xc.sum(axis=1)).ravel()
            d = np.diff(Xc.indptr)
            data = Xc.data
        else:
            A = np.asarray(X)
            t = A.sum(axis=1).ravel()
            d = (A != 0).sum(axis=1).ravel()
            data = A.ravel()
        if data.size:
            if not non_integer and not np.allclose(data, np.round(data)):
                non_integer = True
            if not negative and float(data.min()) < 0:
                negative = True
        totals[at : at + t.size] = t
        detected[at : at + d.size] = d
        at += t.size

    perts = adata.obs["perturbation"].astype(str)
    n_ntc = int((perts == NON_TARGETING).sum())
    targets = sorted(set(perts.unique()) - {NON_TARGETING})
    return {
        "dataset": adata.uns["vccjudge_source"]["dataset"],
        "cell_line": adata.uns["vccjudge_source"]["cell_line"],
        "modality": adata.uns["vccjudge_source"]["modality"],
        "chemistry": adata.uns["vccjudge_source"]["chemistry"],
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "median_umi": float(np.median(totals)),
        "mean_umi": float(totals.mean()),
        "median_genes_per_cell": float(np.median(detected)),
        "n_perturbations": len(targets),
        "n_ntc_cells": n_ntc,
        "median_cells_per_pert": float(
            np.median(perts[perts != NON_TARGETING].value_counts().values)
        ) if targets else 0.0,
        "raw_counts": (not non_integer) and (not negative),
        "non_integer": non_integer,
        "negative": negative,
        "has_controls": n_ntc > 0,
    }
