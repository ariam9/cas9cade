"""Phase 2: put every dataset on the canonical 18,533-gene axis, in order.

The join key is the **stable Ensembl gene ID**, never the symbol. CLAUDE.md is
categorical about this and the data agrees: H1's own `.var` disagrees with a
symbol-driven mapping on four genes, all of them symbols that name more than one
locus in the source annotation. Symbols are for display and for the final
`.var_names`; they are not identity.

Genes on the axis but absent from a source become **structural zeros** —
explicitly, and counted in the report. They are never imputed: a gene the assay
did not measure is not a gene that was measured at zero, and silently conflating
the two would put fabricated values into every downstream delta.

Memory: H1 alone is 1.93e9 stored values (14.4 GB). Nothing here materializes a
full matrix; the transform streams in row blocks and appends to the output file,
so it runs the same on a 24 GB laptop and on an Aqua node.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

DROP = -1  # sentinel in the column map: this source gene is not on the axis


@dataclass
class HarmonizeStats:
    """What the report needs, and what the halt rule is judged on."""

    n_source_genes: int
    n_axis_genes: int
    n_mapped: int              # axis genes that a source gene supplied
    n_source_dropped: int      # source genes with no axis home
    n_structural_zero: int     # axis genes with no source gene
    n_source_unresolved: int   # source genes with no usable ID at all
    nnz_in: int
    nnz_out: int
    n_cells: int

    @property
    def drop_rate(self) -> float:
        """Fraction of the AXIS left empty. This is the number PLAN.md halts on."""
        return self.n_structural_zero / self.n_axis_genes if self.n_axis_genes else 0.0

    @property
    def source_drop_rate(self) -> float:
        return self.n_source_dropped / self.n_source_genes if self.n_source_genes else 0.0


def source_gene_ids(adata, id_col_candidates=("gene_id", "gene_ids", "ensembl_id")) -> np.ndarray:
    """Stable (unversioned) Ensembl IDs for a source's columns, or '' where unknown.

    Prefers an explicit `.var` column -- that is the reference's own answer for
    what each column counts -- and falls back to `.var_names` when those are
    themselves Ensembl IDs. A source with neither cannot be joined safely and is
    rejected rather than symbol-matched behind your back.
    """
    for c in id_col_candidates:
        if c in adata.var.columns:
            ids = adata.var[c].astype(str)
            break
    else:
        vn = np.asarray(list(map(str, adata.var_names)))
        if np.char.startswith(vn, "ENSG").mean() > 0.9:
            ids = vn
        else:
            raise KeyError(
                f".var has no Ensembl ID column (looked for {id_col_candidates}) and "
                f".var_names are not Ensembl IDs. Refusing to join on symbols — "
                f"resolve IDs first (see scripts/build_ensembl_map.py)."
            )
    return np.array([s.split(".", 1)[0] if s and s != "nan" else "" for s in np.asarray(ids)])


def build_column_map(src_ids: np.ndarray, axis_ids: list[str]) -> tuple[np.ndarray, HarmonizeStats]:
    """source column index -> axis column index (or DROP).

    A duplicated source ID would make the mapping many-to-one and silently sum
    or overwrite columns, so the first occurrence wins and the rest are dropped
    and counted -- an explicit, visible loss rather than a quiet corruption.
    """
    axis_pos = {e: i for i, e in enumerate(axis_ids)}
    col_map = np.full(len(src_ids), DROP, dtype=np.int64)
    claimed: set[int] = set()
    unresolved = 0
    for j, e in enumerate(src_ids):
        if not e:
            unresolved += 1
            continue
        a = axis_pos.get(e)
        if a is None or a in claimed:
            continue
        col_map[j] = a
        claimed.add(a)
    stats = HarmonizeStats(
        n_source_genes=len(src_ids),
        n_axis_genes=len(axis_ids),
        n_mapped=len(claimed),
        n_source_dropped=int((col_map == DROP).sum()),
        n_structural_zero=len(axis_ids) - len(claimed),
        n_source_unresolved=unresolved,
        nnz_in=0, nnz_out=0, n_cells=0,
    )
    return col_map, stats


def remap_chunk(X, col_map: np.ndarray, n_axis: int):
    """Reindex one CSR row-block's columns onto the axis, dropping unmapped ones.

    Row boundaries are preserved exactly, so cells never move. `sort_indices` is
    required, not cosmetic: the axis order is a permutation of the source order,
    so remapped indices come out unsorted and a CSR with unsorted indices is
    silently wrong for many scipy ops.
    """
    X = X.tocsr()
    keep = col_map[X.indices] != DROP
    new_indices = col_map[X.indices]
    if keep.all():
        out = sparse.csr_matrix(
            (X.data, new_indices, X.indptr), shape=(X.shape[0], n_axis)
        )
    else:
        # Recompute indptr from per-row survivor counts.
        counts = np.add.reduceat(keep.astype(np.int64), X.indptr[:-1]) if X.nnz else np.zeros(X.shape[0], np.int64)
        counts[np.diff(X.indptr) == 0] = 0
        indptr = np.zeros(X.shape[0] + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        out = sparse.csr_matrix(
            (X.data[keep], new_indices[keep], indptr), shape=(X.shape[0], n_axis)
        )
    out.sort_indices()
    return out
