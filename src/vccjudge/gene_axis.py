"""
Canonical gene axis for VCC 2026.

The challenge fixes a specific ordered set of genes (gene_names.csv in the
validation bundle, stated as 18,533 symbols in a required order). Every
artifact in the pipeline -- harmonized data, predictions, submissions -- must
live on THIS axis in THIS order. This module is the single source of truth for
that ordering. Building it wrong silently corrupts every downstream comparison,
so it is Phase 0 and nothing is allowed to depend on it until its acceptance
test passes.

Design notes
------------
* The axis is authoritatively defined by the *symbols* and their *order* in
  gene_names.csv. That is all the scorer needs to align columns.
* `ensembl_id` is an OPTIONAL enrichment column used later (Phase 2) for
  cross-dataset joins, because symbols are ambiguous and drift across
  annotations while Ensembl IDs are stable. At Phase 0 we do not require it:
  if you pass a symbol->Ensembl mapping we fill it, otherwise it is left null
  with a warning. Do the actual Ensembl resolution once, from a pinned
  annotation (e.g. a GENCODE/Ensembl release or a dataset's own var), and feed
  it in here so the mapping is recorded alongside the axis.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

# Arc states the 2026 axis is 18,533 genes. We do not hard-fail on this (tests
# use small synthetic axes), but we warn loudly if a "real" axis disagrees.
EXPECTED_N_GENES = 18_533

_HEADER_TOKENS = {
    "", "gene", "genes", "gene_name", "gene_names", "symbol", "gene_symbol",
    "var_names", "feature_name", "index",
}


def _read_symbol_column(gene_names_csv: str | Path) -> list[str]:
    """Read the ordered symbol list from gene_names.csv robustly.

    The bundle's exact formatting (header? column name?) is not guaranteed, so
    we read the first column with no header assumption and drop a leading row
    only if it looks like a header token rather than a gene symbol.
    """
    raw = pd.read_csv(gene_names_csv, header=None, dtype=str)
    if raw.shape[1] < 1:
        raise ValueError(f"{gene_names_csv} has no columns")
    col = raw.iloc[:, 0].astype(str).str.strip()
    if len(col) and col.iloc[0].lower() in _HEADER_TOKENS:
        col = col.iloc[1:]
    symbols = col.tolist()
    if not symbols:
        raise ValueError(f"{gene_names_csv} produced an empty symbol list")
    return symbols


def build_gene_axis(
    gene_names_csv: str | Path,
    out_parquet: str | Path,
    ensembl_map: dict[str, str] | str | Path | None = None,
) -> pd.DataFrame:
    """Build and persist the canonical gene axis.

    Parameters
    ----------
    gene_names_csv : path to the bundle's gene_names.csv (ordered symbols).
    out_parquet    : where to write the axis parquet.
    ensembl_map    : optional. Either a {symbol: ensembl_id} dict or a path to a
                     2-column CSV (symbol,ensembl_id). If omitted, ensembl_id is
                     left null (fine for Phase 0; required by Phase 2 joins).

    Returns
    -------
    DataFrame with columns [order_index, symbol, ensembl_id], row i having
    order_index == i, in the required order.
    """
    symbols = _read_symbol_column(gene_names_csv)

    dup = pd.Index(symbols)[pd.Index(symbols).duplicated()].unique().tolist()
    if dup:
        raise ValueError(
            f"gene_names.csv contains duplicate symbols (first few: {dup[:5]}). "
            "The axis must be unique; refusing to build a corrupt axis."
        )

    if len(symbols) != EXPECTED_N_GENES:
        warnings.warn(
            f"gene axis has {len(symbols)} genes, expected {EXPECTED_N_GENES}. "
            "OK for a synthetic/test axis; investigate if this is the real bundle.",
            stacklevel=2,
        )

    df = pd.DataFrame(
        {"order_index": range(len(symbols)), "symbol": symbols, "ensembl_id": pd.NA}
    )

    if ensembl_map is not None:
        if not isinstance(ensembl_map, dict):
            m = pd.read_csv(ensembl_map, dtype=str)
            m.columns = [c.strip().lower() for c in m.columns]
            ensembl_map = dict(zip(m["symbol"], m["ensembl_id"]))
        df["ensembl_id"] = df["symbol"].map(ensembl_map).astype("string")
        n_missing = int(df["ensembl_id"].isna().sum())
        if n_missing:
            warnings.warn(
                f"{n_missing}/{len(df)} symbols had no Ensembl ID in the mapping.",
                stacklevel=2,
            )

    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    return df


def load_gene_axis(parquet: str | Path) -> pd.DataFrame:
    """Load the axis and re-assert its invariants (cheap; do it every time)."""
    df = pd.read_parquet(parquet)
    expected_cols = {"order_index", "symbol", "ensembl_id"}
    if set(df.columns) != expected_cols:
        raise ValueError(f"axis columns {set(df.columns)} != {expected_cols}")
    if list(df["order_index"]) != list(range(len(df))):
        raise ValueError("axis order_index is not 0..n-1 contiguous/sorted")
    if df["symbol"].duplicated().any():
        raise ValueError("axis contains duplicate symbols")
    return df


def axis_symbols(parquet: str | Path) -> list[str]:
    """Ordered list of symbols -- the exact .var_names every artifact must use."""
    return load_gene_axis(parquet)["symbol"].tolist()
