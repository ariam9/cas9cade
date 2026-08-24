"""
The VCC 2026 submission contract, encoded as code.

Every requirement below is a *rejection, never a repair* on Arc's side, and
several failure modes (density cap, gene order, library-size drift) are only
discovered after a multi-gigabyte upload unless you check locally first. So the
rule for this repo is: NO file is called "ready" until `validate_submission`
returns an empty list.

`validate_submission` returns a list of human-readable violation strings
(empty == passes). It is deliberately dependency-light and reads an AnnData
that is already in memory or on disk.

Note the asymmetry between the function and the CLI: `validate_submission`
skips the perturbation-set and context checks when those arguments are None,
because callers legitimately check subsets. The CLI (`python -m
vccjudge.contract`) is the *readiness gate*, so it refuses to print PASS unless
every item was checked -- see the comment above `_main`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

# ---- contract constants (from the vcc2026 spec / briefing) -------------------
CELLS_PER_PERT = 400
MAX_COUNTS_PER_CELL = 1_000_000
MAX_STORED_ENTRIES = 4_750_000_000
#: `vcc prep --max-cell-dim` default. A complete validation submission is exactly
#: 300 perts x 400 cells x 3 contexts = 360,000 cells, so this is headroom, not a
#: constraint you should ever approach -- but Arc rejects on it and we did not
#: check it until the sample file was run through both validators side by side.
MAX_TOTAL_CELLS = 400_000
NON_TARGETING = "non-targeting"
PERT_COL = "target_gene"
CONTEXT_COL = "context"


def _iter_row_totals(X):
    """Yield per-cell total counts without densifying the whole matrix."""
    if sparse.issparse(X):
        return np.asarray(X.sum(axis=1)).ravel()
    return np.asarray(X).sum(axis=1).ravel()


def validate_submission(
    adata,
    gene_axis_symbols: list[str],
    pert_list: list[str] | None = None,
    allowed_contexts: list[str] | None = None,
    cells_per_pert: int = CELLS_PER_PERT,
    max_total_cells: int = MAX_TOTAL_CELLS,
) -> list[str]:
    """Check an AnnData submission against the contract.

    Parameters
    ----------
    adata              : AnnData to check (raw counts in .X).
    gene_axis_symbols  : the canonical ordered symbols (from gene_axis).
    pert_list          : optional exact set of perturbation symbols expected
                         per context (from pert_counts.csv). If given, missing
                         or extra perturbations are violations.
    allowed_contexts   : optional exact set of context labels expected.
    cells_per_pert     : required cells per (context, perturbation) group.
    max_total_cells    : cap on total cells in the file (Arc's --max-cell-dim).

    Returns
    -------
    list[str] of violations; empty means the file satisfies the contract.
    """
    v: list[str] = []

    # --- total cell count cap (Arc's --max-cell-dim) ------------------------
    if adata.n_obs > max_total_cells:
        v.append(
            f"{adata.n_obs:,} cells exceeds the {max_total_cells:,}-cell submission cap"
        )

    # --- genes: exact set AND exact order -----------------------------------
    var_names = list(map(str, adata.var_names))
    if var_names != list(gene_axis_symbols):
        if set(var_names) == set(gene_axis_symbols):
            v.append("gene set matches but ORDER differs from gene_axis; reorder .var to the axis")
        else:
            missing = set(gene_axis_symbols) - set(var_names)
            extra = set(var_names) - set(gene_axis_symbols)
            v.append(
                f".var genes do not match axis (missing {len(missing)}, extra {len(extra)}; "
                f"n_var={len(var_names)}, n_axis={len(gene_axis_symbols)})"
            )

    # --- required obs columns ------------------------------------------------
    for col in (PERT_COL, CONTEXT_COL):
        if col not in adata.obs.columns:
            v.append(f".obs is missing required column '{col}'")
    if PERT_COL not in adata.obs.columns or CONTEXT_COL not in adata.obs.columns:
        return v  # can't run group checks without these

    perts = adata.obs[PERT_COL].astype(str)
    contexts = adata.obs[CONTEXT_COL].astype(str)

    # --- no non-targeting / control rows (inputs only, never emitted) -------
    n_ntc = int((perts.str.lower() == NON_TARGETING).sum())
    if n_ntc:
        v.append(f"{n_ntc} non-targeting/control rows present; control cells are inputs only")

    # --- perturbation labels are gene symbols, not construct IDs ------------
    labels = set(perts.unique()) - {NON_TARGETING}
    construct_like = sorted(l for l in labels if l.rsplit("-", 1)[-1].isdigit() and "-" in l)
    if construct_like:
        v.append(
            f"{len(construct_like)} perturbation labels look like construct IDs, not symbols "
            f"(e.g. {construct_like[:3]}); use 'ADNP', never 'ADNP-1'"
        )
    unknown = sorted(labels - set(gene_axis_symbols))
    if unknown:
        v.append(
            f"{len(unknown)} perturbation labels are not genes on the axis "
            f"(e.g. {unknown[:3]}); a label that does not resolve leaves the target in the scored vector"
        )

    # --- context set ---------------------------------------------------------
    ctx_present = set(contexts.unique())
    if allowed_contexts is not None:
        extra_ctx = ctx_present - set(allowed_contexts)
        missing_ctx = set(allowed_contexts) - ctx_present
        if extra_ctx:
            v.append(f"unexpected context labels: {sorted(extra_ctx)}")
        if missing_ctx:
            v.append(f"missing expected contexts: {sorted(missing_ctx)}")

    # --- exactly N cells per (context, perturbation); exact pert set --------
    grp = adata.obs.groupby([CONTEXT_COL, PERT_COL], observed=True).size()
    bad_counts = grp[grp != cells_per_pert]
    non_ntc_bad = [(c, p, int(n)) for (c, p), n in bad_counts.items() if str(p).lower() != NON_TARGETING]
    if non_ntc_bad:
        v.append(
            f"{len(non_ntc_bad)} (context,perturbation) groups do not have exactly "
            f"{cells_per_pert} cells (e.g. {non_ntc_bad[:3]})"
        )
    if pert_list is not None:
        want = set(pert_list)
        for c in sorted(ctx_present):
            got = set(perts[contexts == c].unique()) - {NON_TARGETING}
            miss, ext = want - got, got - want
            if miss or ext:
                v.append(
                    f"context '{c}': perturbation set mismatch "
                    f"(missing {len(miss)}, extra {len(ext)})"
                )

    # --- .X: one streaming pass for every matrix-level rule -----------------
    # Streamed, not loaded. A real submission is 360,000 x 18,533 with ~2.05e9
    # stored values (~16 GB); reading it whole OOM-killed this very check on a
    # 22 GB machine. The gate has to survive the artifact it exists to gate.
    scan = _scan_X(adata)
    if scan["dense"]:
        v.append(
            ".X is dense; store sparse (a dense 18,533-gene matrix exceeds the density cap on its own)"
        )
    if scan["nonfinite"]:
        v.append(".X contains non-finite values (NaN/inf)")
    if scan["negative"]:
        v.append(".X contains negative values; counts must be non-negative")
    if scan["non_integer"]:
        v.append(".X is not integer-valued; submit raw counts (no normalize_total/log1p)")
    if scan["explicit_zeros"]:
        v.append("explicitly-stored zeros present in .X; call .eliminate_zeros()")
    if scan["n_stored"] > MAX_STORED_ENTRIES:
        v.append(f"stored entries in .X = {scan['n_stored']:,} exceeds cap {MAX_STORED_ENTRIES:,}")
    if scan["n_over_cap"]:
        v.append(f"{scan['n_over_cap']} cells exceed {MAX_COUNTS_PER_CELL:,} total counts")

    return v


def _scan_X(adata, chunk: int = 20_000) -> dict:
    """Single streaming pass over .X collecting every matrix-level verdict.

    Works identically on an in-memory or a backed AnnData; peak memory is one
    row block regardless of how large the submission is.
    """
    out = dict(dense=False, nonfinite=False, negative=False, non_integer=False,
               explicit_zeros=False, n_stored=0, n_over_cap=0)
    n = adata.n_obs
    for start in range(0, max(n, 1), chunk):
        stop = min(start + chunk, n)
        if stop <= start:
            break
        X = adata.X[start:stop]
        if sparse.issparse(X):
            Xc = X.tocsr()
            data = Xc.data
            out["n_stored"] += int(Xc.nnz)
            if data.size and np.any(data == 0):
                out["explicit_zeros"] = True
            totals = np.asarray(Xc.sum(axis=1)).ravel()
        else:
            A = np.asarray(X)
            out["dense"] = True
            data = A.ravel()
            out["n_stored"] += int(A.size)
            totals = A.sum(axis=1).ravel()
        if data.size:
            finite = np.isfinite(data)
            if not finite.all():
                out["nonfinite"] = True
            d = data[finite]
            if d.size:
                if d.min() < 0:
                    out["negative"] = True
                if not np.allclose(d, np.round(d)):
                    out["non_integer"] = True
        if totals.size:
            out["n_over_cap"] += int((totals > MAX_COUNTS_PER_CELL).sum())
    return out


# --- CLI ---------------------------------------------------------------------
# The CLI is the readiness gate CLAUDE.md names, so it must never be able to
# say PASS about a file it did not fully check. `validate_submission` skips
# contract item 2 (exact perturbation set) when `pert_list is None` and item 8
# (context labels) when `allowed_contexts is None` -- both silently. A bare
# invocation therefore used to print PASS on a submission missing 200
# perturbations, which Arc rejects outright and never repairs. So the gate
# reports four distinct outcomes and reserves the word PASS for the one case
# where every item was actually verified.
def _main(argv: list[str] | None = None) -> int:
    import argparse

    import anndata as ad

    from vccjudge.gene_axis import axis_symbols

    ap = argparse.ArgumentParser(description="Validate a VCC2026 submission h5ad against the contract.")
    ap.add_argument("h5ad", help="prediction .h5ad to validate")
    ap.add_argument("--axis", default="artifacts/gene_axis.parquet", help="gene_axis.parquet")
    ap.add_argument("--perts", default=None, help="pert_counts.csv (first column = symbols); required for a full check")
    ap.add_argument("--contexts", default=None, help="comma-separated context labels; required for a full check")
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="dev-loop escape hatch: tolerate unchecked contract items. Reports PARTIAL, never PASS. "
        "Never use this to clear a file for submission.",
    )
    args = ap.parse_args(argv)

    # Backed: the gate must not need more RAM than the file it checks.
    adata = ad.read_h5ad(args.h5ad, backed="r")
    symbols = axis_symbols(args.axis)
    pert_list = None
    if args.perts:
        pert_list = pd.read_csv(args.perts).iloc[:, 0].astype(str).str.strip().tolist()
    allowed = args.contexts.split(",") if args.contexts else None

    unchecked = []
    if pert_list is None:
        unchecked.append("item 2 (exact perturbation set per context) -- pass --perts pert_counts.csv")
    if allowed is None:
        unchecked.append("item 8 (context labels verbatim from the control files) -- pass --contexts A,B,C")

    violations = validate_submission(adata, symbols, pert_list=pert_list, allowed_contexts=allowed)

    if violations:
        print(f"FAIL  {args.h5ad}  ({len(violations)} violation(s)):")
        for i, msg in enumerate(violations, 1):
            print(f"  {i}. {msg}")
        if unchecked:
            print(f"  (plus {len(unchecked)} contract item(s) not checked -- this list may be incomplete)")
        return 1

    if not unchecked:
        print(f"PASS  {args.h5ad}  (contract satisfied; all items checked)")
        return 0

    if args.allow_partial:
        print(f"PARTIAL  {args.h5ad}  (no violations found, but {len(unchecked)} item(s) unchecked):")
        for i, msg in enumerate(unchecked, 1):
            print(f"  {i}. {msg}")
        return 0

    print(f"INCOMPLETE  {args.h5ad}  ({len(unchecked)} contract item(s) could not be checked):")
    for i, msg in enumerate(unchecked, 1):
        print(f"  {i}. {msg}")
    print("  no violations were found in the items that WERE checked, but this file is not cleared.")
    print("  supply the missing arguments, or use --allow-partial for a dev-loop check.")
    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
