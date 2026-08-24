#!/usr/bin/env python
"""
Phase 0 acceptance test.

Certifies two things before ANY later phase is allowed to depend on Phase 0:
  (1) the gene axis builds and round-trips with its invariants intact;
  (2) the contract validator PASSES a well-formed submission AND CATCHES every
      individual contract violation (a validator that never fails is useless).

Uses a small synthetic axis so it runs in seconds on a laptop; the identical
code runs against the real 18,533-gene bundle. Exit code 0 == certified.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vccjudge.contract import validate_submission  # noqa: E402
from vccjudge.gene_axis import axis_symbols, build_gene_axis, load_gene_axis  # noqa: E402
from vccjudge.synth import make_random_submission  # noqa: E402

CELLS = 400
GENES = [f"GENE{i:03d}" for i in range(60)]   # stand-in for the 18,533 symbols
PERTS = GENES[:6]                             # perturbed genes must be on the axis
CONTEXTS = ["A", "B"]


def _check(name: str, condition: bool) -> bool:
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    return condition


def test_gene_axis(tmp: Path) -> bool:
    ok = True
    csv = tmp / "gene_names.csv"
    pd.Series(GENES).to_csv(csv, index=False, header=False)
    out = tmp / "gene_axis.parquet"
    build_gene_axis(csv, out)
    df = load_gene_axis(out)
    ok &= _check("axis length matches input", len(df) == len(GENES))
    ok &= _check("axis order preserved", axis_symbols(out) == GENES)
    ok &= _check("order_index is 0..n-1", list(df.order_index) == list(range(len(GENES))))
    # header row is dropped if present
    csv2 = tmp / "gene_names_hdr.csv"
    pd.Series(["gene"] + GENES).to_csv(csv2, index=False, header=False)
    build_gene_axis(csv2, tmp / "axis2.parquet")
    ok &= _check("header row auto-dropped", axis_symbols(tmp / "axis2.parquet") == GENES)
    return ok


def test_valid_submission_passes() -> bool:
    adata = make_random_submission(GENES, PERTS, CONTEXTS, cells_per_pert=CELLS, seed=1)
    violations = validate_submission(
        adata, GENES, pert_list=PERTS, allowed_contexts=CONTEXTS, cells_per_pert=CELLS
    )
    if violations:
        for m in violations:
            print("     unexpected violation:", m)
    return _check("well-formed submission passes with 0 violations", violations == [])


def test_cli_gate(tmp: Path) -> bool:
    """The readiness gate must reserve the word PASS for a fully-checked file.

    `validate_submission` skips contract items 2 and 8 when --perts/--contexts
    are absent, so a bare CLI invocation checks less than it appears to. This
    certifies that the CLI says so out loud instead of printing PASS.
    """
    import contextlib
    import io

    from vccjudge.contract import _main

    csv = tmp / "cli_genes.csv"
    pd.Series(GENES).to_csv(csv, index=False, header=False)
    axis = tmp / "cli_axis.parquet"
    build_gene_axis(csv, axis)

    good = tmp / "good.h5ad"
    make_random_submission(GENES, PERTS, CONTEXTS, cells_per_pert=CELLS, seed=3).write_h5ad(good)
    bad = tmp / "bad.h5ad"
    _break_cell_count(
        make_random_submission(GENES, PERTS, CONTEXTS, cells_per_pert=CELLS, seed=4)
    ).write_h5ad(bad)

    perts_csv = tmp / "pert_counts.csv"
    pd.DataFrame({"target_gene": PERTS, "n_cells": CELLS}).to_csv(perts_csv, index=False)

    full = ["--axis", str(axis), "--perts", str(perts_csv), "--contexts", ",".join(CONTEXTS)]

    def run(argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = _main(argv)
        return code, buf.getvalue()

    ok = True
    code, out = run([str(good), *full])
    ok &= _check("CLI: valid file, all args -> PASS exit 0", code == 0 and "PASS" in out)

    code, out = run([str(good), "--axis", str(axis)])
    ok &= _check(
        "CLI: no --perts/--contexts -> INCOMPLETE exit 1, never says PASS",
        code == 1 and "INCOMPLETE" in out and "PASS" not in out,
    )

    code, out = run([str(good), "--axis", str(axis), "--allow-partial"])
    ok &= _check(
        "CLI: --allow-partial -> PARTIAL exit 0, never says PASS",
        code == 0 and "PARTIAL" in out and "PASS" not in out,
    )

    code, out = run([str(bad), *full])
    ok &= _check(
        "CLI: contract violation -> FAIL exit 1, never says PASS",
        code == 1 and "FAIL" in out and "PASS" not in out,
    )
    return ok


def _expect_violation(name: str, mutate, **kw) -> bool:
    """Build a valid submission, break it, expect >=1 violation."""
    adata = make_random_submission(GENES, PERTS, CONTEXTS, cells_per_pert=CELLS, seed=2)
    adata = mutate(adata)
    violations = validate_submission(
        adata, GENES, pert_list=PERTS, allowed_contexts=CONTEXTS, cells_per_pert=CELLS, **kw
    )
    return _check(f"caught: {name}", len(violations) >= 1)


def test_total_cell_cap() -> bool:
    """Arc's `vcc prep --max-cell-dim` (400,000) rejects an oversized file.

    Tested with a lowered cap rather than by building a 400k-cell fixture: the
    constant is the contract's, the comparison is what needs certifying.
    """
    adata = make_random_submission(GENES, PERTS, CONTEXTS, cells_per_pert=CELLS, seed=5)
    over = validate_submission(
        adata, GENES, pert_list=PERTS, allowed_contexts=CONTEXTS,
        cells_per_pert=CELLS, max_total_cells=adata.n_obs - 1,
    )
    under = validate_submission(
        adata, GENES, pert_list=PERTS, allowed_contexts=CONTEXTS,
        cells_per_pert=CELLS, max_total_cells=adata.n_obs,
    )
    return _check("caught: total cells over --max-cell-dim", len(over) >= 1) and _check(
        "exactly at the cap is allowed (no off-by-one)", under == []
    )


def _break_cell_count(a):
    return a[:-1].copy()  # one group now has 399 cells


def _add_ntc_row(a):
    extra = a[:1].copy()
    extra.obs["target_gene"] = "non-targeting"
    import anndata as ad
    return ad.concat([a, extra])


def _explicit_zeros(a):
    a2 = a.copy(); X = a2.X.tocsr().copy()
    X.data[0] = 0.0  # first stored entry becomes an explicit zero (nnz unchanged)
    a2.X = X; return a2  # deliberately NOT calling eliminate_zeros


def _non_integer(a):
    a2 = a.copy(); X = a2.X.tocsr(); X.data = X.data + 0.5; a2.X = X; return a2


def _over_cap(a):
    a2 = a.copy(); X = a2.X.tolil(); X[0, 1] = 2_000_000; a2.X = X.tocsr(); return a2


def _shuffle_genes(a):
    order = np.random.default_rng(0).permutation(a.n_vars)
    return a[:, order].copy()


def _construct_id(a):
    a2 = a.copy()
    tg = a2.obs["target_gene"].astype(str).to_numpy()
    tg[tg == PERTS[0]] = PERTS[0] + "-1"
    a2.obs["target_gene"] = tg
    return a2


def _dense(a):
    a2 = a.copy(); a2.X = np.asarray(a.X.todense()); return a2


def main() -> int:
    print("Phase 0 acceptance test")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        results = {
            "gene_axis": test_gene_axis(tmp),
            "valid_passes": test_valid_submission_passes(),
        }
        print("  -- readiness-gate CLI checks --")
        results["cli_gate"] = test_cli_gate(tmp)
    print("  -- violation-detection checks --")
    checks = {
        "wrong cell count (399)": _break_cell_count,
        "non-targeting row present": _add_ntc_row,
        "explicit stored zero": _explicit_zeros,
        "non-integer .X": _non_integer,
        "cell over 1e6 counts": _over_cap,
        "genes out of axis order": _shuffle_genes,
        "construct-ID label (GENE000-1)": _construct_id,
        "dense .X": _dense,
    }
    det = {name: _expect_violation(name, fn) for name, fn in checks.items()}
    det["total_cell_cap"] = test_total_cell_cap()

    all_ok = all(results.values()) and all(det.values())
    print("\nRESULT:", "PASS — Phase 0 certified" if all_ok else "FAIL — do not proceed")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
