#!/usr/bin/env python
"""Phase 2: write each raw source onto the canonical axis, in canonical order.

    python scripts/phase2_harmonize.py                 # all registered sources
    python scripts/phase2_harmonize.py --only vcc2025_h1
    python scripts/phase2_harmonize.py --verify-only   # re-check existing outputs

Output: `data/harmonized/<dataset>/<cell_line>.h5ad`, every one with `.var_names`
identical and identically ordered to `artifacts/gene_axis.parquet`.

Streaming, because H1 alone is 1.93e9 stored values. Rows are read, remapped and
appended in blocks; peak memory is one block, not one matrix.

PLAN.md's halt rule is enforced here: a source that leaves more than
`--max-drop-rate` of the axis empty stops the pipeline instead of quietly
producing a mostly-zero file.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vccjudge.gene_axis import load_gene_axis  # noqa: E402
from vccjudge.harmonize import (  # noqa: E402
    build_column_map,
    remap_chunk,
    source_gene_ids,
)
from vccjudge.loaders import SOURCES, load_raw  # noqa: E402

HOUSEKEEPING = ["ACTB", "GAPDH", "TUBB", "RPL13A"]


def harmonize_one(dataset, cell_line, axis, out_path, root=".", chunk=10_000, dry_run=False):
    import h5py
    from anndata.io import sparse_dataset, write_elem

    axis_ids = axis["ensembl_id"].tolist()
    axis_syms = axis["symbol"].tolist()

    adata = load_raw(dataset, cell_line, backed="r", root=root)
    try:
        src_ids = source_gene_ids(adata)
        col_map, st = build_column_map(src_ids, axis_ids)
        st.n_cells = int(adata.n_obs)
        if dry_run:
            return st, None

        obs = adata.obs.copy()
        var = pd.DataFrame(
            {"ensembl_id": axis_ids, "order_index": np.arange(len(axis_syms))},
            index=pd.Index(axis_syms, name=None),
        )
        # Flag structural zeros in .var so no downstream stage has to re-derive
        # which genes this source never measured.
        measured = np.zeros(len(axis_syms), dtype=bool)
        measured[col_map[col_map >= 0]] = True
        var["measured_in_source"] = measured

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".h5ad.partial")
        n_axis = len(axis_syms)

        t0 = time.time()
        with h5py.File(tmp, "w") as f:
            f.attrs["encoding-type"] = "anndata"
            f.attrs["encoding-version"] = "0.1.0"
            write_elem(f, "obs", obs)
            write_elem(f, "var", var)
            write_elem(f, "X", sparse.csr_matrix((0, n_axis), dtype=np.float32))
            X_out = sparse_dataset(f["X"])
            for start in range(0, adata.n_obs, chunk):
                stop = min(start + chunk, adata.n_obs)
                block = adata.X[start:stop]
                st.nnz_in += int(block.nnz if sparse.issparse(block) else np.count_nonzero(block))
                out = remap_chunk(block, col_map, n_axis)
                st.nnz_out += int(out.nnz)
                X_out.append(out)
                pct = 100 * stop / adata.n_obs
                el = time.time() - t0
                print(f"    {stop:>7,}/{adata.n_obs:,} ({pct:5.1f}%)  "
                      f"nnz {st.nnz_out:>13,}  {el:5.0f}s  eta {el*(100/pct-1):5.0f}s",
                      flush=True)
        tmp.replace(out_path)
        return st, out_path
    finally:
        if adata.isbacked:
            adata.file.close()


def verify(out_path, axis) -> list[str]:
    """The Phase 2 acceptance test: assert alignment, do not hope for it."""
    import anndata as ad

    problems = []
    a = ad.read_h5ad(out_path, backed="r")
    try:
        got = list(map(str, a.var_names))
        want = axis["symbol"].tolist()
        if got != want:
            if set(got) == set(want):
                problems.append(f"{out_path.name}: .var_names ORDER differs from the axis")
            else:
                problems.append(
                    f"{out_path.name}: .var_names differ from the axis "
                    f"(n={len(got)} vs {len(want)})"
                )
        if "ensembl_id" in a.var.columns:
            if list(map(str, a.var["ensembl_id"])) != axis["ensembl_id"].tolist():
                problems.append(f"{out_path.name}: .var['ensembl_id'] does not match the axis")
        # Round-trip: a housekeeping gene must sit at the same index everywhere.
        for hk in HOUSEKEEPING:
            if hk in want:
                if got.index(hk) != want.index(hk):
                    problems.append(f"{out_path.name}: {hk} at index {got.index(hk)}, axis says {want.index(hk)}")
                break
    finally:
        if a.isbacked:
            a.file.close()
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--axis", default="artifacts/gene_axis.parquet")
    ap.add_argument("--out-dir", default="data/harmonized")
    ap.add_argument("--report", default="data/harmonization_report.md")
    ap.add_argument("--only", default=None)
    ap.add_argument("--chunk", type=int, default=10_000)
    ap.add_argument("--max-drop-rate", type=float, default=0.15,
                    help="halt if this fraction of the AXIS is left empty (PLAN.md: 10-15%%)")
    ap.add_argument("--dry-run", action="store_true", help="compute mapping stats, write nothing")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="rewrite outputs that already exist")
    a = ap.parse_args()

    axis = load_gene_axis(Path(a.root) / a.axis)
    if axis["ensembl_id"].isna().any():
        print("FATAL: axis ensembl_id is not fully populated — run scripts/build_ensembl_map.py first")
        return 1

    rows, problems = [], []
    for (dataset, cell_line), src in sorted(SOURCES.items()):
        if a.only and dataset != a.only:
            continue
        out_path = Path(a.root) / a.out_dir / dataset / f"{cell_line}.h5ad"

        if a.verify_only:
            if out_path.exists():
                problems += verify(out_path, axis)
                print(f"[verify] {dataset}/{cell_line}: {'OK' if not problems else 'PROBLEMS'}")
            else:
                print(f"[skip] {dataset}/{cell_line}: no output at {out_path}")
            continue

        if not (Path(a.root) / src.path).exists():
            print(f"[skip] {dataset}/{cell_line}: source not downloaded")
            continue
        if out_path.exists() and not a.force and not a.dry_run:
            print(f"[skip] {dataset}/{cell_line}: {out_path} exists (--force to rewrite)")
            problems += verify(out_path, axis)
            continue

        print(f"[harmonize] {dataset}/{cell_line} -> {out_path}")
        st, written = harmonize_one(dataset, cell_line, axis, out_path,
                                    root=a.root, chunk=a.chunk, dry_run=a.dry_run)
        rows.append((dataset, cell_line, st))
        print(f"    mapped {st.n_mapped:,}/{st.n_axis_genes:,} axis genes | "
              f"{st.n_structural_zero:,} structural zeros ({100*st.drop_rate:.2f}%) | "
              f"{st.n_source_dropped:,} source genes dropped")

        if st.drop_rate > a.max_drop_rate:
            problems.append(
                f"{dataset}/{cell_line}: {100*st.drop_rate:.1f}% of the axis is structurally "
                f"zero, over the {100*a.max_drop_rate:.0f}% halt threshold"
            )
        if written:
            problems += verify(written, axis)

    if not a.verify_only:
        Path(Path(a.root) / a.report).write_text(_render(rows, problems, a.max_drop_rate))
        print(f"\nwrote {a.report}")

    if problems:
        print(f"\nFAIL — {len(problems)} problem(s):")
        for p in problems:
            print("  -", p)
        return 1
    print("\nPASS — Phase 2: every output is on the canonical axis, in canonical order")
    return 0


def _render(rows, problems, max_drop) -> str:
    L = ["# Phase 2 — harmonization report", "",
         "Join key is the **stable Ensembl gene ID**, never the symbol "
         "(`CLAUDE.md`). Axis genes absent from a source become **structural "
         "zeros**, flagged in `.var['measured_in_source']` and counted below — "
         "never imputed.", ""]
    if rows:
        L += ["| dataset | line | cells | source genes | mapped to axis | structural zeros | source dropped | unresolved |",
              "|---|---|---|---|---|---|---|---|"]
        for d, c, s in rows:
            L.append(f"| {d} | {c} | {s.n_cells:,} | {s.n_source_genes:,} | {s.n_mapped:,} "
                     f"| {s.n_structural_zero:,} ({100*s.drop_rate:.2f}%) "
                     f"| {s.n_source_dropped:,} ({100*s.source_drop_rate:.2f}%) | {s.n_source_unresolved:,} |")
        L += ["", f"Halt threshold: **{100*max_drop:.0f}%** of the axis structurally zero.", "",
              "## Stored values", "", "| dataset | line | nnz in | nnz out | lost |", "|---|---|---|---|---|"]
        for d, c, s in rows:
            lost = s.nnz_in - s.nnz_out
            L.append(f"| {d} | {c} | {s.nnz_in:,} | {s.nnz_out:,} | {lost:,} "
                     f"({100*lost/s.nnz_in:.3f}%) |" if s.nnz_in else
                     f"| {d} | {c} | 0 | 0 | 0 |")
    else:
        L += ["_Nothing harmonized._", ""]
    L += ["", "## Problems", ""]
    L += [f"- ❌ {p}" for p in problems] if problems else ["_None._"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
