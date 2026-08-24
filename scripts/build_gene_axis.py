#!/usr/bin/env python
"""Phase 0 entrypoint: build artifacts/gene_axis.parquet from the bundle.

Usage:
  python scripts/build_gene_axis.py path/to/gene_names.csv \
      [--ensembl-map symbol_to_ensembl.csv] [-o artifacts/gene_axis.parquet]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vccjudge.gene_axis import build_gene_axis, load_gene_axis  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gene_names_csv")
    ap.add_argument("--ensembl-map", default=None)
    ap.add_argument("-o", "--out", default="artifacts/gene_axis.parquet")
    a = ap.parse_args()
    df = build_gene_axis(a.gene_names_csv, a.out, ensembl_map=a.ensembl_map)
    load_gene_axis(a.out)  # re-assert invariants
    n_ens = int(df["ensembl_id"].notna().sum())
    print(f"built {a.out}: {len(df)} genes in order; ensembl_id filled for {n_ens}/{len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
