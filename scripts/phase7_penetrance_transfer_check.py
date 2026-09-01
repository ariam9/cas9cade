#!/usr/bin/env python
"""Idea 3 step 2, local half: fit the basal-state -> pi model on H1.

    .venv/bin/python scripts/phase7_penetrance_transfer_check.py

Writes ONLY the fitted coefficients (a few floats) to
artifacts/reference/h1_penetrance_basal_model.json. The K562 Kaggle kernel
applies this model to K562's own basal covariates (computable from ITS
controls alone -- leakage-safe) and compares the resulting n_pred against
K562's real nreal, entirely on Kaggle. No per-cell data crosses the wire
either direction for this check.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vccjudge.penetrance import basal_features_for_genes, fit_basal_pi_model  # noqa: E402


def main() -> int:
    import anndata as ad

    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="artifacts/reference/h1_penetrance_fit__seed0.parquet")
    ap.add_argument("--rendered", default="data/regime/vcc2025_h1/H1__seed0.h5ad")
    ap.add_argument("--out", default="artifacts/reference/h1_penetrance_basal_model.json")
    a = ap.parse_args()

    pi_table = pd.read_parquet(a.fit)
    pi_table = pi_table[pi_table.reason.isin(["ok", "unimodal_shifted", "unimodal_no_effect"])]
    print(f"{len(pi_table)} perturbations with a usable pi_hat")

    adata = ad.read_h5ad(a.rendered)
    basal = basal_features_for_genes(adata, pi_table["perturbation"].tolist())

    model = fit_basal_pi_model(pi_table, basal)
    print(f"fitted basal-state -> pi model: {model}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(model, indent=2))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
