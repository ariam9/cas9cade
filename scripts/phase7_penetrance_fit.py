#!/usr/bin/env python
"""Idea 3 step 1: fit penetrance (pi) per perturbation on real regime-rendered data.

    bash scripts/capped.sh --mem 14G -- .venv/bin/python \
        scripts/phase7_penetrance_fit.py --dataset vcc2025_h1 --cell-line H1

Calls `vccjudge.penetrance.fit_penetrance_all` -- the same function later runs
inside the K562 Kaggle kernel (develop-small/run-big) -- and writes one row per
perturbation: pi_hat, the two mixture-component means, log-likelihood, cell
count, convergence, and a `reason` code (`ok`, `unimodal_shifted`,
`unimodal_no_effect`, `too_few_cells`, `gene_not_on_axis`).
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vccjudge.penetrance import fit_penetrance_all  # noqa: E402


def main() -> int:
    import anndata as ad

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="vcc2025_h1")
    ap.add_argument("--cell-line", default="H1")
    ap.add_argument("--rendered", default=None)
    ap.add_argument("--out", default="artifacts/reference")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ren = Path(a.rendered or f"data/regime/{a.dataset}/{a.cell_line}__seed{a.seed}.h5ad")
    if not ren.exists():
        sys.exit(f"FATAL: {ren} not found — run scripts/phase3_render.py first")

    print(f"fitting penetrance from {ren}", flush=True)
    t0 = time.time()
    adata = ad.read_h5ad(ren)
    df = fit_penetrance_all(adata)
    print(f"  {len(df)} perturbations fitted in {time.time()-t0:.0f}s")

    out_path = Path(a.out) / f"{a.cell_line.lower()}_penetrance_fit__seed{a.seed}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"wrote {out_path}")

    print(f"\nreason counts:\n{df.reason.value_counts().to_string()}")
    ok = df[df.reason.isin(["ok", "unimodal_shifted", "unimodal_no_effect"])]
    print(f"\npi_hat distribution (n={len(ok)}, excluding too_few_cells/gene_not_on_axis):")
    print(ok.pi_hat.describe().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
