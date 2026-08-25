#!/usr/bin/env python
"""Phase 3 acceptance: does rendering actually move `nreal`?

    python scripts/phase3_acceptance.py --n-perts 8

PLAN.md's regime-sensitivity check, and the experiment that settles whether
downsampling H1 costs us anything. For a handful of perturbations it computes
the number of significant genes BEFORE rendering (native cells, native depth)
and AFTER (400 cells, ~20k UMI), using cell_eval2's own DE routine with the
vcc2026 constants -- so "reference DE" and "scored DE" are literally the same
code path, as CLAUDE.md requires.

Reading the result:
  * nreal barely moves  -> rendering is cheap insurance; full-depth estimates
                           lose little and the surplus was not buying power.
  * nreal moves a lot   -> rendering is load-bearing: a reference measured at
                           H1's native 53k UMI would calibrate four of the six
                           scored metrics to the wrong regime.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vccjudge.regime import NON_TARGETING, group_rng  # noqa: E402

# Pulled from EvalConfig.from_preset("vcc2026") — do not hand-edit; re-read the
# preset if cell-eval2 is upgraded.
VCC2026_DE = dict(backend="scanpy", mean_calc="arithmetic", epsilon=1e-9,
                  input_type="counts", target_sum=1e6, clip_value=None,
                  filter_gene_min_cpm_cell=5.0, fdr_scope="per_pert")
P_ADJ = 0.05


def de_ngenes(adata, groupby="perturbation") -> dict[str, int]:
    """Significant genes per perturbation, by the competition's own test."""
    from cell_eval2.de_compute import compute_de
    df = compute_de(adata, groupby=groupby, reference=NON_TARGETING, **VCC2026_DE)
    d = df.to_pandas() if hasattr(df, "to_pandas") else df
    padj = next(c for c in d.columns if "adj" in c.lower() or c.lower() in ("fdr", "qval"))
    tgt = next(c for c in d.columns if c.lower() in ("target", "target_gene", "perturbation", "group"))
    sig = d[d[padj] < P_ADJ]
    return sig.groupby(tgt).size().to_dict()


def main() -> int:
    import anndata as ad

    ap = argparse.ArgumentParser()
    ap.add_argument("--harmonized", default="data/harmonized/vcc2025_h1/H1.h5ad")
    ap.add_argument("--rendered", default="data/regime/vcc2025_h1/H1__seed0.h5ad")
    ap.add_argument("--n-perts", type=int, default=8)
    ap.add_argument("--ntc", type=int, default=6000,
                    help="control cells used on BOTH sides, so only pert-count and depth differ")
    ap.add_argument("--out", default="data/regime/phase3_acceptance.json")
    a = ap.parse_args()

    ren = ad.read_h5ad(a.rendered)
    rp = ren.obs["perturbation"].astype(str)
    full = [p for p, n in rp.value_counts().items() if p != NON_TARGETING and n == 400]
    rng = group_rng("vcc2025_h1", "H1", "__acceptance__", 0)
    chosen = sorted(rng.choice(sorted(full), size=min(a.n_perts, len(full)), replace=False).tolist())
    print(f"{len(full)} full-size groups available; testing {len(chosen)}: {chosen}")

    def subset(adata, perts, n_ntc, tag):
        obs = adata.obs["perturbation"].astype(str)
        keep = [np.flatnonzero((obs == p).to_numpy()) for p in perts]
        ntc = np.flatnonzero((obs == NON_TARGETING).to_numpy())
        r = group_rng("vcc2025_h1", "H1", f"__ntc_{tag}__", 0)
        if ntc.size > n_ntc:
            ntc = np.sort(r.choice(ntc, n_ntc, replace=False))
        idx = np.sort(np.concatenate(keep + [ntc]))
        sub = adata[idx].to_memory() if adata.isbacked else adata[idx].copy()
        umi = np.asarray(sub.X.sum(axis=1)).ravel()
        return sub, float(np.median(umi))

    print("\n[after]  rendered — 400 cells/pert, thinned")
    t0 = time.time()
    sub_a, med_a = subset(ren, chosen, a.ntc, "after")
    n_a = de_ngenes(sub_a)
    print(f"  {sub_a.n_obs:,} cells, median UMI {med_a:,.0f}, DE in {time.time()-t0:.0f}s")
    del sub_a, ren

    print("[before] harmonized — native cells, native depth")
    har = ad.read_h5ad(a.harmonized, backed="r")
    t0 = time.time()
    sub_b, med_b = subset(har, chosen, a.ntc, "before")
    n_cells_b = sub_b.n_obs
    n_b = de_ngenes(sub_b)
    print(f"  {n_cells_b:,} cells, median UMI {med_b:,.0f}, DE in {time.time()-t0:.0f}s")
    har.file.close(); del sub_b

    rows = []
    for p in chosen:
        b, aa = int(n_b.get(p, 0)), int(n_a.get(p, 0))
        rows.append({"perturbation": p, "nreal_before": b, "nreal_after": aa,
                     "ratio": (aa / b) if b else float("nan")})
    df = pd.DataFrame(rows)
    print(f"\n{'perturbation':<14}{'nreal BEFORE':>14}{'nreal AFTER':>13}{'after/before':>14}")
    for _, r in df.iterrows():
        print(f"{r.perturbation:<14}{r.nreal_before:>14,}{r.nreal_after:>13,}{r.ratio:>14.2f}")
    med_ratio = float(np.nanmedian(df.ratio))
    print(f"{'MEDIAN':<14}{df.nreal_before.median():>14,.0f}{df.nreal_after.median():>13,.0f}{med_ratio:>14.2f}")
    print(f"\nbefore: median UMI {med_b:,.0f}   after: {med_a:,.0f}")
    verdict = ("rendering is LOAD-BEARING — nreal changes materially, so a native-depth "
               "reference would miscalibrate the DE metrics"
               if (med_ratio < 0.8 or med_ratio > 1.25) else
               "rendering changes nreal little — the depth surplus was not buying much DE power")
    print(f"VERDICT: {verdict}")
    Path(a.out).write_text(json.dumps(
        {"perturbations": rows, "median_ratio": med_ratio,
         "median_umi_before": med_b, "median_umi_after": med_a,
         "ntc_cells": a.ntc, "verdict": verdict}, indent=2))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
