#!/usr/bin/env python
"""Idea 3 step-1 kill-check: is the fitted penetrance distribution degenerate?

    .venv/bin/python scripts/phase7_penetrance_killcheck.py

Kill criterion (verbatim from the doc): if fitted pi is close to 1 for nearly
all perturbations in reference data, the mixture is unnecessary and a simpler
fixed-dispersion decoder wins. Operationalized as: fraction with pi_hat > 0.95
must be well under "nearly all" -- the threshold below is 90%.

As a bonus sanity check (not the literal kill criterion, but cheap and
directly diagnostic): correlate pi_hat against nreal from the same reference
line -- a real, nameable penetrance signal should track detectability.

Run this BEFORE spending Kaggle time on step 2 (predicting pi from basal
state on K562) -- it is the cheapest check in the whole phase 7 plan and gates
real wall-clock time.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="artifacts/reference/h1_penetrance_fit__seed0.parquet")
    ap.add_argument("--nreal", default="artifacts/reference/nreal_table__vcc2025_h1__H1.parquet")
    ap.add_argument("--degenerate-threshold", type=float, default=0.95)
    ap.add_argument("--degenerate-fraction", type=float, default=0.90,
                    help="kill if this fraction (or more) of perturbations have "
                         "pi_hat above --degenerate-threshold")
    ap.add_argument("--out", default="artifacts/reference/phase7_penetrance_killcheck.json")
    a = ap.parse_args()

    df = pd.read_parquet(a.fit)
    fitted = df[df.reason.isin(["ok", "unimodal_shifted", "unimodal_no_effect"])]
    if fitted.empty:
        sys.exit(f"FATAL: no perturbations had a usable fit in {a.fit}")

    frac_near_one = float((fitted.pi_hat > a.degenerate_threshold).mean())
    print(f"n perturbations fitted: {len(fitted)} (of {len(df)} total)")
    print(f"pi_hat distribution:\n{fitted.pi_hat.describe().to_string()}")
    print(f"\nfraction with pi_hat > {a.degenerate_threshold}: {frac_near_one:.1%}")

    degenerate = frac_near_one >= a.degenerate_fraction
    verdict = ("KILL -- pi is close to 1 for nearly all perturbations; a simpler "
               "fixed-dispersion decoder wins"
               if degenerate else
               "CONTINUE -- pi is non-degenerate; the mixture carries real information")
    print(f"\nVERDICT: {verdict}")

    corr = None
    nreal_path = Path(a.nreal)
    if nreal_path.exists():
        nreal = pd.read_parquet(nreal_path)[["perturbation", "nreal"]]
        merged = fitted.merge(nreal, on="perturbation", how="inner")
        if len(merged) > 2:
            corr = float(merged["pi_hat"].corr(np.log1p(merged["nreal"]), method="spearman"))
            print(f"\nsanity check: Spearman(pi_hat, log1p(nreal)) = {corr:.3f} "
                  f"(n={len(merged)}) -- expect positive if pi tracks detectability")

    json.dump({"n_fitted": len(fitted), "n_total": len(df),
               "degenerate_threshold": a.degenerate_threshold,
               "fraction_near_one": frac_near_one, "degenerate": degenerate,
               "verdict": verdict, "spearman_pi_vs_log_nreal": corr},
              open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
