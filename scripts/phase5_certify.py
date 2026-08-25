#!/usr/bin/env python
"""Phase 5: build the judge's scale for one line, and certify it.

    python scripts/phase5_certify.py --real data/regime/vcc2025_h1/H1__seed0.h5ad

PLAN.md calls this "the one test that certifies the whole judge". A raw metric
value means nothing on its own; the competition rescales it against two anchors,
and unless BOTH reproduce, no number built on the scorer can be trusted:

    0 = the cell-context-mean baseline (paste the average response everywhere)
    1 = a split-half replicate of the real experiment

We wrap `cell-eval2 --preset vcc2026` rather than reimplementing six metrics:
the preset is public, so the local scorer is the competition's own code path.

The chain, all of it upstream:
    baseline          -> the 0 end (a mean profile; fractional by nature)
    prep-real-bundle  -> both ends + the identity binding them
    run               -> raw metrics for one prediction
    score             -> scaled against the bundle

⚠️ Do NOT pass --set to any step. It changes the config hash, and cell-eval2
then marks the bundle DIAGNOSTIC and refuses it the competition average -- for
good reason: "an anchor is a property of ONE dataset scored under ONE
configuration". A pristine preset is what makes a local number comparable.
"""
from __future__ import annotations

import argparse, json, subprocess, sys, time
from pathlib import Path

import pandas as pd

SCORED_SIX = ["pds_cosine", "expr_mse_unbiased_capped_norm", "de_wilcoxon_lfc_nmae",
              "de_wilcoxon_direction_fidelity_yield_raw", "de_wilcoxon_direction_reach_raw",
              "de_wilcoxon_sig_jaccard"]
NOISE = ("RuntimeWarning", "scores = ", "invalid value", "omits", "de_lfc_nmae:",
         "LOAD-BEARING", "allow_fractional", "config_digest", "backend")


def run(cmd: list[str], label: str) -> None:
    print(f"\n[{label}] {' '.join(cmd[:4])} ...", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    for line in (p.stdout + p.stderr).splitlines():
        if line.strip() and not any(n in line for n in NOISE):
            print("   ", line[:200])
    if p.returncode:
        sys.exit(f"FATAL: {label} failed ({p.returncode})")
    print(f"    done in {time.time()-t0:.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, help="rendered, in-regime reference h5ad")
    ap.add_argument("--out", default="data/judge")
    ap.add_argument("--pert-col", default="perturbation")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--ce2", default=".venv/bin/cell-eval2")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    # Re-running must be safe: prep-real-bundle refuses a non-empty target
    # ("two bundles' files interleaved in one directory are unreadable"), which
    # is right, so clear it rather than passing --force blindly.
    import shutil
    if (out / "bundle").exists():
        shutil.rmtree(out / "bundle")
    preset = ["--preset", "vcc2026", "--pert-col", a.pert_col]
    base_pred = out / "baseline_pred.h5ad"

    run([a.ce2, "baseline", "-ar", a.real, *preset, "--emit", "dispersed", "--seed", "0",
         "-o", str(out / "baseline"), "--save-pred", str(base_pred)], "0-end: baseline")
    run([a.ce2, "prep-real-bundle", "--real", a.real, "--baseline", str(base_pred),
         "-o", str(out / "bundle"), *preset,
         "--anchor-base-seed", str(a.base_seed), "--anchor-splits", str(a.splits)],
        "scale: both ends")
    run([a.ce2, "run", "-ap", a.real, "-ar", a.real, *preset,
         "-o", str(out / "run_ceiling")], "1-end: reference predicts itself")
    run([a.ce2, "score", "--user-agg", str(out / "run_ceiling/agg_results.csv"),
         "--real-bundle", str(out / "bundle"), "-o", str(out / "score_ceiling.csv")],
        "scale the ceiling")

    ceil = pd.read_csv(out / "score_ceiling.csv").set_index("metric")["from_replicate"]
    anchor = pd.read_parquet(out / "bundle/anchor_agg.parquet").set_index("metric")

    print("\n" + "=" * 68)
    print(f"{'metric':<44}{'ceiling':>10}{'replicate_raw':>14}")
    for m in SCORED_SIX:
        raw = anchor["replicate"].get(m, float("nan"))
        print(f"  {m:<42}{ceil.get(m, float('nan')):>10.4f}{raw:>14.4f}")
    print(f"  {'avg_score':<42}{ceil.get('avg_score', float('nan')):>10.4f}")

    checks = {
        "every scored metric present": all(m in ceil.index for m in SCORED_SIX),
        "ceiling beats a replicate (avg > 1)": float(ceil.get("avg_score", 0)) > 1.0,
        "mse is capped at exactly 1.0": abs(float(ceil.get("expr_mse_unbiased_capped_norm", 0)) - 1.0) < 1e-6,
        "the other five are NOT capped at 1": sum(
            float(ceil.get(m, 0)) > 1.0001 for m in SCORED_SIX if m != "expr_mse_unbiased_capped_norm") >= 4,
        "bundle is COMPETITION, not diagnostic": "diagnostic" not in
            json.loads((out / "bundle/manifest.json").read_text()).get("kind", "competition").lower(),
    }
    print("\n" + "=" * 68)
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    ok = all(checks.values())
    (out / "phase5_certification.json").write_text(json.dumps(
        {"ceiling": {k: float(v) for k, v in ceil.items() if isinstance(v, (int, float))},
         "checks": checks, "real": a.real, "splits": a.splits, "base_seed": a.base_seed},
        indent=2))
    print("\nRESULT:", "PASS — Phase 5 scale certified" if ok else "FAIL — do not trust local scores")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
