#!/usr/bin/env python
"""Idea 4 kill-check: how much does emission itself cost you?

    bash scripts/capped.sh --mem 14G -- \
        .venv/bin/python scripts/phase7_emission_killcheck.py

The idea's own recipe: take real reference perturbation data, compute its true
DE table, discard the cells, regenerate 400 cells from the table alone, score
the regenerated cells. The gap to the real cells' score is the emission loss.

"The real cells' score" needs an operational definition. H1 has three
independently-rendered regime seeds on disk (different random draws +
thinning, same underlying native cells). We use seed0 as ground truth ("real.h5ad")
and seed1's real cells for the SAME perturbations as the "achievable ceiling"
prediction -- an actual independent real draw, not a self-identical trivial
ceiling. The regenerated prediction uses ONLY seed0's real DE table + seed0's
own real control cells (emission.regenerate_from_real) -- exactly what a
predictor has access to. Both predictions are scored against the same real.h5ad
and the same anchors (baseline + bundle built once, reused for both `run`
calls), so the gap isolates what regeneration costs relative to an actual
independent real sample, not scoring noise between separately-built anchors.

Kill criterion (verbatim from the doc): if the gap on the four DE-table metrics
(jac/nmae/fid/reach) is under ~5% of the achievable (ceiling) score, emission
is already near-lossless -- do not build the calibrated emitter into anything
downstream.

Tested perturbations are drawn disjoint from phase7_emission_calibrate.py's
calibration set, so the kill-check isn't graded on the same perturbations the
dispersion_scale was tuned on.
"""
from __future__ import annotations

import argparse, json, shutil, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vccjudge.emission import regenerate_from_real  # noqa: E402
from vccjudge.regime import NON_TARGETING, group_rng  # noqa: E402

NOISE = ("RuntimeWarning", "scores = ", "invalid value", "omits", "de_lfc_nmae:",
         "LOAD-BEARING", "allow_fractional", "config_digest", "it/s]", "%|")
DE_METRICS = ("de_wilcoxon_sig_jaccard", "de_wilcoxon_lfc_nmae",
              "de_wilcoxon_direction_fidelity_yield_raw", "de_wilcoxon_direction_reach_raw")


def sh(cmd, label):
    print(f"  [{label}]", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    for ln in (p.stdout + p.stderr).splitlines():
        if ln.strip() and not any(n in ln for n in NOISE):
            print("     ", ln[:190])
    if p.returncode:
        sys.exit(f"FATAL: {label} failed ({p.returncode})")
    print(f"      {time.time()-t0:.0f}s")


def main() -> int:
    import anndata as ad
    from scipy import sparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--rendered-real", default="data/regime/vcc2025_h1/H1__seed0.h5ad")
    ap.add_argument("--rendered-ceiling", default="data/regime/vcc2025_h1/H1__seed1.h5ad")
    ap.add_argument("--de-table", default="artifacts/reference/reference_de__vcc2025_h1__H1.parquet")
    ap.add_argument("--calibration", default="artifacts/reference/emission_calibration.json")
    ap.add_argument("--n-perts", type=int, default=8)
    ap.add_argument("--ntc", type=int, default=6000,
                    help="control cells on all three files; a smoke-test scale, per "
                         "phase6_harness.py's own note that 18,400 controls densified "
                         "is what kills a laptop-sized ceiling")
    ap.add_argument("--out", default="data/harness/phase7_emission_killcheck")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ce2", default=".venv/bin/cell-eval2")
    a = ap.parse_args()

    calib = json.loads(Path(a.calibration).read_text())
    dispersion_scale = calib["best_dispersion_scale"]
    calib_perts = set(calib["chosen_perturbations"])
    print(f"using dispersion_scale={dispersion_scale} from {a.calibration}")

    real = ad.read_h5ad(a.rendered_real)
    ceiling_src = ad.read_h5ad(a.rendered_ceiling)
    axis = list(map(str, real.var_names))
    obs = real.obs["perturbation"].astype(str).to_numpy()
    vc = pd.Series(obs).value_counts()
    full = sorted(p for p in vc.index if p != NON_TARGETING and vc[p] == 400)
    eligible = [p for p in full if p not in calib_perts]
    rng0 = group_rng("vcc2025_h1", "H1", "__idea4_killcheck__", a.seed)
    chosen = sorted(rng0.choice(eligible, size=min(a.n_perts, len(eligible)), replace=False).tolist())
    print(f"testing {len(chosen)} perturbations (disjoint from calibration set): {chosen}")

    de_table = pd.read_parquet(a.de_table)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # ---- ground truth: real.h5ad (seed0) ----------------------------------
    ntc_rows = np.flatnonzero(obs == NON_TARGETING)
    ntc_rows = np.sort(rng0.choice(ntc_rows, min(a.ntc, ntc_rows.size), replace=False))
    keep = np.sort(np.concatenate([ntc_rows] + [np.flatnonzero(obs == p) for p in chosen]))
    truth = real[keep].copy()
    truth.write_h5ad(out / "real.h5ad")
    print(f"ground truth: {truth.n_obs:,} cells ({len(chosen)} perts + {ntc_rows.size:,} controls)")
    del truth

    # ---- ceiling prediction: seed1's real cells, same perturbations -------
    cobs = ceiling_src.obs["perturbation"].astype(str).to_numpy()
    cntc_rows = np.flatnonzero(cobs == NON_TARGETING)
    cntc_rows = np.sort(rng0.choice(cntc_rows, min(a.ntc, cntc_rows.size), replace=False))
    ckeep = np.sort(np.concatenate([cntc_rows] + [np.flatnonzero(cobs == p) for p in chosen]))
    ceiling = ceiling_src[ckeep].copy()
    ceiling.write_h5ad(out / "pred_ceiling.h5ad")
    print(f"ceiling prediction (seed1, independent real draw): {ceiling.n_obs:,} cells")
    del ceiling, ceiling_src

    # ---- regenerated prediction: real DE table + real seed0 controls ------
    ctrl_pool = real.X.tocsr()[np.flatnonzero(obs == NON_TARGETING)]
    blocks, labels = [], []
    for p in chosen:
        r = group_rng("vcc2025_h1", "H1", f"__idea4_kc_ctrl__{p}", a.seed)
        pick = np.sort(r.choice(ctrl_pool.shape[0], 400, replace=False))
        rgen_rng = group_rng("vcc2025_h1", "H1", f"__idea4_kc_regen__{p}", a.seed)
        regen = regenerate_from_real(de_table, p, axis, ctrl_pool[pick], rgen_rng,
                                      dispersion_scale=dispersion_scale)
        blocks.append(regen.tocsr()); labels += [p] * regen.shape[0]
    pred_ctrl_block = ctrl_pool[np.sort(rng0.choice(ctrl_pool.shape[0], ntc_rows.size, replace=False))]
    X = sparse.vstack(blocks + [pred_ctrl_block]).tocsr(); X.eliminate_zeros()
    regen_obs = pd.DataFrame({"perturbation": labels + [NON_TARGETING] * pred_ctrl_block.shape[0]})
    regen_ad = ad.AnnData(X=X.astype(np.float32), obs=regen_obs, var=pd.DataFrame(index=pd.Index(axis)))
    regen_ad.obs_names_make_unique()
    regen_ad.write_h5ad(out / "pred_regen.h5ad")
    print(f"regenerated prediction: {regen_ad.n_obs:,} cells")
    del real, ctrl_pool, regen_ad

    # ---- score both against the SAME anchors -------------------------------
    preset = ["--preset", "vcc2026", "--pert-col", "perturbation"]
    bp = out / "baseline_pred.h5ad"
    if (out / "bundle").exists():
        shutil.rmtree(out / "bundle")
    sh([a.ce2, "baseline", "-ar", str(out / "real.h5ad"), *preset, "--emit", "dispersed",
        "--seed", "0", "-o", str(out / "baseline"), "--save-pred", str(bp)], "baseline")
    sh([a.ce2, "prep-real-bundle", "--real", str(out / "real.h5ad"), "--baseline", str(bp),
        "-o", str(out / "bundle"), *preset, "--anchor-base-seed", "0", "--anchor-splits", "5"],
       "anchors (shared by both scoring runs)")

    scores = {}
    for tag, pred_name in (("ceiling", "pred_ceiling.h5ad"), ("regenerated", "pred_regen.h5ad")):
        run_dir = out / f"run_{tag}"
        sh([a.ce2, "run", "-ap", str(out / pred_name), "-ar", str(out / "real.h5ad"), *preset,
            "-o", str(run_dir)], f"score: {tag}")
        sh([a.ce2, "score", "--user-agg", str(run_dir / "agg_results.csv"),
            "--real-bundle", str(out / "bundle"), "-o", str(out / f"score_{tag}.csv")],
           f"scale: {tag}")
        scores[tag] = pd.read_csv(out / f"score_{tag}.csv").set_index("metric")["from_replicate"]

    ceil_s, regen_s = scores["ceiling"], scores["regenerated"]
    gap = ceil_s - regen_s

    print("\n" + "=" * 70)
    print(f"{'metric':<44}{'ceiling':>9}{'regen':>9}{'gap':>9}")
    for m in ceil_s.index:
        print(f"{m:<44}{ceil_s[m]:>9.4f}{regen_s[m]:>9.4f}{gap[m]:>9.4f}")

    # Three readings, reported together rather than picking one: a signed sum
    # lets metrics cancel (sig_jaccard is expected to look inflated here, since
    # the belief is copied straight from the ground-truth DE table -- that is
    # not evidence emission is lossless, it is a property of this test's
    # design). An unsigned sum and the whole-pipeline avg_score gap are both
    # conservative in a way the signed sum is not.
    present = [m for m in DE_METRICS if m in ceil_s.index]
    denom = float(ceil_s[present].abs().sum()) if present else float("nan")
    gap_signed = float(gap[present].sum() / denom) if present and denom else float("nan")
    gap_unsigned = float(gap[present].abs().sum() / denom) if present and denom else float("nan")
    gap_avg_score = float(gap["avg_score"] / abs(ceil_s["avg_score"])) if "avg_score" in ceil_s else float("nan")

    print(f"\nDE-table metric gap, signed sum ({', '.join(present)}):   {gap_signed:>7.1%}")
    print(f"DE-table metric gap, unsigned sum:                        {gap_unsigned:>7.1%}")
    print(f"avg_score gap (whole-pipeline bottom line):               {gap_avg_score:>7.1%}")
    worst = max(gap_unsigned, gap_avg_score)
    verdict = ("PASS (kill) -- emission is already near-lossless; effort belongs elsewhere"
               if worst < 0.05 else
               "FAIL (build the calibrated emitter) -- meaningful score is on the table")
    print(f"VERDICT (driven by the more conservative of unsigned-sum/avg_score, "
          f"n={len(chosen)} perturbations): {verdict}")

    json.dump({"chosen_perturbations": chosen, "dispersion_scale": dispersion_scale,
               "scores": {k: v.to_dict() for k, v in scores.items()},
               "gap_fraction_signed": gap_signed, "gap_fraction_unsigned": gap_unsigned,
               "gap_fraction_avg_score": gap_avg_score, "verdict": verdict},
              open(out / "killcheck_result.json", "w"), indent=2)
    print(f"\nwrote {out / 'killcheck_result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
