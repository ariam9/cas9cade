#!/usr/bin/env python
"""Phase 6: run a predictor against a held-out line and score it.

    python scripts/phase6_harness.py --held-out vcc2025_h1/H1 \
        --effects-from replogle2022/K562 --predictor delta_transfer

Splits the held-out line's rendered cells into what a predictor may see (its
non-targeting controls) and what it is scored against (its perturbed cells),
builds that line's own 0/1 anchors with cell-eval2's vcc2026 preset, and reports
scaled metrics.

The scale is per-line by construction, so a shallow reference does not read as a
bad predictor: 1.0 always means "as good as a split-half replicate OF THIS LINE".
"""
from __future__ import annotations

import argparse, json, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vccjudge.harness import (  # noqa: E402
    CELLS_PER_PERT, ContextMean, DeltaTransfer, assemble, leakage_check,
)
from vccjudge.regime import NON_TARGETING, group_rng  # noqa: E402

NOISE = ("RuntimeWarning", "scores = ", "invalid value", "omits", "de_lfc_nmae:",
         "LOAD-BEARING", "allow_fractional", "config_digest", "it/s]", "%|")


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
    ap.add_argument("--held-out", required=True, help="dataset/cell_line, e.g. vcc2025_h1/H1")
    ap.add_argument("--effects-from", default=None, help="dataset/cell_line supplying the deltas")
    ap.add_argument("--predictor", default="delta_transfer",
                    choices=["delta_transfer", "context_mean"])
    ap.add_argument("--regime-dir", default="data/regime")
    ap.add_argument("--ref-dir", default="artifacts/reference")
    ap.add_argument("--out", default="data/harness")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-perts", type=int, default=None)
    ap.add_argument("--all-perts", action="store_true",
                    help="score EVERY perturbation of the held-out line, not just those the "
                         "effects source covers. Uncovered ones fall back to the control "
                         "profile and score ~0 -- which is what the leaderboard does with the "
                         "28 panel genes no reference measures. Restricting to shared "
                         "perturbations scores a 22x-easier subset (median nreal 178 vs 8) "
                         "and inflates the result.")
    ap.add_argument("--max-ntc", type=int, default=None,
                    help="subsample the control arm. The scorer carries it into every "
                         "comparison, so it dominates memory: 18,400 controls x 18,533 "
                         "genes densified is what kills a 13 GB ceiling. Use for smoke "
                         "tests only -- the anchors are only meaningful at full size.")
    ap.add_argument("--ce2", default=".venv/bin/cell-eval2")
    a = ap.parse_args()

    ho_ds, ho_line = a.held_out.split("/")
    src = next(Path(a.regime_dir, ho_ds).glob(f"*{ho_line}*seed{a.seed}.h5ad"), None)
    if src is None:
        sys.exit(f"FATAL: no rendered cells for {a.held_out} — run scripts/phase3_render.py")
    out = Path(a.out) / f"{ho_ds}__{ho_line}__{a.predictor}"
    out.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(src)
    obs = adata.obs["perturbation"].astype(str).to_numpy()
    ctrl_rows = np.flatnonzero(obs == NON_TARGETING)
    if a.max_ntc and ctrl_rows.size > a.max_ntc:
        _r = group_rng(ho_ds, ho_line, "__ntc_subsample__", a.seed)
        ctrl_rows = np.sort(_r.choice(ctrl_rows, a.max_ntc, replace=False))
    controls = adata.X.tocsr()[ctrl_rows]
    axis = list(map(str, adata.var_names))
    print(f"held out {a.held_out}: {adata.n_obs:,} cells, {ctrl_rows.size:,} controls visible")

    # Build the predictor. Anything it uses must exclude the held-out line.
    if a.predictor == "context_mean":
        pred = ContextMean()
        targets = sorted(set(obs.tolist()) - {NON_TARGETING})
    else:
        if not a.effects_from:
            sys.exit("FATAL: --effects-from is required for delta_transfer")
        ef_ds, ef_line = a.effects_from.split("/")
        leakage_check(f"{ho_ds}/{ho_line}", f"{ef_ds}/{ef_line}")
        ef = pd.read_parquet(Path(a.ref_dir) / f"reference_effects__{ef_ds}__{ef_line}.parquet")
        ef = ef.set_index("perturbation").drop(columns=[c for c in ("n_cells",) if c in ef.columns])
        ef = ef.reindex(columns=axis).fillna(0.0)
        ctrl_sum = np.asarray(controls.sum(axis=0)).ravel()
        ctrl_cpm = ctrl_sum / max(ctrl_sum.sum(), 1) * 1e6
        pred = DeltaTransfer(effects=ef, control_cpm=ctrl_cpm)
        allp = sorted(set(obs.tolist()) - {NON_TARGETING})
        if a.all_perts:
            targets = allp
            cov = sum(1 for t in targets if t in ef.index)
            print(f"  predicting from {a.effects_from}: {len(targets)} perturbations, "
                  f"{cov} covered ({100*cov/len(targets):.1f}%), {len(targets)-cov} fall back to control")
        else:
            targets = sorted(set(allp) & set(ef.index))
            print(f"  predicting from {a.effects_from}: {len(targets)} perturbations in both lines")
    if a.limit_perts and len(targets) > a.limit_perts:
        _r = group_rng(ho_ds, ho_line, "__pert_sample__", a.seed)
        targets = sorted(_r.choice(targets, a.limit_perts, replace=False).tolist())

    # Ground truth: the held-out line's own perturbed cells for those targets,
    # plus its controls (the scorer needs the reference arm).
    keep = np.sort(np.concatenate(
        [ctrl_rows] + [np.flatnonzero(obs == t) for t in targets]))
    truth = adata[keep].copy()
    truth.write_h5ad(out / "real.h5ad")
    print(f"  ground truth: {truth.n_obs:,} cells over {len(targets)} perturbations + controls")

    rng = group_rng(ho_ds, ho_line, f"__harness_{pred.name}__", a.seed)
    t0 = time.time()
    blocks = pred.predict(controls, targets, rng)
    pa = assemble(blocks, axis)
    # The scorer needs the control arm present on the prediction side too.
    ctrl_block = controls[np.sort(rng.choice(controls.shape[0],
                          size=min(ctrl_rows.size, controls.shape[0]), replace=False))]
    pa = ad.concat([pa, ad.AnnData(X=ctrl_block.astype(np.float32),
                    obs=pd.DataFrame({"perturbation": [NON_TARGETING]*ctrl_block.shape[0]}),
                    var=pd.DataFrame(index=pd.Index(axis)))], join="outer")
    pa.obs_names_make_unique()
    pa.write_h5ad(out / "pred.h5ad")
    print(f"  prediction: {pa.n_obs:,} cells in {time.time()-t0:.0f}s -> {out/'pred.h5ad'}")
    del adata, truth, pa

    preset = ["--preset", "vcc2026", "--pert-col", "perturbation"]
    bp = out / "baseline_pred.h5ad"
    if (out / "bundle").exists():
        import shutil; shutil.rmtree(out / "bundle")
    sh([a.ce2, "baseline", "-ar", str(out/"real.h5ad"), *preset, "--emit", "dispersed",
        "--seed", "0", "-o", str(out/"baseline"), "--save-pred", str(bp)], "0-end: baseline")
    sh([a.ce2, "prep-real-bundle", "--real", str(out/"real.h5ad"), "--baseline", str(bp),
        "-o", str(out/"bundle"), *preset, "--anchor-base-seed", "0", "--anchor-splits", "5"],
       "scale: both ends")
    sh([a.ce2, "run", "-ap", str(out/"pred.h5ad"), "-ar", str(out/"real.h5ad"), *preset,
        "-o", str(out/"run")], "score the prediction")
    sh([a.ce2, "score", "--user-agg", str(out/"run/agg_results.csv"),
        "--real-bundle", str(out/"bundle"), "-o", str(out/"score.csv")], "scale it")

    s = pd.read_csv(out / "score.csv").set_index("metric")["from_replicate"]
    print("\n" + "="*62)
    print(f"  {a.predictor}: {a.effects_from or '—'} -> {a.held_out}")
    print("="*62)
    for m, v in s.items():
        print(f"  {m:<44}{v:>10.4f}")
    json.dump({"held_out": a.held_out, "effects_from": a.effects_from,
               "predictor": a.predictor, "n_perturbations": len(targets),
               "scores": {k: float(v) for k, v in s.items()}},
              open(out / "harness_result.json", "w"), indent=2)
    print(f"\nwrote {out/'harness_result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
