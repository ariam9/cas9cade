#!/usr/bin/env python
"""Idea 4 calibration: pick emission.py's dispersion_scale from real H1 data.

    bash scripts/capped.sh --mem 14G -- \
        .venv/bin/python scripts/phase7_emission_calibrate.py

For a handful of real, full-size (400-cell) H1 perturbations, regenerate 400
cells from each one's own real DE table (emission.regenerate_from_real) across
a small grid of dispersion_scale candidates, recompute DE on the regenerated
cells with the scorer's own routine (compute_de, same VCC2026_DE constants as
phase4_precompute.py), and score each candidate against two things the real
DE table already tells us: how many genes SHOULD be significant (n_real) and
which DIRECTION they moved. Deliberately a single global scalar, not a
per-gene/per-bucket table -- Idea 4's own kill criterion only asks whether a
calibrated emitter closes the gap enough to be worth anything; escalate to a
richer calibration only if the kill-check says the gap is real and this scalar
does not close it.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vccjudge.emission import regenerate_from_real  # noqa: E402
from vccjudge.regime import NON_TARGETING, group_rng  # noqa: E402

# Same DE contract as phase4_precompute.py / phase3_acceptance.py, so the DE
# table this script computes on regenerated cells is directly comparable to
# the real reference_de it was calibrated against.
VCC2026_DE = dict(backend="pdex", mean_calc="arithmetic", epsilon=1e-9,
                  input_type="counts", target_sum=1e6, clip_value=None,
                  filter_gene_min_cpm_cell=5.0, fdr_scope="per_pert")
P_ADJ = 0.05
CANDIDATES = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0]


def de_summary(adata, groupby="perturbation") -> tuple[dict, dict]:
    """Per-perturbation {n significant genes} and {gene -> log2fc} (ALL tested genes,
    not just significant ones, so magnitude can be compared even where the
    regenerated cells failed to cross the significance threshold)."""
    from cell_eval2.de_compute import compute_de
    df = compute_de(adata, groupby=groupby, reference=NON_TARGETING, **VCC2026_DE)
    d = df.to_pandas() if hasattr(df, "to_pandas") else df
    n = d[d["p_adj"] < P_ADJ].groupby("target").size().to_dict()
    lfc = {t: dict(zip(g["feature"], g["log2_fold_change"])) for t, g in d.groupby("target")}
    return n, lfc


def main() -> int:
    import anndata as ad
    from scipy import sparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--rendered", default="data/regime/vcc2025_h1/H1__seed0.h5ad")
    ap.add_argument("--de-table", default="artifacts/reference/reference_de__vcc2025_h1__H1.parquet")
    ap.add_argument("--n-perts", type=int, default=12)
    ap.add_argument("--ntc", type=int, default=6000)
    ap.add_argument("--out", default="artifacts/reference/emission_calibration.json")
    a = ap.parse_args()

    de_table = pd.read_parquet(a.de_table)
    real_nreal = de_table[de_table.p_adj < P_ADJ].groupby("target").size().to_dict()
    real_lfc = {t: dict(zip(g["feature"], g["log2_fold_change"]))
                for t, g in de_table[de_table.p_adj < P_ADJ].groupby("target")}

    adata = ad.read_h5ad(a.rendered)
    axis = list(map(str, adata.var_names))
    obs = adata.obs["perturbation"].astype(str).to_numpy()
    vc = pd.Series(obs).value_counts()
    full = sorted(p for p in vc.index if p != NON_TARGETING and vc[p] == 400 and p in real_nreal)
    rng0 = group_rng("vcc2025_h1", "H1", "__idea4_calibrate__", 0)
    chosen = sorted(rng0.choice(full, size=min(a.n_perts, len(full)), replace=False).tolist())
    print(f"{len(full)} eligible full-size perturbations; calibrating on {len(chosen)}: {chosen}")

    ntc_rows = np.flatnonzero(obs == NON_TARGETING)
    ntc_rows = np.sort(rng0.choice(ntc_rows, min(a.ntc, ntc_rows.size), replace=False))
    ctrl_block = adata.X.tocsr()[ntc_rows]

    # One real control sample per perturbation, fixed across candidates so the
    # comparison isolates dispersion_scale, not which cells got drawn.
    ctrl_pool = adata.X.tocsr()[np.flatnonzero(obs == NON_TARGETING)]
    pert_controls = {}
    for p in chosen:
        r = group_rng("vcc2025_h1", "H1", f"__idea4_calib_ctrl__{p}", 0)
        pick = np.sort(r.choice(ctrl_pool.shape[0], 400, replace=False))
        pert_controls[p] = ctrl_pool[pick]

    results = []
    for ci, scale in enumerate(CANDIDATES):
        t0 = time.time()
        blocks, labels = [], []
        for p in chosen:
            rng = group_rng("vcc2025_h1", "H1", f"__idea4_calib__{p}", ci)
            regen = regenerate_from_real(de_table, p, axis, pert_controls[p], rng,
                                          dispersion_scale=scale)
            blocks.append(regen.tocsr()); labels += [p] * regen.shape[0]
        X = sparse.vstack(blocks + [ctrl_block]).tocsr()
        X.eliminate_zeros()
        cand_obs = pd.DataFrame({"perturbation": labels + [NON_TARGETING] * ctrl_block.shape[0]})
        cand = ad.AnnData(X=X.astype(np.float32), obs=cand_obs,
                          var=pd.DataFrame(index=pd.Index(axis)))
        n_pred, pred_lfc = de_summary(cand)

        nreal_err, sign_agree, mag_err = [], [], []
        for p in chosen:
            nr, npd = real_nreal[p], n_pred.get(p, 0)
            nreal_err.append(abs(npd - nr) / max(nr, 1))
            real_p, pred_p = real_lfc.get(p, {}), pred_lfc.get(p, {})
            for g, rl in real_p.items():
                pl = pred_p.get(g, 0.0)
                sign_agree.append(1.0 if np.sign(pl) == np.sign(rl) else 0.0)
                # nmae's own asymmetric floor punishes overshoot harder than
                # undershoot (HANDOFF/appendix: shrinkage toward baseline is
                # the safe default) -- mirror that here so calibration doesn't
                # pick a dispersion level that wins on count/sign but wrecks
                # magnitude accuracy for genes with strong, confident evidence.
                rel = (pl - rl) / (abs(rl) + 0.5)
                mag_err.append(abs(rel) * (1.5 if abs(pl) > abs(rl) else 1.0))
        mean_nreal_err = float(np.mean(nreal_err))
        mean_sign_agree = float(np.mean(sign_agree)) if sign_agree else float("nan")
        mean_mag_err = float(np.mean(mag_err)) if mag_err else float("nan")
        loss = (mean_nreal_err + (1.0 - mean_sign_agree if sign_agree else 1.0)
                + (mean_mag_err if mag_err else 0.0))
        results.append(dict(dispersion_scale=scale, mean_nreal_rel_err=mean_nreal_err,
                             mean_sign_agreement=mean_sign_agree, mean_mag_rel_err=mean_mag_err,
                             n_genes_checked=len(mag_err), loss=loss))
        print(f"  scale={scale:<5} nreal_err={mean_nreal_err:.3f} "
              f"sign_agree={mean_sign_agree:.3f} mag_err={mean_mag_err:.3f} "
              f"loss={loss:.3f}  ({time.time()-t0:.0f}s)")

    best = min(results, key=lambda r: r["loss"])
    print(f"\nBEST: dispersion_scale={best['dispersion_scale']} (loss={best['loss']:.3f})")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"chosen_perturbations": chosen, "candidates": results,
               "best_dispersion_scale": best["dispersion_scale"]},
              open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
