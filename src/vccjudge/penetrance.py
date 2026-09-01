"""Idea 3 (vcc2026-architecture-ideas.pdf) -- penetrance: a perturbation acts
on a fraction of cells.

Every emitter in this repo (`apply_effect`, and `emission.emit_from_belief`)
treats the 400 emitted cells as draws from ONE distribution. Real CRISPRi
knockdown is incomplete and cell-cycle/state-dependent: a fraction pi of cells
actually enter the perturbed state, the rest stay close to control. This
module estimates pi from the one signal that needs almost no biology: the
target gene's OWN per-cell expression among its perturbed cells, which a
successful knockdown should visibly suppress in some fraction of cells and not
others.

Step 1 only (see the phase 7 plan): fit pi per perturbation on real reference
data and check the distribution is non-degenerate. Step 2 (predicting pi from
basal state on a held-out line) is blocked on the K562 regime rebuild and is
not implemented here yet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from .regime import CHALLENGE_CELLS_PER_PERT, NON_TARGETING

RESULT_COLUMNS = ["perturbation", "pi_hat", "control_component_mean",
                   "responder_component_mean", "loglik", "n_cells_used",
                   "converged", "reason"]


def _log1p_cpm_column(X_csc, gene_idx: int, lib: np.ndarray) -> np.ndarray:
    col = np.asarray(X_csc[:, gene_idx].todense()).ravel()
    return np.log1p(col / lib * 1e6)


def _fit_from_values(ctrl_vals: np.ndarray, pert_vals: np.ndarray,
                      min_cells: int, random_state: int) -> dict:
    if pert_vals.size < min_cells:
        return dict(pi_hat=float("nan"),
                    control_component_mean=float(ctrl_vals.mean()) if ctrl_vals.size else float("nan"),
                    responder_component_mean=float("nan"), loglik=float("nan"),
                    n_cells_used=int(pert_vals.size), converged=False, reason="too_few_cells")

    from sklearn.mixture import GaussianMixture

    ctrl_mean = float(ctrl_vals.mean())
    x = pert_vals.reshape(-1, 1)

    # A 2-component GaussianMixture ALWAYS finds some split, even fit to a
    # genuinely unimodal population -- verified: a perturbation with no real
    # effect and one with (near-)complete penetrance both got fitted into two
    # near-identical, overlapping components, producing a spurious mid-range
    # pi_hat for both instead of the correct ~0 and ~1. BIC against a
    # 1-component fit decides whether a real two-population split is
    # justified at all before any weight is trusted as pi.
    gm1 = GaussianMixture(n_components=1, random_state=random_state, reg_covar=1e-4).fit(x)
    # Warm-start the "responder" guess at the perturbed population's own low
    # tail: a real knockdown suppresses the target gene, so a lower log1p(CPM)
    # than control is the biologically sensible starting point.
    means_init = np.array([[ctrl_mean], [float(np.percentile(pert_vals, 20))]])
    gm2 = GaussianMixture(n_components=2, means_init=means_init, random_state=random_state,
                          reg_covar=1e-4, n_init=1).fit(x)

    if gm2.bic(x) >= gm1.bic(x):
        # Unimodal: classify by whether that single mode is detectably
        # shifted from control, rather than trusting an unjustified 2-way split.
        single_mean = float(gm1.means_.ravel()[0])
        ctrl_std = float(ctrl_vals.std()) or 1.0
        shifted = abs(single_mean - ctrl_mean) > ctrl_std
        return dict(pi_hat=(1.0 if shifted else 0.0),
                    control_component_mean=(float("nan") if shifted else single_mean),
                    responder_component_mean=(single_mean if shifted else float("nan")),
                    loglik=float(gm1.score(x) * pert_vals.size), n_cells_used=int(pert_vals.size),
                    converged=bool(gm1.converged_),
                    reason=("unimodal_shifted" if shifted else "unimodal_no_effect"))

    # The component is identified AFTER fitting, by proximity to the
    # empirical control mean, not by fit order (sklearn has no clean API to
    # hard-pin one component during EM).
    means, weights = gm2.means_.ravel(), gm2.weights_.ravel()
    ctrl_idx = int(np.argmin(np.abs(means - ctrl_mean)))
    resp_idx = 1 - ctrl_idx
    return dict(pi_hat=float(weights[resp_idx]), control_component_mean=float(means[ctrl_idx]),
                responder_component_mean=float(means[resp_idx]),
                loglik=float(gm2.score(x) * pert_vals.size),
                n_cells_used=int(pert_vals.size), converged=bool(gm2.converged_), reason="ok")


def fit_penetrance(target_gene: str, adata_rendered, ntc_mask: np.ndarray, pert_mask: np.ndarray,
                    min_cells: int = 20, random_state: int = 0) -> dict:
    """Fit pi for ONE perturbation from its target gene's own per-cell log1p(CPM).

    Standalone convenience wrapper matching the plan's declared signature --
    `fit_penetrance_all` does the equivalent work for every perturbation at
    once without repeating the CSC conversion / library-size pass this
    function redoes on each call.
    """
    var_names = np.asarray(adata_rendered.var_names)
    hit = np.flatnonzero(var_names == target_gene)
    if hit.size == 0:
        return dict(pi_hat=float("nan"), control_component_mean=float("nan"),
                    responder_component_mean=float("nan"), loglik=float("nan"),
                    n_cells_used=0, converged=False, reason="gene_not_on_axis")
    X = adata_rendered.X
    Xcsc = X.tocsc() if hasattr(X, "tocsc") else X
    lib = np.asarray(X.sum(axis=1)).ravel()
    lib[lib == 0] = 1.0
    log1p_cpm = _log1p_cpm_column(Xcsc, int(hit[0]), lib)
    return _fit_from_values(log1p_cpm[ntc_mask], log1p_cpm[pert_mask], min_cells, random_state)


def fit_penetrance_all(adata_rendered, perturbation_col: str = "perturbation",
                        min_cells: int = 20, random_state: int = 0) -> pd.DataFrame:
    """Fit pi for every perturbation in `adata_rendered` in one pass."""
    obs = adata_rendered.obs[perturbation_col].astype(str).to_numpy()
    ntc_mask = obs == NON_TARGETING
    perts = sorted(set(obs.tolist()) - {NON_TARGETING})

    var_names = np.asarray(adata_rendered.var_names)
    Xcsc = adata_rendered.X.tocsc()
    lib = np.asarray(adata_rendered.X.sum(axis=1)).ravel()
    lib[lib == 0] = 1.0

    rows = []
    for p in perts:
        pert_mask = obs == p
        hit = np.flatnonzero(var_names == p)
        if hit.size == 0:
            r = dict(pi_hat=float("nan"), control_component_mean=float("nan"),
                      responder_component_mean=float("nan"), loglik=float("nan"),
                      n_cells_used=int(pert_mask.sum()), converged=False, reason="gene_not_on_axis")
        else:
            log1p_cpm = _log1p_cpm_column(Xcsc, int(hit[0]), lib)
            r = _fit_from_values(log1p_cpm[ntc_mask], log1p_cpm[pert_mask], min_cells, random_state)
        r["perturbation"] = p
        rows.append(r)
    return pd.DataFrame(rows)[RESULT_COLUMNS]


# ---------------------------------------------------------------------------
# Step 2: predict pi from basal state, check transfer on a held-out line.
# ---------------------------------------------------------------------------

def basal_features_for_genes(adata_rendered, genes, perturbation_col: str = "perturbation") -> pd.DataFrame:
    """Per-gene basal covariates computable from CONTROLS ALONE: log1p(mean
    CPM) and the coefficient of variation across NTC cells. These are exactly
    what a held-out line can supply -- no perturbed-cell data, so this is the
    leakage-safe input side of the basal-state -> pi model.
    """
    obs = adata_rendered.obs[perturbation_col].astype(str).to_numpy()
    ntc_mask = obs == NON_TARGETING
    var_names = np.asarray(adata_rendered.var_names)
    Xcsc = adata_rendered.X.tocsc()
    lib = np.asarray(adata_rendered.X.sum(axis=1)).ravel()
    lib[lib == 0] = 1.0
    ntc_lib = lib[ntc_mask]

    rows = []
    for g in genes:
        hit = np.flatnonzero(var_names == g)
        if hit.size == 0:
            rows.append(dict(gene=g, log_mean_cpm=float("nan"), cv=float("nan")))
            continue
        col = np.asarray(Xcsc[:, int(hit[0])].todense()).ravel()[ntc_mask]
        cpm = col / ntc_lib * 1e6
        mean_cpm = float(cpm.mean())
        cv = float(cpm.std() / mean_cpm) if mean_cpm > 0 else 0.0
        rows.append(dict(gene=g, log_mean_cpm=float(np.log1p(mean_cpm)), cv=cv))
    return pd.DataFrame(rows)


def _logit(p: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _inv_logit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_basal_pi_model(pi_table: pd.DataFrame, basal_features: pd.DataFrame) -> dict:
    """Logistic-on-logit regression of pi_hat on basal covariates (log mean
    CPM, coefficient of variation) -- deliberately simple (closed-form OLS in
    logit space, two features), since the fitting corpus is one line's ~150
    points. Returns plain floats so the model is a few-KB JSON, not a pickle.
    """
    merged = pi_table.merge(basal_features, left_on="perturbation", right_on="gene")
    merged = merged.dropna(subset=["pi_hat", "log_mean_cpm", "cv"])
    y = _logit(merged["pi_hat"].to_numpy(dtype=np.float64))
    X = np.column_stack([np.ones(len(merged)), merged["log_mean_cpm"].to_numpy(dtype=np.float64),
                          merged["cv"].to_numpy(dtype=np.float64)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return dict(intercept=float(coef[0]), coef_log_mean_cpm=float(coef[1]), coef_cv=float(coef[2]),
                n_fit=int(len(merged)))


def predict_pi(model: dict, basal_features: pd.DataFrame) -> pd.Series:
    """Apply a `fit_basal_pi_model` model to a (possibly held-out) line's own
    basal covariates."""
    z = (model["intercept"] + model["coef_log_mean_cpm"] * basal_features["log_mean_cpm"]
         + model["coef_cv"] * basal_features["cv"])
    pi = _inv_logit(z.to_numpy(dtype=np.float64))
    return pd.Series(pi, index=basal_features["gene"].to_numpy(), name="predicted_pi")


def penetrance_transfer_check(
    adata_rendered, reference_effects: pd.DataFrame, nreal_table: pd.DataFrame,
    basal_model: dict, min_nreal_strong: int = 10, shard_size: int = 40, seed: int = 0,
) -> tuple[dict, pd.DataFrame]:
    """Idea 3 step 2's real check, on a (possibly held-out) line: does a
    basal-state-predicted-pi decoder track n_real better than a
    fixed-dispersion (pi=1) one, given the SAME magnitude estimate for both?

    `reference_effects`/`nreal_table` are THIS line's own (e.g. K562's own),
    not borrowed cross-line -- this isolates whether modeling penetrance helps
    n_pred, independent of the separate cross-line magnitude-transfer question
    Idea 5 already covers. Sharded (`shard_size` perturbations/compute_de call
    + one shared control arm) because an unsharded DE pass over hundreds of
    perturbations already OOM'd once in this repo (phase4_precompute.py).

    Returns (summary dict, per-perturbation DataFrame).
    """
    from cell_eval2.de_compute import compute_de
    from scipy import sparse as sp

    from .predictors import effect_ratios
    from .regime import group_rng

    VCC2026_DE = dict(backend="pdex", mean_calc="arithmetic", epsilon=1e-9,
                      input_type="counts", target_sum=1e6, clip_value=None,
                      filter_gene_min_cpm_cell=5.0, fdr_scope="per_pert")
    dataset, cell_line = (str(adata_rendered.obs["dataset"].iloc[0]),
                          str(adata_rendered.obs["cell_line"].iloc[0]))

    obs = adata_rendered.obs["perturbation"].astype(str).to_numpy()
    axis = list(map(str, adata_rendered.var_names))
    ntc_rows = np.flatnonzero(obs == NON_TARGETING)
    ctrl_pool = adata_rendered.X.tocsr()[ntc_rows]
    ctrl_sum = np.asarray(ctrl_pool.sum(axis=0)).ravel()
    ctrl_cpm = ctrl_sum * (1e6 / max(float(ctrl_sum.sum()), 1.0))

    eff = reference_effects.set_index("perturbation")
    eff = eff.drop(columns=[c for c in ("n_cells",) if c in eff.columns]).reindex(columns=axis).fillna(0.0)
    nreal_map = nreal_table.set_index("perturbation")["nreal"].to_dict()

    basal = basal_features_for_genes(adata_rendered, [p for p in eff.index if p in set(obs.tolist())])
    predicted_pi = predict_pi(basal_model, basal)
    perts = [p for p in eff.index if p in predicted_pi.index and p in set(obs.tolist())]
    n_sample = min(CHALLENGE_CELLS_PER_PERT, ctrl_pool.shape[0])

    r0 = group_rng(dataset, cell_line, "__penetrance_transfer_ntc__", seed)
    ctrl_shared = ctrl_pool[np.sort(r0.choice(ctrl_pool.shape[0], min(6000, ctrl_pool.shape[0]),
                                              replace=False))]

    def _n_pred_sharded(pi_map: dict) -> dict:
        out: dict = {}
        for i in range(0, len(perts), shard_size):
            group = perts[i:i + shard_size]
            blocks, labels = [], []
            for p in group:
                rp = group_rng(dataset, cell_line, f"__penetrance_transfer__{p}", seed)
                pick = np.sort(rp.choice(ctrl_pool.shape[0], n_sample, replace=False))
                pert_cpm = ctrl_cpm + eff.loc[p].to_numpy(dtype=np.float64)
                ratios = effect_ratios(np.clip(pert_cpm, 0, None), ctrl_cpm).astype(np.float32)
                pi_val = float(np.clip(pi_map.get(p, 1.0), 0.0, 1.0))
                blocks.append(emit_with_penetrance(ctrl_pool[pick], ratios, pi_val, rp))
                labels += [p] * n_sample
            X = sp.vstack(blocks + [ctrl_shared]).tocsr()
            X.eliminate_zeros()
            obs_df = pd.DataFrame({"perturbation": labels + [NON_TARGETING] * ctrl_shared.shape[0]})
            import anndata as ad
            sub = ad.AnnData(X=X.astype(np.float32), obs=obs_df, var=pd.DataFrame(index=pd.Index(axis)))
            sub.obs_names_make_unique()
            d = compute_de(sub, groupby="perturbation", reference=NON_TARGETING, **VCC2026_DE)
            d = d.to_pandas() if hasattr(d, "to_pandas") else d
            out.update(d[d.p_adj < 0.05].groupby("target").size().to_dict())
            del blocks, X, sub, d
        return out

    n_fixed = _n_pred_sharded({p: 1.0 for p in perts})
    n_mixed = _n_pred_sharded(predicted_pi.to_dict())

    rows = [dict(perturbation=p, n_real=int(nreal_map.get(p, 0)),
                 n_pred_fixed=int(n_fixed.get(p, 0)), n_pred_mixed=int(n_mixed.get(p, 0)),
                 predicted_pi=float(predicted_pi.get(p, 1.0))) for p in perts]
    df = pd.DataFrame(rows)
    df["err_fixed"] = (df.n_pred_fixed - df.n_real).abs()
    df["err_mixed"] = (df.n_pred_mixed - df.n_real).abs()
    strong = df[df.n_real >= min_nreal_strong]

    mae_fixed_all, mae_mixed_all = float(df.err_fixed.mean()), float(df.err_mixed.mean())
    mae_fixed_strong = float(strong.err_fixed.mean()) if len(strong) else float("nan")
    mae_mixed_strong = float(strong.err_mixed.mean()) if len(strong) else float("nan")
    beats = np.isfinite(mae_mixed_strong) and np.isfinite(mae_fixed_strong) and mae_mixed_strong < mae_fixed_strong
    verdict = (f"CONTINUE -- mixture beats fixed-dispersion on the nreal>={min_nreal_strong} subset"
               if beats else
               f"KILL -- mixture does not beat fixed-dispersion on the nreal>={min_nreal_strong} subset")

    summary = dict(n_perturbations=len(df), n_strong=int(len(strong)), min_nreal_strong=min_nreal_strong,
                    mae_fixed_all=mae_fixed_all, mae_mixed_all=mae_mixed_all,
                    mae_fixed_strong=mae_fixed_strong, mae_mixed_strong=mae_mixed_strong, verdict=verdict)
    return summary, df


def emit_with_penetrance(control_block, ratios: np.ndarray, pi: float, rng: np.random.Generator):
    """cells ~ pi * D_responder + (1-pi) * D_control, literally: round(pi * n)
    cells receive `apply_effect(ratios)`, the rest stay untouched real control
    cells."""
    from .predictors import apply_effect

    X = control_block.tocsr()
    n = X.shape[0]
    n_resp = int(round(float(np.clip(pi, 0.0, 1.0)) * n))
    order = rng.permutation(n)
    resp_idx, ctrl_idx = np.sort(order[:n_resp]), np.sort(order[n_resp:])
    resp = (apply_effect(X[resp_idx], ratios, rng) if n_resp > 0
            else sparse.csr_matrix((0, X.shape[1]), dtype=np.float32))
    ctrl = X[ctrl_idx].astype(np.float32)
    out = sparse.vstack([resp, ctrl]).tocsr()
    out.eliminate_zeros()
    return out
