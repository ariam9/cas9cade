#!/usr/bin/env python
"""Phase 4: precompute reference effects and in-regime DE.

    python scripts/phase4_precompute.py --dataset vcc2025_h1 --cell-line H1

Two artifacts, computed from DIFFERENT sources on purpose -- this is the
Role A / Role B split CLAUDE.md now mandates:

  reference_effects.parquet   FULL DEPTH, from data/harmonized/
      A pseudobulk delta is a mean vector, not a significance call. As the
      transfer signal a model learns from, it wants every cell and every read:
      H1 at native depth carries 7.05x the counts of its rendered version, so
      ~2.7x smaller standard error on the estimate. Downsampling here would be
      throwing away precision for nothing.

  nreal_table.parquet + de/   IN REGIME, from data/regime/
      How many genes test significant is a function of cell count and depth,
      not biology alone -- measured: nreal falls 630 -> 110 between H1's native
      regime and the challenge's. Anything deciding significance must be
      rendered first, and must use the scorer's own routine so reference DE and
      scored DE cannot drift.

The delta pass is one sequential read with per-group accumulators (151 groups x
18,533 genes = 22 MB), so it streams a 15 GB matrix in constant memory.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vccjudge.regime import NON_TARGETING  # noqa: E402

# From EvalConfig.from_preset("vcc2026") — the scorer's own DE contract.
# backend is the ONE field not taken from the preset (which says "auto"): pinning
# it makes the engine explicit and reproducible. pdex is ~10x faster than scanpy
# and verified to give IDENTICAL nreal on a matched slice (8/8 exact), so this is
# speed, not a different answer. cell-eval2 warns engines can disagree -- hence
# the check rather than the assumption.
VCC2026_DE = dict(backend="pdex", mean_calc="arithmetic", epsilon=1e-9,
                  input_type="counts", target_sum=1e6, clip_value=None,
                  filter_gene_min_cpm_cell=5.0, fdr_scope="per_pert")
P_ADJ = 0.05
BLOCK = 4096


def pseudobulk_cpm(adata, groups: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Mean CPM profile per group, streamed. Full depth, nothing discarded."""
    names = sorted(set(groups.tolist()))
    gidx = {g: i for i, g in enumerate(names)}
    sums = np.zeros((len(names), adata.n_vars), dtype=np.float64)
    counts = np.zeros(len(names), dtype=np.int64)
    codes = np.array([gidx[g] for g in groups])
    for start in range(0, adata.n_obs, BLOCK):
        stop = min(start + BLOCK, adata.n_obs)
        X = adata.X[start:stop]
        A = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        c = codes[start:stop]
        np.add.at(sums, c, A)
        np.add.at(counts, c, 1)
        if (start // BLOCK) % 20 == 0:
            print(f"    {stop:>7,}/{adata.n_obs:,}", flush=True)
    lib = sums.sum(axis=1, keepdims=True)
    cpm = np.divide(sums, lib, out=np.zeros_like(sums), where=lib > 0) * 1e6
    return pd.DataFrame(cpm, index=names, columns=list(map(str, adata.var_names))), \
           dict(zip(names, counts.tolist()))


def main() -> int:
    import anndata as ad

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="vcc2025_h1")
    ap.add_argument("--cell-line", default="H1")
    ap.add_argument("--harmonized", default=None, help="full-depth source for the deltas")
    ap.add_argument("--rendered", default=None, help="in-regime source for DE")
    ap.add_argument("--out", default="artifacts/reference")
    ap.add_argument("--shard-size", type=int, default=20,
                    help="perturbations per DE shard; each also carries the full control arm")
    ap.add_argument("--skip-de", action="store_true", help="deltas only (DE is the slow half)")
    a = ap.parse_args()

    har = Path(a.harmonized or f"data/harmonized/{a.dataset}/{a.cell_line}.h5ad")
    ren = Path(a.rendered or f"data/regime/{a.dataset}/{a.cell_line}__seed0.h5ad")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.dataset}__{a.cell_line}"

    # ---- Role A: full-depth deltas ---------------------------------------
    if har.exists():
        print(f"[deltas] FULL DEPTH from {har}", flush=True)
        adata = ad.read_h5ad(har, backed="r")
        try:
            groups = adata.obs["perturbation"].astype(str).to_numpy()
            t0 = time.time()
            cpm, n_cells = pseudobulk_cpm(adata, groups)
        finally:
            if adata.isbacked:
                adata.file.close()
        if NON_TARGETING not in cpm.index:
            sys.exit(f"FATAL: no '{NON_TARGETING}' group in {har}")
        delta = cpm.drop(index=NON_TARGETING).sub(cpm.loc[NON_TARGETING], axis=1)
        delta.insert(0, "n_cells", [n_cells[p] for p in delta.index])
        delta.index.name = "perturbation"
        dpath = out / f"reference_effects__{tag}.parquet"
        delta.reset_index().to_parquet(dpath, index=False)
        print(f"  wrote {dpath}: {delta.shape[0]} perturbations x {delta.shape[1]-1} genes "
              f"({time.time()-t0:.0f}s)")
    else:
        print(f"[deltas] SKIP — {har} not found")

    # ---- Role B: in-regime DE -------------------------------------------
    if a.skip_de:
        print("[DE] skipped (--skip-de)")
        return 0
    if not ren.exists():
        sys.exit(f"FATAL: {ren} not found — run scripts/phase3_render.py first")
    print(f"\n[DE] IN REGIME from {ren}", flush=True)
    from cell_eval2.de_compute import compute_de

    # Sharded, per PLAN.md ("per (cell line, perturbation) shards with per-shard
    # checkpoints, so a killed batch job resumes"). Not optional: DE over all
    # 151 groups at once was OOM-killed at a 14 GB ceiling, because the control
    # arm (18,400 cells) is carried into every comparison and scanpy densifies.
    # A shard is <shard_size> perturbations against that same control arm.
    ckpt = out / f"de_shards__{tag}"; ckpt.mkdir(parents=True, exist_ok=True)
    src = ad.read_h5ad(ren, backed="r")
    obs = src.obs["perturbation"].astype(str).to_numpy()
    ntc_rows = np.flatnonzero(obs == NON_TARGETING)
    perts = sorted(set(obs.tolist()) - {NON_TARGETING})
    shards = [perts[i:i + a.shard_size] for i in range(0, len(perts), a.shard_size)]
    print(f"  {len(perts)} perturbations in {len(shards)} shards of <={a.shard_size}"
          f" (+{ntc_rows.size:,} control cells each)", flush=True)

    t0 = time.time()
    for si, group in enumerate(shards):
        cp = ckpt / f"shard{si:04d}.parquet"
        if cp.exists():
            print(f"  shard {si+1}/{len(shards)}: cached", flush=True)
            continue
        rows = np.sort(np.concatenate([ntc_rows] +
                       [np.flatnonzero(obs == p) for p in group]))
        sub = src[rows].to_memory()
        part = compute_de(sub, groupby="perturbation", reference=NON_TARGETING, **VCC2026_DE)
        pdf = part.to_pandas() if hasattr(part, "to_pandas") else part
        pdf.to_parquet(cp, index=False)
        del sub, part, pdf
        print(f"  shard {si+1}/{len(shards)}: {len(group)} perts  {time.time()-t0:5.0f}s", flush=True)
    if src.isbacked:
        src.file.close()

    d = pd.concat([pd.read_parquet(f) for f in sorted(ckpt.glob("shard*.parquet"))],
                  ignore_index=True)
    padj = next(c for c in d.columns if "adj" in c.lower() or c.lower() in ("fdr", "qval"))
    tgt = next(c for c in d.columns if c.lower() in ("target", "target_gene", "perturbation", "group"))
    de_path = out / f"reference_de__{tag}.parquet"
    d.to_parquet(de_path, index=False)
    nreal = (d[d[padj] < P_ADJ].groupby(tgt).size()
             .rename("nreal").reset_index().rename(columns={tgt: "perturbation"}))
    allp = pd.DataFrame({"perturbation": sorted(d[tgt].astype(str).unique())})
    nreal = allp.merge(nreal, on="perturbation", how="left").fillna({"nreal": 0})
    nreal["nreal"] = nreal["nreal"].astype(int)
    nreal["dataset"], nreal["cell_line"] = a.dataset, a.cell_line
    npath = out / f"nreal_table__{tag}.parquet"
    nreal.to_parquet(npath, index=False)
    print(f"  wrote {de_path} ({len(d):,} rows) and {npath} in {time.time()-t0:.0f}s")
    q = nreal["nreal"].describe()
    print(f"  nreal: median {nreal.nreal.median():.0f}  mean {q['mean']:.0f}  "
          f"min {q['min']:.0f}  max {q['max']:.0f}  zero-response {int((nreal.nreal==0).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
