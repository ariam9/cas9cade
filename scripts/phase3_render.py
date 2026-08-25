#!/usr/bin/env python
"""Phase 3: render a harmonized line into the challenge measurement regime.

    python scripts/phase3_render.py --dataset vcc2025_h1 --cell-line H1 --seed 0

Writes `data/regime/<dataset>/<cell_line>__seed<k>.h5ad` (gzipped) and skips the
work entirely if that file already exists -- a finished render is never redone,
which keeps repeat runs off the SSD.

I/O shape, deliberately: row indices are chosen up front and sorted, so the
matrix is read in ONE sequential pass with no random seeks; output is written
once, streamed in blocks, compressed. For H1 that is ~15 GB read (reads do not
wear flash) and ~0.9 GB written.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vccjudge.regime import (  # noqa: E402
    CHALLENGE_MEDIAN_UMI, group_rng, plan_render, thin_counts, thinning_probability,
)

BLOCK = 4096


def estimate_median_umi(X, rows: np.ndarray, rng, n_sample: int = 3000) -> float:
    """Median UMI of the picked cells, from a sample.

    The median is robust and 3,000 draws pin it tightly, so this reads ~4% of the
    data instead of all of it. The achieved median is measured exactly during the
    write pass anyway, so an error here shows up immediately rather than silently.
    """
    take = rows if rows.size <= n_sample else np.sort(rng.choice(rows, n_sample, replace=False))
    tot = []
    for i in range(0, take.size, BLOCK):
        blk = take[i : i + BLOCK]
        tot.append(np.asarray(X[blk].sum(axis=1)).ravel())
    return float(np.median(np.concatenate(tot)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="vcc2025_h1")
    ap.add_argument("--cell-line", default="H1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--in-dir", default="data/harmonized")
    ap.add_argument("--out-dir", default="data/regime")
    ap.add_argument("--target-umi", type=float, default=CHALLENGE_MEDIAN_UMI)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit-groups", type=int, default=None, help="smoke test with N groups")
    a = ap.parse_args()

    import anndata as ad
    import h5py
    from anndata.io import sparse_dataset, write_elem

    src = Path(a.in_dir) / a.dataset / f"{a.cell_line}.h5ad"
    if not src.exists():
        sys.exit(f"FATAL: {src} not found — run scripts/phase2_harmonize.py first")
    out = Path(a.out_dir) / a.dataset / f"{a.cell_line}__seed{a.seed}.h5ad"
    if out.exists() and not a.force and not a.limit_groups:
        print(f"{out} exists — nothing to do (--force to re-render)")
        return 0

    adata = ad.read_h5ad(src, backed="r")
    try:
        perts = adata.obs["perturbation"].astype(str).to_numpy()
        plan = plan_render(perts, a.dataset, a.cell_line, a.seed)
        if a.limit_groups:
            plan.groups = plan.groups[: a.limit_groups]
        rows = plan.order
        print(f"source {src}: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
        s0 = plan.summary()
        print(f"plan: {s0['n_groups']} groups -> {s0['n_cells']:,} cells "
              f"({100*s0['n_cells']/adata.n_obs:.0f}% of source), {s0['n_short_groups']} short")

        rng = group_rng(a.dataset, a.cell_line, "__depth__", a.seed)
        med = estimate_median_umi(adata.X, rows, rng)
        p, deficit = thinning_probability(med, a.target_umi)
        plan.thin_p, plan.source_median_umi, plan.depth_deficit = p, med, deficit
        print(f"depth: source median ~{med:,.0f} UMI -> target {a.target_umi:,.0f} "
              f"| thin p={p:.4f}{'  ⚠ DEFICIT: cannot thin upward' if deficit else ''}")

        pos = {int(r): k for k, r in enumerate(rows)}
        obs = pd.DataFrame({
            "perturbation": [g.perturbation for g in plan.groups for _ in range(g.n_picked)],
            "cell_line": a.cell_line, "dataset": a.dataset,
            "source_row": np.concatenate([g.picked for g in plan.groups]),
        })
        # obs rows must line up with the ORDER cells are written (sorted by source row)
        obs = obs.sort_values("source_row", kind="stable").reset_index(drop=True)

        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".h5ad.partial")
        t0, nnz_out, totals = time.time(), 0, []
        with h5py.File(tmp, "w") as f:
            f.attrs["encoding-type"] = "anndata"; f.attrs["encoding-version"] = "0.1.0"
            write_elem(f, "obs", obs)
            write_elem(f, "var", adata.var.copy())
            write_elem(f, "X", sparse.csr_matrix((0, adata.n_vars), dtype=np.float32),
                       dataset_kwargs={"compression": "gzip"})
            Xds = sparse_dataset(f["X"])
            for i in range(0, rows.size, BLOCK):
                blk = rows[i : i + BLOCK]
                thinned = thin_counts(adata.X[blk], p, rng)
                Xds.append(thinned)
                nnz_out += int(thinned.nnz)
                totals.append(np.asarray(thinned.sum(axis=1)).ravel())
                if (i // BLOCK) % 5 == 0:
                    done = min(i + BLOCK, rows.size)
                    el = time.time() - t0
                    print(f"  {done:>7,}/{rows.size:,}  nnz {nnz_out:>12,}  {el:5.0f}s", flush=True)
        tmp.replace(out)

        tot = np.concatenate(totals)
        achieved = float(np.median(tot))
        # Re-summarise AFTER the depth fields are set: the first call happens before
        # thinning is decided, so reusing it recorded thin_p=1.0 / median=0.0 into the
        # provenance of an artifact that had in fact been thinned by p=0.371.
        meta = {**plan.summary(), "achieved_median_umi": achieved, "target_umi": a.target_umi,
                "nnz_out": nnz_out, "seed": a.seed,
                "short_groups": [g.perturbation for g in plan.groups if g.short]}
        assert meta["thin_p"] == p and meta["source_median_umi"] == med, "stale summary"
        out.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
        print(f"\nwrote {out} ({out.stat().st_size/2**30:.2f} GB)")
        print(f"  {len(obs):,} cells | nnz {nnz_out:,} | median UMI {achieved:,.0f} "
              f"(target {a.target_umi:,.0f}, off by {100*abs(achieved-a.target_umi)/a.target_umi:.1f}%)")
        if meta["n_short_groups"]:
            print(f"  ⚠ {meta['n_short_groups']} group(s) rendered short — lower DE power, flagged in the json")
        return 0
    finally:
        if adata.isbacked:
            adata.file.close()


if __name__ == "__main__":
    raise SystemExit(main())
