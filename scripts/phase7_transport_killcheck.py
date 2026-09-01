#!/usr/bin/env python
"""Idea 5 kill-check: does gene-specific weighting beat global similarity?

    .venv/bin/python scripts/phase7_transport_killcheck.py \
        --held-out vcc2025_h1/H1 --effects-from replogle2022/K562

Runs `delta_transfer`, `transport_transfer_global`, and `transport_transfer`
through phase6_harness.py with identical seeds and --all-perts, then diffs
score.csv's pds_cosine / de_wilcoxon_sig_jaccard between the gene-specific and
global-similarity arms -- the idea's own named metrics.

Kill criterion (verbatim from the doc): if gene-specific weighting does not
beat global weighting on pds and jac on held-out lines, drop it -- and that
result is itself informative (effect transfer governed by global cell state,
not local pathway state).

⚠️ Requires `{effects_from_dataset}_control_cpm__full_axis.parquet` and
`{effects_from_dataset}_coexpr_neighbors__panel.parquet` under --ref-dir for
the DONOR line -- not just its reference_effects. For a K562 donor, these do
not exist until the K562 Kaggle rebuild lands (see HANDOFF/the phase 7 plan);
this script will fail fast with a clear FATAL from phase6_harness.py rather
than silently skip if they're missing.
"""
from __future__ import annotations

import argparse, json, subprocess, sys
from pathlib import Path

import pandas as pd

PREDICTORS = ["delta_transfer", "transport_transfer_global", "transport_transfer"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--held-out", required=True)
    ap.add_argument("--effects-from", required=True)
    ap.add_argument("--regime-dir", default="data/regime")
    ap.add_argument("--ref-dir", default="artifacts/reference")
    ap.add_argument("--out", default="data/harness")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-ntc", type=int, default=None)
    ap.add_argument("--limit-perts", type=int, default=None,
                    help="passthrough to phase6_harness.py, for fast iteration only -- "
                         "the real kill-check needs --all-perts's full coverage")
    ap.add_argument("--ce2", default=".venv/bin/cell-eval2")
    a = ap.parse_args()

    scores = {}
    for predictor in PREDICTORS:
        cmd = [sys.executable, "scripts/phase6_harness.py",
               "--held-out", a.held_out, "--effects-from", a.effects_from,
               "--predictor", predictor, "--all-perts",
               "--regime-dir", a.regime_dir, "--ref-dir", a.ref_dir,
               "--out", a.out, "--seed", str(a.seed), "--ce2", a.ce2]
        if a.max_ntc:
            cmd += ["--max-ntc", str(a.max_ntc)]
        if a.limit_perts:
            cmd += ["--limit-perts", str(a.limit_perts)]
        print(f"\n$ {' '.join(cmd)}", flush=True)
        p = subprocess.run(cmd)
        if p.returncode:
            sys.exit(f"FATAL: {predictor} run failed ({p.returncode})")
        ho_ds, ho_line = a.held_out.split("/")
        result_path = Path(a.out) / f"{ho_ds}__{ho_line}__{predictor}" / "harness_result.json"
        scores[predictor] = json.loads(result_path.read_text())["scores"]

    df = pd.DataFrame(scores).T
    print("\n" + "=" * 78)
    print(f"  {a.effects_from} -> {a.held_out}")
    print("=" * 78)
    print(df.to_string(float_format=lambda v: f"{v:.4f}"))

    specific, glob = scores["transport_transfer"], scores["transport_transfer_global"]
    checks = {m: specific[m] - glob[m] for m in ("pds_cosine", "de_wilcoxon_sig_jaccard")
              if m in specific and m in glob}
    print("\nspecific - global, on the idea's own named metrics:")
    for m, d in checks.items():
        print(f"  {m:<30}{d:+.4f}")
    beats_global = all(d > 0 for d in checks.values())
    verdict = ("CONTINUE -- gene-specific weighting beats global on pds/jac"
               if beats_global else
               "KILL -- gene-specific weighting does not beat global; transfer is "
               "governed by global cell state, not local pathway state")
    print(f"\nVERDICT: {verdict}")

    out_path = Path(a.out) / f"phase7_transport_killcheck__{a.held_out.replace('/', '_')}.json"
    json.dump({"held_out": a.held_out, "effects_from": a.effects_from, "scores": scores,
               "specific_minus_global": checks, "verdict": verdict},
              open(out_path, "w"), indent=2)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
