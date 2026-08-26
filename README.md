# VCC 2026 — offline judge

Score a perturbation-prediction idea **locally**, on the same scale the Arc
Virtual Cell Challenge leaderboard uses, without spending a submission.

The competition gives you one number per submission, days apart. This harness
gives you the same six metrics in minutes, calibrated against the same two
anchors, so you can find out *why* something scores badly instead of only *that*
it did.

---

## Quickstart

```bash
git clone https://github.com/ariam9/cas9cade && cd cas9cade
uv venv --python 3.12 .venv && uv pip install -r requirements.txt
uv pip install -e . --no-deps          # required: the package lives under src/

.venv/bin/python scripts/phase0_acceptance.py     # must print: PASS — Phase 0 certified
```

If that prints `PASS`, your environment is correct. Everything else depends on it.

### Getting the data

Data is **not** in the repo (it is ~40 GB). Every source is listed with its URL
and checksum in `config/fetch_manifest.tsv`.

```bash
uv tool install vcc-cli                # the challenge CLI; the command is `vcc`
vcc login --token-stdin                # API key from the challenge portal
vcc datasets download controls -d data/bundle
cd data/bundle && unzip -o vcc_2026_controls.zip && cd ../..

.venv/bin/python scripts/build_gene_axis.py data/bundle/gene_names.csv
.venv/bin/python scripts/build_ensembl_map.py \
    --gencode data/annotation/gencode.v32.annotation.gtf.gz --write-axis
```

⚠️ **`pip install vcc` installs the wrong package** (Varnish Custom Counters).
It is `vcc-cli`.

---

## What the phases do

Each phase has an acceptance test that must pass before the next one is
trustworthy. Run them in order; each is resumable.

| # | Script | What it produces | Certified by |
|---|---|---|---|
| 0 | `phase0_acceptance.py` | the gene axis + the submission contract as code | itself (19 checks) |
| 1 | `phase1_data_report.py` | `data/data_report.md` — per-source stats | raw counts + controls verified |
| 2 | `phase2_harmonize.py` | every dataset on the 18,533-gene axis | `.var_names` identical + ordered |
| 3 | `phase3_render.py` | references rendered into the challenge regime | `phase3_acceptance.py` |
| 4 | `phase4_precompute.py` | `reference_effects`, `reference_de`, `nreal_table` | reproduces phase 3's spot-check |
| 5 | `phase5_certify.py` | the 0/1 scale for a line | baseline → 0.0, ceiling → ~1.78 |
| 6 | `phase6_harness.py` | a scored prediction for a held-out line | `context_mean` lands near 0 |

### The scale, and why it is the point

Raw metric values mean nothing alone. Every score is rescaled against two
anchors computed **per cell line**:

- **0** = the cell-context-mean baseline (paste the average response everywhere)
- **1** = a split-half replicate of the real experiment

So 0.36 means *36% of the way from trivial to as-good-as-rerunning-the-wet-lab*.
**1.0 is a landmark, not a maximum** — a perfect prediction beats a noisy
replicate on five of the six metrics. Measured ceiling here: **1.78**, with
`expr_mse_unbiased_capped_norm` capped at exactly 1.0 and the rest unbounded.

Because the anchors are per-line, a shallow reference does not read as a bad
predictor: 1.0 always means "as good as a replicate **of this line**".

---

## Scoring your own model

Implement one method. The harness handles splitting, anchors, and scoring.

```python
# src/vccjudge/harness.py defines the protocol
class MyPredictor:
    name = "my_model"

    def predict(self, controls, perturbations, rng):
        """controls: the held-out line's non-targeting cells, on the canonical axis.
        perturbations: the list to predict.
        Returns [(perturbation_name, matrix_of_400_cells), ...]

        You never see the held-out line's perturbed cells. That is the point.
        """
        ...
```

Then:

```bash
.venv/bin/python scripts/phase6_harness.py \
    --held-out vcc2025_h1/H1 --effects-from replogle2022/K562 \
    --predictor delta_transfer --all-perts
```

Two floor predictors ship with it — `context_mean` (defines the bottom) and
`delta_transfer` (borrow another line's measured effect). Beat `delta_transfer`
and you have something.

**Leakage is checked, not trusted.** `harness.leakage_check()` refuses to score a
predictor handed the held-out line's own measurements.

---

## Making a submission

```bash
.venv/bin/python scripts/make_submission.py -o sub.h5ad --compress gzip
.venv/bin/python -m vccjudge.contract sub.h5ad \
    --axis artifacts/gene_axis.parquet \
    --perts data/bundle/pert_counts.csv --contexts A,B,C     # must print PASS
.venv/bin/python scripts/make_vcc.py sub.h5ad -o prediction.vcc
vcc submit prediction.vcc -m "model name" --wait
```

⚠️ **Do not use `vcc prep`** to package. It needs ~61 GB of RAM for a
realistic-density submission and is OOM-killed on both a 22 GB laptop and a
31.3 GB Kaggle session. `make_vcc.py` produces an equivalent `.vcc` by
streaming — verified byte-identical `X` against prep's own output — but it
**skips Arc's validation**, so only ever run it on a file that already printed
`PASS` above.

---

## Running the heavy stages elsewhere

`scripts/kaggle/` submits jobs to Kaggle from the CLI (32 GB RAM, fast network,
free GPU). `scripts/aqua/` has PBS scripts for an HPC cluster. Both reduce raw
cells to small artifacts that come back over the wire; the judge itself runs on
a laptop.

```bash
.venv/bin/kaggle kernels push -p scripts/kaggle/replogle_render
.venv/bin/kaggle kernels status <user>/vcc-replogle-render
```

---

## If you are an agent working in this repo

**Building the model? Read `HANDOFF.md`** — it is the Phase 7 brief: what the
judge gives you, the training data on disk, the floor to beat, and the finding
that should drive the architecture (cross-line delta transfer gets direction
right only 53.7% of the time).

Read **`CLAUDE.md` first** — it is the non-negotiable contract, and it encodes
failures that already cost real time. In particular: `/tmp` is a RAM-backed
tmpfs on the dev machine, anything touching a multi-GB matrix runs under
`scripts/capped.sh`, and every submission rule is a rejection rather than a
repair. `PLAN.md` has the phase plan and the reasoning behind it.

---

## Current state

| Phase | Status |
|---|---|
| 0 contract + gene axis | ✅ certified (19 checks) |
| 1 raw sources | ✅ H1 + K562 landed |
| 2 harmonize | ✅ 99.978% of 1.93e9 values preserved |
| 3 regime renderer | ✅ certified — `nreal` 630 → 110 |
| 4 deltas + `nreal_table` | ✅ both lines |
| 5 scorer + anchors | ✅ certified — 0.0 and 1.78 |
| 6 harness | ✅ runs end-to-end, both floor predictors |
| 7 model tournament | not started |

### Corpus, and what it can and cannot measure

| | perturbations | on the 300-gene panel | regime |
|---|---|---|---|
| H1 (`vcc2025_h1`) | 150 | 13 | **in-regime** — 20,164 UMI, 400 cells |
| K562 (`replogle2022`) | 395 | 272 | **deficit** — 7,712 UMI, mostly <400 cells |
| shared (the transfer test set) | **136** | 13 | |

Things worth knowing before trusting a number out of this:

- **Leave-one-line-out and leave-one-dataset-out are the same split** right now,
  because each line *is* its whole dataset. PLAN.md wants the gap between them as
  an optimism correction; that needs a third line sharing a dataset with one of
  these. Current numbers are the pessimistic kind.
- **K562 is depth-deficient** — median `nreal` 4 vs H1's 190, and 46 of its
  perturbations register no detectable response at all. That is the regime, not
  biology: those genes visibly respond in H1.
- **24 of H1's 150 groups are near-essential knockdowns** (42% appear in
  Replogle's essential screen vs 20% baseline) with 12× the `nreal`. The 2026
  panel deliberately excludes that class. Treat them as a separate stratum;
  they are listed in `data/regime/*/​*.json` under `short_groups`.
- **28 of the 300 panel genes have no reference in either line.** Those need
  prediction from gene features, not transfer.

### Measured floors

Held out H1, predicted from K562:

| predictor | avg | reading |
|---|---|---|
| `context_mean` | −0.117 | predicting *no change* is slightly worse than the 0-anchor, which is the line's **mean perturbation response**, not the control profile |
| `delta_transfer` | **0.105** | borrowing K562's effects genuinely helps — `pds_cosine` 0.628 — but log-FC accuracy goes sharply negative |

The pattern to beat: transfer identifies **which** perturbation well while
getting the **magnitude** wrong (`lfc_nmae` −0.445) and expression accuracy
pinned at the floor (`mse` exactly 0.000).

### How far to trust a number from this

This has been checked against a real leaderboard submission, and the answer is
**good for ranking, rough for absolutes**. Read this before quoting a score.

The same predictor, scored three ways:

| metric | shared-only subset | **matched structure** | leaderboard |
|---|---|---|---|
| `pds_cosine` | 0.729 | **0.628** | 0.451 |
| `expr_mse_unbiased_capped_norm` | 0.000 | **0.000** | 0.000 |
| `de_wilcoxon_direction_fidelity` | +0.518 | **+0.365** | −0.172 |
| `de_wilcoxon_direction_reach` | −0.020 | **−0.022** | +0.098 |
| `de_wilcoxon_sig_jaccard` | 0.090 | **0.102** | −0.019 |
| `de_wilcoxon_lfc_nmae` | −0.219 | **−0.445** | +0.018 |
| **avg_score** | **0.183** | **0.105** | **0.063** |

**Score only the perturbations your effects source happens to cover and you will
flatter yourself.** Restricting to perturbations measured in both lines picks a
22x-easier subset — median `nreal` 178 against 8 for the rest. Passing
`--all-perts` includes the ~9% with no reference, which fall back to the control
profile and score ~0, exactly as the leaderboard treats the 28 panel genes no
dataset measures. That one change moved the score from 0.183 to 0.105 against a
leaderboard target of 0.063 — **65% of the gap, from experimental design alone**.

**Use `--all-perts` unless you have a specific reason not to.**

What this establishes:

- **Ranking is trustworthy.** `context_mean` < `delta_transfer` locally and on the
  leaderboard. If the harness says B beats A, believe it.
- **It responds correctly to design.** Matching the coverage ratio moved it
  two-thirds of the way to the real number — the behaviour of a calibrated
  instrument.
- **`pds` and `mse` track well.** 0.628 vs 0.451, and `mse` pinned at exactly
  0.000 in all three runs.

What it does **not** establish:

- **Absolute scores.** The residual 0.105 vs 0.063 rests on ONE held-out line and
  36 perturbations. Discount local gains accordingly.
- **`de_wilcoxon_direction_fidelity` is not yet reproduced.** It keeps the wrong
  sign (+0.365 local vs −0.172 leaderboard) even after the structural correction,
  so this is not a sampling artefact. The remaining variable is the target line:
  transfer appears to preserve DE direction into H1 but not into the challenge's
  contexts. That is the sharpest open question here, and it becomes testable the
  moment a third cell line exists.

Reproduce with:

```bash
.venv/bin/python scripts/phase6_harness.py \
    --held-out vcc2025_h1/H1 --effects-from replogle2022/K562 \
    --predictor delta_transfer --all-perts --limit-perts 36 --max-ntc 3000
```

⚠️ `--max-ntc` exists for tractability only: cell-eval2 densifies, and the
control arm rides into every comparison, so the full 18,400 controls with 150
perturbations exceeds a 22 GB machine. Anchors are only fully meaningful at full
control size — subsample for iteration, not for a number you intend to quote.
