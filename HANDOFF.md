# Handoff — the judge is done, the model is not

**You are picking this up to build the model (Phase 7).** Phases 0–6 are
finished and certified: there is a working offline judge that scores a
prediction on the same scale the leaderboard uses. Do not rebuild it. Use it.

Read `CLAUDE.md` first (non-negotiable contract), then this. `README.md` has
setup and the phase-by-phase reference; `PLAN.md` has the original plan and where
reality diverged from it.

---

## 1. The task, precisely

Zero-shot. Predict the CRISPRi single-gene knockdown response for **300 genes**
in **three cell lines you have never seen perturbed** (`A`, `B`, `C`). For each
line you get **18,400 non-targeting control cells** and nothing else. There is no
challenge training set.

Emit **400 cells per (context, perturbation)** as raw integer counts on a fixed
18,533-gene axis. 360,000 cells total. Scored by six metrics, flat mean.

**0 = the cell-context-mean baseline. 1 = a split-half replicate of the real
experiment.** 1.0 is a landmark, not a maximum — measured ceiling is **1.78**.
Only `expr_mse_unbiased_capped_norm` is capped at 1.0.

---

## 2. What the judge gives you

```bash
.venv/bin/python scripts/phase6_harness.py \
    --held-out vcc2025_h1/H1 --effects-from replogle2022/K562 \
    --predictor delta_transfer --all-perts --limit-perts 36 --max-ntc 3000
```

Implement one method (`src/vccjudge/harness.py`):

```python
class MyPredictor:
    name = "my_model"
    def predict(self, controls, perturbations, rng):
        # controls: held-out line's NTC cells, sparse, on the canonical axis
        # returns [(perturbation_name, matrix_of_400_cells), ...]
```

The harness hides the line, builds its 0/1 anchors, scores, and refuses to run
if you were handed the held-out line's own measurements (`leakage_check`).

**Always pass `--all-perts`.** Without it you score only perturbations your
effects source covers — a 22× easier subset (median `nreal` 178 vs 8).

### How much to trust it

Calibrated against a real submission: **good for ranking, rough for absolutes.**
Matching the leaderboard's coverage structure moved the score 0.183 → 0.105
against a real 0.063 — 65% of the gap, from experimental design alone. Ranking
held (`context_mean` < `delta_transfer` both locally and on the leaderboard).
Absolute calibration rests on **one** held-out line; discount local gains.

---

## 3. Training data on disk

| artifact | shape | what it is |
|---|---|---|
| `reference_effects__replogle2022__K562.parquet` | 395 × 18,533 | CPM delta vs control, **full depth** |
| `reference_effects__vcc2025_h1__H1.parquet` | 150 × 18,533 | same, for H1 |
| `reference_de__*.parquet` | 3.0M / 1.6M rows | per-gene DE, in-regime, scorer's own routine |
| `nreal_table__*.parquet` | 395 / 150 | significant-gene count per perturbation |
| `data/regime/vcc2025_h1/H1__seed{0,1,2}.h5ad` | 73,953 cells each | in-regime cells, 3 seeds |
| `data/bundle/context_{A,B,C}.h5ad` | 18,400 × 18,533 | the controls you predict from |

Deltas are **full depth** (best effect estimate); DE is **in-regime** (400 cells,
~20k UMI). That split is deliberate — see `CLAUDE.md`.

### Panel coverage — this shapes the problem

| | of the 300 panel genes |
|---|---|
| K562 has a measured effect | **272 (91%)** |
| H1 has one | 13 |
| **neither** | **28 (9%)** |
| H1 ∩ K562 (transfer training pairs) | **136** |

So ~91% is **cross-context transfer** (you have the gene in K562, need it in an
unseen line) and ~9% is **cross-gene generalisation** (no reference anywhere —
needs gene features: pathway, complex, expression, essentiality).

---

## 4. The finding that should drive the architecture

**Naive delta transfer gets the DIRECTION of change right 53.7% of the time.**

On gene-perturbation pairs where both K562 and H1 move by >1 CPM (n = 550,874),
the sign agrees on **53.7%**. Per perturbation the median is **0.52**, and
**32 of 136 perturbations are worse than a coin flip**.

That single number explains the whole score profile:

| metric | floor predictor | reading |
|---|---|---|
| `pds_cosine` | **0.628** | *which* perturbation is identifiable — the pattern transfers |
| `de_wilcoxon_direction_fidelity` | +0.365 local, **−0.172 on the leaderboard** | direction does **not** transfer |
| `de_wilcoxon_lfc_nmae` | **−0.445** | magnitude is worse than the trivial baseline |
| `expr_mse_unbiased_capped_norm` | **0.000** | expression accuracy sits exactly on the floor |

**Copying another line's delta is close to uninformative about direction.** Any
architecture whose core move is "look up the effect in K562 and apply it" hits
this wall. The pattern of *which* genes respond carries across lines; the *sign*
and *magnitude* of the response largely do not.

The obvious hypothesis, untested: direction depends on the target line's own
baseline expression. A gene abundant in K562 and near-absent in H1 cannot move
the same way. You have the target's control profile — 18,400 cells of it — and
the floor predictor barely uses it (only to scale multiplicatively). That is the
most obvious source of unexploited signal.

### Where the headroom is

`mse` is pinned at **exactly 0.000** — expression accuracy is entirely
unexploited, and it is the one metric **capped at 1.0**, so it is bounded, safe,
and currently contributing nothing. `pds` at 0.628 is the one thing already
working; don't regress it.

---

## 5. The floor to beat

| predictor | avg (matched structure) |
|---|---|
| `context_mean` (predict no change) | −0.117 |
| `delta_transfer` (copy K562's effect) | **0.105** |
| the real submission, on the leaderboard | 0.063 |

Beat `delta_transfer` in the harness with `--all-perts` and you have something
worth submitting.

---

## 6. Corpus caveats that will bite you

- **K562 is depth-deficient** — 7,712 median UMI vs the challenge's ~20,000,
  mostly <400 cells/perturbation. Median `nreal` **4** vs H1's **190**; 46 of its
  perturbations show no detectable response at all. Its DE is low-powered by
  construction.
- **24 of H1's 150 perturbations are near-essential knockdowns** (42% appear in
  Replogle's essential screen vs 20% baseline) carrying **12× the `nreal`**. The
  2026 panel deliberately excludes essential genes — you cannot measure a
  transcriptional response in dead cells. Treat them as a separate stratum; they
  are listed under `short_groups` in `data/regime/*/*.json`.
- **Leave-one-line-out and leave-one-dataset-out are the same split** — each line
  *is* its whole dataset. The optimism correction `PLAN.md` wants needs a third
  line.
- **Only one held-out line exists.** Every absolute number rests on H1.

---

## 7. Practical constraints — read before running anything heavy

- **`/tmp` is tmpfs (RAM-backed, 12 GB)** on the dev laptop. Scratch goes to
  `data/` on `/mnt/data`. Writing 8.8 GB of test files to `/tmp` once killed the
  session and destroyed an in-flight upload.
- **Anything touching a multi-GB matrix runs under `scripts/capped.sh`**
  (`bash scripts/capped.sh --mem 14G -- <cmd>`). A capped process dies alone.
- **cell-eval2 densifies.** The 18,400-cell control arm rides into every scoring
  comparison; ~30k cells at full gene width exceeds a 22 GB machine. Use
  `--max-ntc` to iterate, full controls for a number you intend to quote.
- **Heavy work goes to Kaggle** (`scripts/kaggle/`, 32 GB, fast network, free
  GPU). Reduce there, bring back small artifacts. Do not try to move multi-GB
  files back — a 546 MB kernel output failed to transfer.
- **Package submissions with `scripts/make_vcc.py`, never `vcc prep`** — prep
  needs ~61 GB and is OOM-killed on every machine available.

---

## 8. Do not redo

The judge is certified and its acceptance tests pass. Specifically settled:

- the gene axis (18,533, Ensembl-resolved from GENCODE v32, validated against
  H1's own `var`)
- the submission contract, cross-checked against Arc's `vcc prep`
- the regime renderer — rendering is **load-bearing**, `nreal` falls 630 → 110
  between H1's native depth and the challenge's
- the 0/1 scale — baseline 0.0, ceiling 1.78, `mse` capped at exactly 1.0
- Replogle's two "essential" screens cover **0/300** panel genes and are
  deliberately not fetched

Your job is Phase 7: a `Predictor` that beats 0.105.
