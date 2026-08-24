# VCC 2026 — offline judge + models

See `PLAN.md` for the phased build and `CLAUDE.md` for the non-negotiable submission contract.

## Environment

Pinned to **Python 3.12** via `uv`, deliberately: `CLAUDE.md` requires identical code across
laptop / Kaggle / Colab / Aqua, and those tiers ship older interpreters than a rolling Arch box.
`requirements.txt` also caps `pandas<3` for the same reason — a bare `>=2.0` resolves to 3.0.x,
which changed the default string dtype and made copy-on-write unconditional.

```bash
uv venv --python 3.12 .venv         # .python-version pins this
uv pip install -r requirements.txt  # or: uv pip sync requirements.lock.txt
uv pip install -e . --no-deps       # so `python -m vccjudge.contract` resolves anywhere
```

The editable install is not optional: `vccjudge` lives under `src/`, and without it the readiness
gate below fails with `ModuleNotFoundError` from the repo root. `pyproject.toml` carries no
dependency list on purpose — `requirements.txt` remains the single source of truth for those.

`requirements.lock.txt` is the fully-pinned set (51 packages) for reproducing this environment
on Aqua/Kaggle/Colab. Regenerate with `uv pip compile requirements.txt -p 3.12 -o requirements.lock.txt`.

## Phase 0 — certified ✅

```bash
.venv/bin/python scripts/phase0_acceptance.py    # prints: PASS — Phase 0 certified
```

Verified green on **both** pandas 2.3.3 and 3.0.5, which is the evidence that the code crosses
the tier boundary rather than merely running here.

Once you have the real bundle:

```bash
.venv/bin/python scripts/build_gene_axis.py path/to/gene_names.csv
```

### The readiness gate

```bash
python -m vccjudge.contract <file.h5ad> --axis artifacts/gene_axis.parquet \
    --perts pert_counts.csv --contexts A,B,C
```

Four outcomes, and the word `PASS` is reserved for the case where **every** contract item was checked:

| Outcome | Exit | Meaning |
|---|---|---|
| `PASS` | 0 | all items checked, zero violations — the only state that clears a file |
| `FAIL` | 1 | at least one contract violation, listed |
| `INCOMPLETE` | 1 | `--perts`/`--contexts` absent, so items 2 and 8 went unchecked |
| `PARTIAL` | 0 | as `INCOMPLETE`, but `--allow-partial` given; dev loop only |

`validate_submission()` silently skips the perturbation-set and context checks when those
arguments are `None`. Before this was closed, the bare command in `CLAUDE.md` printed `PASS` on a
submission missing perturbations — a file Arc rejects outright. The CLI now refuses to.

## Phase 1 — H1 landed and verified ✅

```bash
.venv/bin/python scripts/phase1_data_report.py     # writes data/data_report.md
```

`PASS — 1 source landed, raw counts and controls verified`:

| dataset | line | modality | chemistry | cells | genes | perts | NTC cells |
|---|---|---|---|---|---|---|---|
| vcc2025_h1 | H1 | CRISPRi | 10x_flex | 221,273 | 18,080 | 150 | 38,176 |

`.obs` is `target_gene` / `guide_id` (189) / `batch` (48, e.g. `Flex_1_01` — chemistry confirmed
from the data, not assumed). Controls are already labelled `non-targeting`. `.X` is float32 raw
counts, verified integral and non-negative across all 221,273 cells.

**Off-regime by 2.7×**, measured on every cell rather than inferred from the pert_counts summary:

| | H1 | challenge | ratio |
|---|---|---|---|
| median UMI/cell | 53,912 | ~20,000 | **2.70×** |
| median cells/pert | 1,045 | 400 | **2.61×** |
| median genes/cell | 8,761 | ~6,000 | 1.4× |

Both are *surpluses*, which is the good direction — Phase 3 can subsample and binomial-thin down
to the challenge regime. A deficit could not be rendered at all.

`src/vccjudge/loaders.py` exposes `load_raw(dataset, cell_line) -> AnnData` (backed) with a uniform
`.obs` schema — `perturbation`, `cell_line`, `dataset`, `modality`, `chemistry` — and
`profile_raw()` for the per-source stats PLAN.md asks for. Control labels are normalized from each
source's dialect (`NTC`, `control`, …) to `non-targeting`.

**Deviation from PLAN.md, deliberate:** the schema is normalized *on read* rather than by rewriting
each source to `data/raw/<name>/<cell_line>.h5ad`. Landing H1 alone would copy 14.4 GB to change
nothing but four `.obs` columns — `.X` is already raw counts, and Phases 2–3 write new files
anyway. The uniform thing downstream sees is the interface, at 0 bytes instead of 30+ GB across the
corpus. Per-source mappings are declared in `SOURCES`, so they stay as inspectable as a landed file.

Two conditions fail the report (exit 1) because they make a source unusable: `.X` not raw counts,
and no non-targeting cells.

### ⚠️ H1 covers only 8.3% of the 2026 panel

`PLAN.md` calls H1 "your single most valuable reference" on matched chemistry (10x Flex) and
modality (CRISPRi). Measured against the real panel:

| | |
|---|---|
| 2026 panel | 300 perturbations |
| H1 (VCC 2025) total | 300 perturbations |
| **overlap** | **25 (8.3%)** — Training 13, Validation +4, Test +8 |
| 2026 panel genes with no H1 reference | **275** |

So H1 cannot supply per-gene deltas for 92% of the panel. Its value is calibration in matched
chemistry/regime, transferable CRISPRi-response structure, and 25 genes of direct validation.
**Replogle (genome-scale) is the primary source of per-gene deltas** — a reweighting of the corpus
plan that only became visible once the real `pert_counts.csv` was in hand.

H1 is also far off-regime — ~54,000 median UMI and ~1,050 cells/pert vs the challenge's ~20,000 and
400 — which is exactly the gap Phase 3's renderer exists to close.

## Phase 2 — H1 harmonized ✅

```bash
.venv/bin/python scripts/phase2_harmonize.py                 # writes data/harmonized/
.venv/bin/python scripts/phase2_harmonize.py --dry-run       # mapping stats only
.venv/bin/python scripts/phase2_harmonize.py --verify-only   # re-assert alignment
```

`data/harmonized/vcc2025_h1/H1.h5ad` — 221,273 × **18,533**, `.var_names` identical and identically
ordered to the axis.

| cells | source genes | mapped | structural zeros | source dropped | nnz kept |
|---|---|---|---|---|---|
| 221,273 | 18,080 | 18,077 | 456 (2.46%) | 3 (0.02%) | 1,932,128,341 of 1,932,554,688 (99.978%) |

Streamed in row blocks — 1.93e9 stored values never fit in RAM. Structural zeros are flagged in
`.var['measured_in_source']` and **never imputed**: a gene the assay did not measure is not a gene
measured at zero. The 15% halt threshold is enforced, not just reported.

Verified against the source rather than trusting the script: cell order and `.obs` preserved, and on
random cells every one of ~8,000 nonzero values lands at its correct axis index, with structural
zeros confirmed actually zero.

### The 3 dropped genes prove the reference-var fix was right

They are `TBCE-1`, `HSPA14-1`, `TMSB15B-1` — the *second* copies of symbols that name two loci,
which CellRanger disambiguates with a `-1` suffix. H1 carries both:

| | H1 bare symbol | H1 `-1` copy | 2026 axis uses |
|---|---|---|---|
| TBCE | `ENSG00000285053` | `ENSG00000284770` | **`ENSG00000285053`** ✅ |
| HSPA14 | `ENSG00000284024` | `ENSG00000187522` | **`ENSG00000284024`** ✅ |
| TMSB15B | `ENSG00000158427` | `ENSG00000269226` | **`ENSG00000158427`** ✅ |

`ENSG00000284770` and `ENSG00000187522` are exactly the IDs the HGNC heuristic had originally chosen
for `TBCE` and `HSPA14`. Keeping them would have pointed the axis's `TBCE` at the locus the reference
itself calls `TBCE-1` — a different gene, joined silently and wrongly for the entire project.

## Running on Aqua (PBS Pro)

See `scripts/aqua/README.md`. The governing constraint: **compute nodes have no outbound internet**,
so `01_setup_env_login.sh` (builds the venv) and `02_fetch_login.sh` (downloads via
`config/fetch_manifest.tsv`) are login-node steps, and only already-present files are touched by
`qsub`. Aqua is PBS Pro — `-J`, `$PBS_ARRAY_INDEX`, `-l select=…` — not SLURM.

## Upstream tooling — which package is which

Two similarly-named packages exist and **only one is the 2026 scorer.** Verified by installing
both in isolated venvs (2026-08).

| Package | Verdict |
|---|---|
| `cell-eval2` **0.16.0** | ✅ the 2026 scorer. `--preset vcc2026` |
| `cell-eval` 0.8.2 | ❌ the **2025** scorer. `EXPECTED_GENE_DIM = 18080`, 2025 metric names |
| `vcc-cli` (cmd `vcc`) | ✅ Arc's official download/validate/package/submit CLI |
| `vcc` on PyPI | ⚠️ **Varnish Custom Counters** — unrelated. Never install it |

### `cell-eval2` — the scorer (Phase 5 wraps it, does not reimplement)

`PLAN.md` Phase 5 offered "wrap Arc's scorer" or "reimplement six metrics". **The preferred path
is live**, which removes the largest risk in the plan. Confirmed directly from the package:

```python
from cell_eval2.competition import competition_members, derived_seeds
competition_members()   # the six scored metrics, in catalog order
```

- scored six: `pds_cosine`, `expr_mse_unbiased_capped_norm`, `de_wilcoxon_lfc_nmae`,
  `de_wilcoxon_direction_fidelity_yield_raw`, `de_wilcoxon_direction_reach_raw`,
  `de_wilcoxon_sig_jaccard` (plus 4 unscored diagnostics; 10 computed in total)
- `BASE_SEED = 0`, `N_SPLITS = 5` — exactly `PLAN.md`'s "5 disjoint half-splits, base seed 0"
- `BULK_TARGET_SUM = 50000.0`, comparator space `bulk_lognorm`, `input_type = 'counts'`
- `DEFAULT_PENALTY_CAP = 6.0` — this is the −6 floor on `nmae` the challenge docs quote
- `CONTROL_SOURCE_SCORED = 'real'` vs `CONTROL_SOURCE_ANCHOR = 'pred'` — the two arms use
  *different* control sources; getting this backwards silently biases the anchors
- subcommands: `run`, `prep-cache`, `prep-real-bundle`, `baseline`, `score` — `prep-real-bundle`
  and `baseline` are the Phase 5 anchors, already implemented upstream
- ⚠️ `cell-eval2` does **not** enforce the submission contract (no 18,533 / 4.75e9 / 1e6 checks).
  That is `vcc prep`'s job, and `vccjudge.contract`'s locally.

Phase 4 also benefits: `CLAUDE.md` demands reference DE and scored DE be "byte-for-byte the same
routine", which is now literally achievable by calling `cell_eval2.de_compute`.

### `vcc-cli` — the official gate

```bash
uv tool install vcc-cli     # command is `vcc`
vcc prep prediction.h5ad -g gene_names.csv --perts pert_counts.csv --dry-run
```

`vcc prep` validates gene set, contexts, perturbation labels, per-perturbation cell counts, raw
counts, and the no-control-cells rule — the same ground `vccjudge.contract` covers. Run ours in CI
and on intermediates; run `vcc prep --dry-run` before submitting. **If they disagree, `vcc prep` is
right.** Submissions are `.vcc` files, not raw `.h5ad`. `vcc sample` generates a throwaway valid
submission for pipeline testing, and `vcc skill install` installs an agent skill for Claude Code.

### The gene axis is fully resolved to Ensembl IDs

`artifacts/gene_axis.parquet` is **18,533/18,533 filled**, all `ENSG`-shaped, all unique, order
unchanged. Rebuild with:

```bash
.venv/bin/python scripts/build_ensembl_map.py \
    --gencode data/annotation/gencode.v32.annotation.gtf.gz --write-axis
```

**The source annotation is GENCODE v32** — established, not assumed. All 18,533 axis symbols are
present in v32 (100.00%) versus 17,974 (96.98%) in v50, which pins the axis to 10x's
`GRCh38-2020-A` reference. Mapping from the source means reading `gene_id` directly rather than
inferring it from a symbol, which is why every entry resolves.

| source | n | |
|---|---|---|
| `gencode` | 18,517 | unambiguous `gene_name` in v32 |
| `reference_var` | 15 | ambiguous in v32; resolved by H1's own `.var['gene_id']` |
| `gencode_hgnc_tiebreak` | 1 | `POLR2J3` — ambiguous and absent from H1's panel |
| unmapped | **0** | |

**A dataset's `.var` outranks every heuristic.** A GTF says which gene is
*canonically* named X; H1's `.var['gene_id']` says which gene Arc's pipeline
actually counted reads into for column X — and that is the question a join key
has to answer. Pass it with `--reference-var <h5ad>`.

The agreement is the evidence: H1 matches the v32 parse on **18,073/18,077**
symbols, and **all four disagreements are symbols already known to be ambiguous
in v32** (`TBCE`, `ZNF883`, `HSPA14`, `GGT1`). Zero conflicts on unambiguous
calls — which is also the strongest available evidence that the 2026 axis and
H1 are the same reference build, since the 2026 controls carry no `gene_id`
column to check directly. Adopting H1's four picks moved the Phase 2 join from
18,073 to **18,077**, exactly equal to the symbol overlap; keeping our heuristic
would have turned four real genes into spurious structural zeros.

Provenance (`data/annotation/`, sha256 recorded in `artifacts/ensembl_map_report.md`):
`gencode.v32.annotation.gtf.gz` (43 MB, the source), `hgnc_complete_set.txt` (16 MB, tie-breaks +
drift), `gencode.v50.basic.annotation.gtf.gz` (76 MB, kept only as the evidence for the v32 finding).

⚠️ **Do not "modernise" this to a current release.** HGNC and GENCODE v50 both report *current*
IDs, and for 10 of 12 disagreements the current ID does not exist in v32 at all. The clearest case
is `SOD2`: HGNC says `ENSG00000291237`, which is absent from v32 — the correct ID for this axis is
`ENSG00000112096` (`level 1`, `hgnc_id HGNC:11180`). An ID that never existed in the annotation the
data was counted against cannot be the right join key.

A first pass using HGNC + GENCODE v50 by symbol reached only 99.54%, needing `prev_symbol` for 469
genes and leaving 85 unmapped (75 of them clone-style placeholders like `AC118549.1`, which only
exist in one build). That number is the measure of how much symbol drift this axis carries — and
why `CLAUDE.md` forbids symbol joins.

### Control data profile (measured, all cells)

| ctx | cells | median UMI | mean UMI | median genes/cell |
|---|---|---|---|---|
| A | 18,400 | 20,109 | 21,134 | 6,147 |
| B | 18,400 | 19,946 | 19,990 | 5,756 |
| C | 18,400 | 20,034 | 21,157 | 6,006 |

- **Phase 3's regime target is confirmed empirically at ~20,000 median UMI.** Render references to
  this, not to a guess. (Measuring only the first 200 cells of a context gives ~18.5k — use all of them.)
- `.obs` is `target_gene` (all `non-targeting`), `context`, and **`ntc_id` — 46 distinct control
  guides per context**. Those 46 give a principled way to split controls into pseudo-replicates,
  which is what Phase 5's replicate anchor and any control-variability estimate want.
- Control `.var_names` are **identical and identically ordered to `gene_axis`** — Phase 2's
  acceptance criterion already holds for the challenge data. `.X` is sparse raw integer counts.
- ⚠️ `.var` has **no columns at all** — the bundle carries no Ensembl IDs. The axis's `ensembl_id`
  is still 0/18,533 filled, and Phase 2's cross-dataset joins need it from a pinned external
  annotation (GENCODE/Ensembl release), not from the bundle.
- ~6,000 genes/cell × 360,000 cells ≈ 2.2e9 stored entries, comfortably under the 4.75e9 cap.

## Challenge facts worth not re-deriving

- Controls bundle: `context_A.h5ad`, `context_B.h5ad`, `context_C.h5ad` + `gene_names.csv` +
  `pert_counts.csv`, all at the **zip root** (no `vcc_data/` dir). Final phase uses `D`/`E`/`F`.
- **18,400 non-targeting control cells per context.** Contexts are undisclosed cell lines; the
  control profile *is* the context.
- Never reorder or reassign context labels — the docs call this the most expensive available
  mistake, because a swap looks exactly like a weak model and nothing in the score says otherwise.
- Scale: 0 = cell-context-mean baseline, 1 = split-half replicate, `Overall` = unweighted mean of
  the six. Only `mse` is capped at 1.0; the other five are unbounded above (so >1 is real).
  Floors differ per metric: `mse` at 0, `nmae` at −6, `jac` ≈ −0.1, `fid` ≈ −1.9.
- Leaderboard short names: `pds`, `mse`, `nmae`, `fid`, `reach`, `jac`.

## Verified against the real bundle (2026-08-24)

`vcc datasets download controls` → 631.4 MiB, crc32c verified, in `data/bundle/`
(`context_{A,B,C}.h5ad`, `gene_names.csv`, `pert_counts.csv`, `manifest.json`).
`artifacts/gene_axis.parquet` is now built from the **real** `gene_names.csv`: 18,533 genes.

`manifest.json` confirms every constant `contract.py` had assumed — `n_genes: 18533`,
`pert_col: "target_gene"`, `context_col: "context"`, `control_label: "non-targeting"`,
`cells_per_pert: 400`, contexts `A/B/C`, 300 perturbations, 18,400 control cells per context,
`panel_id: "vcc2026-val-1"`.

The two former open questions are closed by the file itself: `pert_counts.csv` is a **one-column
list of 300 symbols** with no `n_cells` column, so reading `iloc[:, 0]` and hardcoding
`cells_per_pert = 400` are both correct. (Caveat for the final phase: `vcc prep` documents that an
`n_cells` column in `--perts` *would* override the 400 — re-check the D/E/F bundle.)

### Cross-certified against Arc's validator

`vcc sample` → a real 360,000 × 18,533 submission (= 300 × 400 × 3), run through both gates:

| Gate | Result | Time |
|---|---|---|
| `vcc prep --dry-run` (Arc, authoritative) | ✓ valid, `targets: verified against the official list` | 2.3s |
| `python -m vccjudge.contract` (ours) | `PASS (all items checked)` | 1.7s |

**A gap this comparison found and closed:** `vcc prep` enforces `--max-cell-dim 400000` and we did
not check total cell count at all. Now `MAX_TOTAL_CELLS = 400_000` in `contract.py`, with an
at-the-cap off-by-one check in the acceptance test. A complete submission is 360,000 cells, so this
is headroom rather than a live constraint — but it was a way for our gate to pass a file Arc rejects,
which is exactly the failure mode the gate exists to prevent.

## Known gaps (deliberately deferred)

- `tests/` is empty and the repo is not under git, so "run this in CI" has nothing to hang on yet.
- No axis checksum manifest — worth adding the moment Aqua enters at Phase 1, to prove every tier
  built the identical gene axis.
- `scanpy` is in `requirements.txt` but imported nowhere; it drags in ~40 transitive packages.
  Phase 4/5 DE is a hand-written Wilcoxon per `CLAUDE.md`, so it may never be needed — but
  cell-eval depends on it, so it likely arrives at Phase 5 regardless. Revisit then.
