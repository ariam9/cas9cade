# VCC 2026 — Build Plan: The Judge (reference corpus + challenge-regime renderer + offline harness)

**What this plan produces:** a harmonized multi-cell-line reference library, a step that renders any reference into the challenge's exact measurement regime, a faithful local scorer, and a leave-one-dataset-out harness — i.e. a trustworthy offline judge that can score *any* idea against the real metric before you ever submit. **No model yet.** The model tournament comes after this exists, and depends on it.

**How to use this with Claude Code:** drop this file in the repo root as `PLAN.md` next to a `CLAUDE.md` that holds the submission contract (see Phase 0). Work one phase at a time. **Do not start a phase until the previous phase's acceptance test passes.** Each phase names the interface it must expose so the next phase can depend on it without knowing its internals.

---

## Assumptions I've baked in — correct me if wrong
1. **Stack:** Python, `scanpy`/`anndata`/`scipy.sparse`, `pandas`/`pyarrow`. Standard for single-cell; the plan assumes it throughout.
2. **Memory-safe by default:** the plan assumes a modest machine and uses backed/chunked AnnData and streaming ops so nothing requires loading a 2.5M-cell matrix into RAM at once. If you have a big-RAM box, some steps can be simplified (noted inline).
3. **Scorer:** ~~the plan tries to install Arc's `cell-eval` (`vcc2026` profile) first, and falls back to a faithful reimplementation~~ — **settled 2026-08-24:** the scorer is **`cell-eval2`** (`--preset vcc2026`), verified installable and public. No reimplementation needed. See the Phase 5 note.
4. **v1 corpus is small on purpose:** stand the whole judge up on 2–3 lines, fix the harmonization bugs cheaply, then expand. The plan is written so adding a dataset later is a config change, not a rewrite.

**Decisions locked:** v1 corpus = **H1 (VCC 2025) + Replogle K562 + RPE1** (3 lines, 2 datasets — enough to exercise both leave-one-line-out and leave-one-dataset-out from day one). Compute = tiered laptop / Aqua cluster / Kaggle+Colab (see next section).

---

## Compute placement (24GB laptop + Colab + Kaggle + IITM Aqua cluster)

**Governing principle:** the raw single-cell data is heavy but is touched exactly once. Phases 1–4 run on Aqua and reduce millions of raw cells to small derived artifacts (per-perturbation deltas, DE tables, `nreal`) — a few MB per line. Phases 5–6 (the judge) consume only those, so they run fine on the laptop. **After Phase 4 the RAM constraint is gone.** GPU only matters at Phase 7.

| Phase | Where | Why |
|---|---|---|
| 0 scaffold, contract, gene axis | **Laptop** | tiny; sets the axis/contract all tiers share |
| 1 acquire raw | **Aqua** (download onto cluster storage) | big disk + bandwidth; never pull genome-scale Replogle to the laptop. H1 also kept locally for dev |
| 2 harmonize | **Dev on laptop (H1)** → **run on Aqua (Replogle)** | identical code both tiers, guaranteed by the Phase-0 axis |
| 3 regime renderer | **Aqua**, sharded per (line, perturbation) | downsampling millions of cells; array-job friendly |
| 4 precompute deltas + DE | **Aqua**, batch array jobs | the tier boundary — outputs the small artifacts |
| 5 scorer + anchors | **Laptop** | operates on small in-regime artifacts; certify here |
| 6 harness + floor | **Laptop** | consumes derived artifacts, not raw cells |
| 7 model tournament | **Laptop** (light) + **Kaggle/Colab/Aqua-GPU** (neural) | GPU only enters here |

**Rules that follow:**
- **Develop-small, run-big:** unit-test every stage on H1 locally, submit identical code to Aqua for Replogle.
- **Resumable + shardable:** design Phases 3–4 as embarrassingly-parallel per (line, perturbation), checkpoint per shard, so a killed batch job resumes. ~~SLURM array jobs~~ — **corrected 2026-08-24: Aqua is PBS Pro**, so `qsub` / `-J 0-N` / `$PBS_ARRAY_INDEX`, not `sbatch`/`--array`. Scaffolding and the SLURM→PBS translation table are in `scripts/aqua/README.md`.
- **Aqua compute nodes have no outbound internet.** Every download — `pip install` included — is a login-node step; a job that tries to fetch fails only *after* the scheduler has spent your queue time. Acquisition (`02_fetch_login.sh`) and processing (`*.cmd`) are therefore permanently separate.
- **Derived artifacts are the interchange:** after Phase 4, publish harmonized + in-regime deltas/DE/`nreal` as a versioned Kaggle Dataset or cluster tarball; every downstream tier mounts that read-only.
- **RAM ordering:** Kaggle (~30GB) > laptop (24GB) > Colab free (~12GB); ephemeral envs (Colab/Kaggle) are for GPU bursts, not persistent state.

---

## Phase 0 — Repo scaffold + the contract as code
**Goal:** make the submission rules and the fixed coordinate system un-violable from day one.

**Build:**
- `CLAUDE.md` encoding the submission contract as hard rules: exactly 18,533 genes in `gene_names.csv` order; exactly 400 cells/perturbation/context; raw non-negative integer counts in `.X`; sparse, **no explicitly-stored zeros**; ≤ ~13,200 stored entries/cell (density cap); ≤ 1,000,000 counts/cell; gene-symbol labels (never construct IDs); context labels copied verbatim from the control files; **control cells are inputs only, never emitted**. Rule: "before any file is called ready, run `vcc prep --dry-run`."
- `config/datasets.yaml` — one entry per source dataset with fields: `name`, `path/uri`, `chemistry` (10x Flex / other), `modality` (CRISPRi / CRISPRa / KO / chemical), `cell_lines` (list), `raw_counts` (bool), `notes`. This file is the single source of truth for provenance.
- Pin the challenge gene axis: load `gene_names.csv` from the 2026 validation bundle (Arc bucket `gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/`), store as `artifacts/gene_axis.parquet` (symbol + resolved stable Ensembl ID + required order index).

**Interface exposed:** `gene_axis` (canonical 18,533 ordering), `datasets.yaml`.

**Acceptance test:** a throwaway random-count h5ad built against `gene_axis` passes `vcc prep --dry-run` with zero rejections. If `vcc` CLI isn't available yet, a local `validate_submission()` stub enforces every contract rule and passes the same file. **You cannot proceed until a trivially-shaped submission validates**, because every downstream artifact inherits this axis.

---

## Phase 1 — Acquire + land each dataset in a common raw schema
**Goal:** every source dataset on disk in an identical, inspectable layout, still in its *native* gene space and depth (no harmonization yet).

**Datasets (resolve exact accessions at build time — do not trust hard-coded URLs):**
- **H1 hESC** — VCC 2025 (Arc bucket). *Matched chemistry (10x Flex) + modality (CRISPRi) — your single most valuable reference.*
- **Replogle 2022** — genome-scale CRISPRi, K562 + RPE1 (GEO/Plus — resolve).
- **Nadig 2025** — CRISPRi essential screens, HepG2 + Jurkat.
- **Jiang 2025** — Perturb-seq, six cancer lines (A549, MCF7, HT29, HAP1, BxPC3, K562).
- (Defer Tahoe-100M: it's *chemical* perturbations, wrong modality for a knockdown transfer corpus. Tag and shelve, don't integrate.)

**Build:** per dataset, a loader that writes `data/raw/<name>/<cell_line>.h5ad` with a fixed `.obs` schema: `perturbation` (target gene symbol, or `non-targeting`), `cell_line`, `dataset`, `modality`, `chemistry`. Keep `.X` as raw counts. **Verify raw-ness** (integers, non-negative) and record per-file: n_cells, n_genes, median UMI/cell, n perturbations, n NTC cells.

**Interface exposed:** `load_raw(dataset, cell_line) -> AnnData` (backed).

**Acceptance test:** a `data_report.md` lists every (dataset, cell_line) with its stats, and asserts each is raw counts with a non-empty NTC group. **Manually eyeball it** — this is where silent corruption enters. Any line lacking controls is unusable for the harness; flag it now.

---

## Phase 2 — Harmonize to the challenge gene axis (Ensembl IDs, not symbols)
**Goal:** every dataset expressed on the identical 18,533-gene axis in the required order.

**Build:** map each dataset's genes to the canonical axis **by stable Ensembl gene ID**, not symbol (symbols are ambiguous and drift across annotations — this is a top silent-bias source). For each dataset record: how many axis genes were found, how many source genes were dropped, and how genes absent in the source are represented (explicit policy: absent → structural zero, flagged, *not* silently imputed). Reorder columns to the canonical index exactly.

**Interface exposed:** `data/harmonized/<name>/<cell_line>.h5ad`, all with identical `.var` = `gene_axis`.

**Acceptance test:** for every harmonized file, `.var_names` is **identical and identically ordered** to `gene_axis` (assert, don't hope). A `harmonization_report.md` shows per-dataset mapping/drop rates; a suspiciously high drop rate (e.g. >10–15%) halts the pipeline for inspection. Round-trip check: a known housekeeping gene (e.g. ACTB) lands at the same axis index across all datasets.

---

## Phase 3 — The challenge-regime renderer (the calibration step everything hinges on)
**Goal:** render any harmonized line into the challenge's exact measurement regime — **400 cells/group, ~20,000 median UMI/cell** — because `nreal` (how many genes test significant) is a function of cell count and depth, and your references sit at very different values. Skip this and four of six metrics are calibrated to the wrong regime.

**Build:** `to_challenge_regime(adata, seed) -> AnnData` that, per perturbation group and per NTC group: subsamples to 400 cells (record when a group has too few — that's a real limitation, not something to fabricate around), and downsamples counts per cell to ~20,000 median UMI via binomial thinning on the raw integer counts. Deterministic under `seed`.

**Interface exposed:** `to_challenge_regime()`; a cached `data/regime/<name>/<cell_line>__seed<k>.h5ad`.

**Acceptance test:** after rendering, median UMI/cell ≈ 20,000 and group sizes = 400 (or flagged short) across all lines. **Regime-sensitivity check:** compute the number of DE genes for a handful of perturbations *before* vs *after* rendering; confirm the after-distribution shifts toward the challenge regime and is stable across 2–3 seeds. This test is the proof that your local `nreal` will resemble the competition's.

---

## Phase 4 — Precompute reference deltas + DE tables (in-regime)
**Goal:** the artifacts your future transfer estimator and gate will actually consume — per (dataset, line, perturbation): a pseudobulk delta vs that line's NTC, and a DE table — computed **in the challenge regime from Phase 3**, using the **exact test the scorer uses** (two-sided Wilcoxon on per-cell CPM-normalized counts, low-expression filter = 5 CPM in control, BH FDR α=0.05 within-perturbation-after-filter, log2FC with ε=1e-9). Reuse the Phase 5 metric code so "reference DE" and "scored DE" are byte-for-byte the same routine.

**Interface exposed:** `reference_effects.parquet` (line × perturbation × gene delta) and `reference_de/` (per line×perturbation significant sets + signs + lfc). Also `nreal_table.parquet`: per (line, perturbation), how many genes responded — the training signal for the gate.

**Acceptance test:** for the H1 line, spot-check that a few well-known strong perturbations produce sizeable DE sets and weak ones don't; confirm the target gene's own row is excluded everywhere (contract). `nreal_table` distribution looks biological (a spread, not all-zero or all-huge).

---

## Phase 5 — The local scorer + the two anchors
**Goal:** reproduce the competition's scaled scoring so a local number *means* what a leaderboard number means.

> **RESOLVED 2026-08-24 — the fork below is settled; take the Preferred path.** The `vcc2026`
> preset **is** public, in **`cell-eval2` 0.16.0** (not `cell-eval`, which is the 2025 scorer).
> Two of the six metric names guessed below are wrong — the real set, from
> `cell_eval2.competition.competition_members()`, is `pds_cosine`,
> `expr_mse_unbiased_capped_norm`, `de_wilcoxon_lfc_nmae`,
> `de_wilcoxon_direction_fidelity_yield_raw`, `de_wilcoxon_direction_reach_raw`,
> `de_wilcoxon_sig_jaccard`. `BASE_SEED = 0` and `N_SPLITS = 5` are confirmed verbatim, and
> `prep-real-bundle` / `baseline` already implement the anchors. See `README.md` for the full
> constant dump. This deletes the reimplementation risk that made Phase 5 the plan's scariest step.

**Build:**
- **Preferred (confirmed available):** install `cell-eval2` and run `--preset vcc2026`; wrap it as `score(prediction, reference) -> per-metric raw + scaled`.
- **Fallback (no longer needed, keep only as a cross-check):** implement all six metrics from the briefing's Part III to its constants, including the panel-wide target-gene removal in PDS and the ρ-cap.
- **The anchors, per held-out line:** baseline `b` = the line's mean perturbation response assigned to every perturbation; replicate `r` = mean over **5 disjoint half-splits (base seed 0)**, each half scored against the other with its own controls. Scale: `s = (u − b)/(r − b)`; overall = unweighted mean over 6 metrics × contexts. ⚠️ The two arms use *different* control sources — `CONTROL_SOURCE_SCORED = 'real'` but `CONTROL_SOURCE_ANCHOR = 'pred'`; swapping them silently biases every scaled number.

**Interface exposed:** `score()`, and `compute_anchors(line) -> (b, r)` per metric.

**Acceptance test — the one that certifies the whole judge:** feed the scorer the **reference against itself** → scaled scores land near the ceilings in the briefing's cheat sheet (jac ≈ 2.5–2.8, mse = 1.0, etc.). Feed it the **context-mean baseline** → every metric scores ≈ 0. If those two calibration points don't reproduce, the scorer is wrong and nothing built on it can be trusted. Do not proceed past this test on faith.

---

## Phase 6 — The leave-one-dataset-out harness + a floor prediction
**Goal:** tie it together into the judge, and prove the loop end-to-end with the dumbest possible prediction.

**Build:**
- `harness(held_out, predictor)`: hides a held-out line's perturbation outcomes, exposes **only its NTC controls + the 300-gene list**, lets `predictor` emit 400 cells/perturbation, renders the held-out reference to challenge regime, scores with Phase 5, returns per-context scaled scores + overall.
- **Two hold-out modes:** leave-one-**line**-out *and* leave-one-**dataset**-out. The dataset-out mode is the honest one — the six blinded challenge lines share no batch/lab with your training data, so holding out a whole dataset estimates that gap. **Report both; the delta between them is your optimism correction.**
- **Floor predictor:** context-mean response (scores ~0 by construction) and a trivial transfer predictor (borrow the same gene's delta from the most basal-similar source line) — just enough to prove the pipeline emits a scoreable, contract-valid submission.

**Interface exposed:** `harness(held_out, predictor, seeds) -> scores`; `predictor` is the plug the model tournament fills.

**Acceptance test:** the floor predictor runs through the full harness on ≥2 held-out lines × ≥3 seeds, produces a **contract-valid** submission (passes Phase 0 validation), and returns finite per-metric scaled scores. The context-mean floor scores ≈ 0 overall — confirming the judge is centered where the real one is.

---

## Phase 7 — Leave the model-shaped hole (stub only)
**Goal:** make the tournament trivial to start once the judge is trusted.

**Build:** define the `Predictor` interface (`fit(source_lines)` optional, `predict(control_cells, gene_list) -> AnnData of 400×perturbation cells`), a `predictors/` registry, and a `run_tournament(predictors, held_outs, seeds)` that tabulates scaled scores per slot. Leakage guard: any graph/paralog features a predictor uses must be built from sources **excluding the held-out line/dataset** (assert this in the harness).

**Acceptance test:** two dummy predictors register and run through `run_tournament` producing a comparison table. Now the judge is done and idea-testing can begin.

---

## Cross-cutting guards (assert these everywhere, not once)
- **Downsample before DE, always.** Any DE computed off non-regime data is a bug.
- **Ensembl IDs for all cross-dataset joins.** Symbols only for final display labels.
- **Modality is not fungible.** CRISPRi and CRISPRa/KO/chemical are tagged and never silently pooled; cross-modality borrowing is a measured error source, not an assumption.
- **Cancer-line bias watch.** Most references are immortalized cancer lines; if a blinded challenge line is non-cancer/primary-like, a cancer-heavy harness is optimistic. Track per-held-out-line scores, never just the average.
- **Leakage.** Graph/essentiality features for the gate come from screens independent of the held-out line.
- **Everything seeded and cached.** Regime rendering, half-splits, subsampling — all deterministic and reproducible.

## Suggested order of first sessions
Phase 0 → 1 → 2 on **just H1 + Replogle** (fastest path to a running harness) → 3 → 5 (scorer + anchors, the certification) → 4 → 6 (floor loop). Expand the corpus (Nadig, Jiang) only after the floor loop is green on the small corpus — a loader bug found at 2 lines is cheap; at 10 it is not.
