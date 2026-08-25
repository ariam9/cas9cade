# CLAUDE.md — operating rules for this repo (VCC 2026)

You are helping build an **offline judge** for the Arc Virtual Cell Challenge 2026, then models that beat its floor. Read `PLAN.md` for the phased build. This file is the non-negotiable contract. **Obey it on every file you generate or modify.**

## What the task is
Zero-shot: predict CRISPRi single-gene knockdown responses in **cell lines never seen perturbed**, given **only their non-targeting control profiles** + a 300-gene list. No challenge training set. Scored by **`cell-eval2`** (`--preset vcc2026`) against held-out Arc data. Six metrics + a flat unweighted mean; a perfect submission scores ≈1.5, not 1.0 — 0 is the cell-context-mean baseline, 1 is a split-half replicate, and only `mse` is capped at 1.0. The `vcc2026` preset **is public** in `cell-eval2` (verified 0.16.0), so Phase 5 wraps it rather than reimplementing. Note `cell-eval` (v1, no "2") is the *2025* scorer — wrong axis, wrong metrics; do not use it.

## The submission contract — every item is a REJECTION, never a repair
A prediction `.h5ad` (one file, all contexts) must satisfy ALL of:
1. **Genes:** exactly the axis genes in `artifacts/gene_axis.parquet`, **in that exact order**, in `.var`.
2. **Perturbations:** exactly those in `pert_counts.csv` per context — none missing, none extra.
3. **No non-targeting / control rows.** Control cells are model **inputs only**, never emitted.
4. **Exactly 400 cells per (context, perturbation).**
5. **Raw counts in `.X`:** non-negative, integer-valued, finite. Never `normalize_total`/`log1p` a submission.
6. **Sparse, no explicitly-stored zeros** (`.eliminate_zeros()`); ≤ 4,750,000,000 stored entries in `.X`.
7. **≤ 1,000,000 total counts per cell.**
8. **Labels:** `target_gene` = gene **symbols** (`ADNP`, never `ADNP-1`); `context` copied verbatim from the control files.

**Hard rule:** no file is "ready" until this prints `PASS`:

```bash
python -m vccjudge.contract <file.h5ad> --axis artifacts/gene_axis.parquet \
    --perts pert_counts.csv --contexts <A,B,C>
```

This requires `uv pip install -e . --no-deps` once (the package lives under `src/`); see `README.md`. `--perts`/`--contexts` are **not optional** — without them items 2 and 8 above go unchecked, so the gate reports `INCOMPLETE` (exit 1) rather than `PASS`. `--allow-partial` downgrades that to `PARTIAL` for the dev loop; it never prints `PASS` and never clears a file for submission. Run the full form in CI and before every submission.

Then confirm against Arc's own validator, which checks the same things (gene set, contexts, perturbation labels, per-perturbation cell counts, raw counts, no control cells):

```bash
vcc prep prediction.h5ad -g gene_names.csv --perts pert_counts.csv --dry-run
```

`vccjudge.contract` is the fast local gate you run in CI and on every intermediate artifact; `vcc prep --dry-run` is the authoritative pre-flight before submitting. Neither replaces the other — if they ever disagree, **`vcc prep` is right and `vccjudge.contract` has a bug.**

Install is `uv tool install vcc-cli` (the command is `vcc`). ⚠️ Do **not** `pip install vcc` — that is an unrelated Varnish package. The submitted artifact is a `.vcc` file produced by `vcc prep`, not the raw `.h5ad`.

## Coordinate system
`artifacts/gene_axis.parquet` (built by `scripts/build_gene_axis.py` from the bundle's `gene_names.csv`) is the **single source of truth** for gene identity and order. Everything — harmonized data, predictions, submissions — lives on this axis. Cross-dataset joins resolve by **stable Ensembl ID**, not symbol (symbols drift); symbols are for display and for the final `.var_names` only.

## Engineering rules
- **Develop-small, run-big:** write and unit-test every stage on the small H1 slice locally, then run the *identical* code on Replogle on the Aqua cluster. The pinned axis + contract are what make the code identical across tiers.
- **Downsample before DE, always.** Any differential expression computed off non-challenge-regime data (≠ 400 cells / ~20k median UMI) is a bug — `nreal` depends on cell count and depth. **This governs the DE machinery only.** A pseudobulk delta is a mean vector, not DE: as a *transfer signal* it should be estimated at full depth, where H1 has 7.05x the total counts and therefore ~2.7x smaller standard error. Render in-regime for anything that decides significance (DE tables, `nreal`, the anchors, the held-out reference); keep full depth for effect-size estimates a model learns from. Compute both; do not conflate them.
- **Resumable + shardable:** heavy stages (Phase 3–4) are per-(cell line, perturbation) shards with per-shard checkpoints, so a killed batch job resumes.
- **Modality is not fungible:** CRISPRi vs CRISPRa/KO/chemical are tagged and never silently pooled.
- **Leakage:** any graph / co-essentiality / paralog features must be built from sources **excluding the held-out line/dataset**.
- **Everything seeded and cached.** Regime rendering, half-splits, subsampling are deterministic.
- Prefer stdlib + `anndata`/`scanpy`/`scipy.sparse`/`pandas`/`pyarrow`. Keep deps light so code runs on laptop, Kaggle, Colab, and Aqua unchanged.

## Resource discipline — this machine will OOM and take the session with it
- **`/tmp` is tmpfs (RAM-backed, 12 GB).** `df` reports it like a disk; it is not. All scratch goes to `data/` on `/mnt/data`, never `/tmp`.
- **Anything that loads a multi-GB matrix runs under `scripts/capped.sh`** (`bash scripts/capped.sh --mem 14G -- <cmd>`). A capped process dies alone; uncapped, the kernel OOM killer picks the session. This has already destroyed an in-flight competition upload once.
- **Never load a submission-sized matrix whole.** 360,000 x 18,533 is ~2.05e9 stored values = 15.3 GB minimum. `vccjudge.contract` and `phase2_harmonize` both stream for this reason; anything new that touches `.X` must too.
- **Verify environment facts before depending on them** — that a path exists, that a filesystem is what you think, that a remote job's state is what you assume. Every expensive failure so far came from asserting one of these instead of checking it.

## Compute placement (see PLAN.md table)
**Laptop:** all development and unit tests, the scorer + certification, the harness. Fast iteration lives here.
**Kaggle (via `kaggle` CLI, `scripts/kaggle/`):** anything data-heavy or GPU-bound — raw downloads, Phase 3–4 at scale, submission builds, Phase 7. 31.3 GB RAM, ~1 TB in `/kaggle/temp`, but `/kaggle/working` caps at 19.5 GB and sessions die at 12 h.
**Aqua (PBS, `scripts/aqua/`):** kept but no longer the backbone — no compute-node internet and an 80-core/10-job cap made it a poor fit for data-heavy work. Still fine for free CPU-parallel sweeps once the corpus is small.

**The raw cells are touched once (Phases 1–4 → small artifacts); the judge runs on those artifacts.** Develop-small/run-big is not a preference here: a bug that surfaces in seconds locally costs a 30-minute round trip on Kaggle.

## Definition of done for Phase 0
`python scripts/phase0_acceptance.py` prints `PASS — Phase 0 certified`. Do not start Phase 1 until it does.
