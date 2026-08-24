#!/usr/bin/env python
"""Resolve the challenge gene axis's symbols to stable Ensembl gene IDs.

CLAUDE.md makes Ensembl IDs the join key for every cross-dataset operation
("symbols drift") and gene_axis.py leaves `ensembl_id` null until something
fills it. This is that something.

Two sources, used for different jobs:

* **GENCODE** (`gene_name` -> `gene_id`) is the *primary*. The challenge axis
  comes from a 10x reference, and 10x references are built from GENCODE, so a
  GENCODE symbol match is the closest thing to reading Arc's own annotation.
* **HGNC** is the *drift handler*. It is the only source carrying
  `prev_symbol` and `alias_symbol`, which is what resolves an axis symbol that
  GENCODE has since renamed.

Resolution order per axis symbol, first hit wins (recorded in `source`):
  1. `gencode`      -- exact, unambiguous GENCODE gene_name
  2. `hgnc`         -- exact HGNC approved symbol
  3. `hgnc_prev`    -- HGNC previous symbol (the drift case)
  4. `hgnc_alias`   -- HGNC alias symbol (loosest; inspect these)

By default this writes only the mapping and a report. Pass --write-axis to
fill `artifacts/gene_axis.parquet`, which is deliberately a separate, explicit
step: a bad mapping silently corrupts every downstream join.
"""
from __future__ import annotations

import argparse
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vccjudge.gene_axis import axis_symbols, build_gene_axis, load_gene_axis  # noqa: E402


def parse_gencode(gtf_gz: Path) -> tuple[dict[str, str], dict[str, list[dict]], int]:
    """gene_name -> stable gene_id, plus the ambiguous ones and a PAR_Y count.

    Versions are stripped (`ENSG00000223972.5` -> `ENSG00000223972`) because the
    contract is a *stable* ID. `_PAR_Y` records duplicate their chrX twin under
    the same symbol; they are skipped so they cannot manufacture ambiguity.

    Ambiguous symbols keep their candidates' `level`, `hgnc_id`, `gene_type` and
    version, because GENCODE's own annotation quality fields are what settle
    them -- see `disambiguate`.
    """
    by_symbol: dict[str, list[dict]] = defaultdict(list)
    n_par_y = 0
    with gzip.open(gtf_gz, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t", 9)
            if len(f) < 9 or f[2] != "gene":
                continue
            attrs = f[8]
            gid = _attr(attrs, "gene_id")
            gname = _attr(attrs, "gene_name")
            if not gid or not gname:
                continue
            if gid.endswith("_PAR_Y"):
                n_par_y += 1
                continue
            stable, _, ver = gid.partition(".")
            lvl = _attr(attrs, "level") or _level(attrs)
            by_symbol[gname].append(
                {
                    "gene_id": stable,
                    "version": int(ver) if ver.isdigit() else 0,
                    "gene_type": _attr(attrs, "gene_type") or "",
                    "hgnc_id": _attr(attrs, "hgnc_id"),
                    "level": int(lvl) if lvl and lvl.isdigit() else 9,
                }
            )
    uniq, ambig = {}, {}
    for s, cands in by_symbol.items():
        ids = {c["gene_id"] for c in cands}
        if len(ids) == 1:
            uniq[s] = next(iter(ids))
        else:
            ambig[s] = cands
    return uniq, ambig, n_par_y


def _level(attrs: str) -> str | None:
    """`level` is unquoted in GTF (`level 1;`), unlike the quoted attributes."""
    i = attrs.find("level ")
    if i < 0:
        return None
    return attrs[i + 6 : attrs.index(";", i)].strip()


def load_reference_var(paths: list[str]) -> dict[str, str]:
    """symbol -> stable gene_id, read from a real dataset's `.var`.

    This outranks every heuristic below. A GTF says which gene is *canonically*
    named X; a dataset's `.var` says which gene the pipeline actually counted
    reads into for column X, which is the question a join key has to answer.
    Measured on H1: it agrees with the v32 parse on 18,073/18,077 symbols, and
    all four disagreements are symbols already known to be ambiguous in v32.
    """
    import anndata as ad

    out: dict[str, str] = {}
    for p in paths:
        a = ad.read_h5ad(p, backed="r")
        try:
            col = next((c for c in ("gene_id", "gene_ids", "ensembl_id") if c in a.var.columns), None)
            if col is None:
                raise KeyError(f"{p}: .var has no gene_id column (has {list(a.var.columns)})")
            for sym, gid in zip(map(str, a.var_names), a.var[col].astype(str)):
                out.setdefault(sym, gid.split(".", 1)[0])
        finally:
            if a.isbacked:
                a.file.close()
    return out


def disambiguate(cands: list[dict], hgnc_choice: str | None,
                 ref_choice: str | None = None) -> tuple[str, str]:
    """Pick one gene_id among same-symbol candidates.

    A real dataset's `.var` wins outright when it names one of the candidates --
    it is evidence about the reference build, not an opinion about nomenclature.
    Failing that, GENCODE's own quality fields decide.

    HGNC is consulted but never trusted blindly: it reports the *current*
    release's ID, which for SOD2 (`ENSG00000291237`) is not in v32 at all. An ID
    that does not exist in the source annotation cannot be the right answer for
    an axis built from it, so HGNC only breaks ties among real candidates.

    Order: dataset `.var` > curated (has hgnc_id) > better GENCODE `level` >
    HGNC's pick > established locus (higher version) > lowest ID for determinism.
    """
    ids = {c["gene_id"] for c in cands}
    if ref_choice and ref_choice in ids:
        return ref_choice, "reference_var"
    pool = [c for c in cands if c["hgnc_id"]] or cands
    best = min(c["level"] for c in pool)
    pool = [c for c in pool if c["level"] == best]
    if len(pool) == 1:
        return pool[0]["gene_id"], "gencode_curated"
    if hgnc_choice:
        for c in pool:
            if c["gene_id"] == hgnc_choice:
                return c["gene_id"], "gencode_hgnc_tiebreak"
    top = max(c["version"] for c in pool)
    pool2 = [c for c in pool if c["version"] == top]
    if len(pool2) == 1:
        return pool2[0]["gene_id"], "gencode_established"
    return sorted(c["gene_id"] for c in pool2)[0], "gencode_arbitrary"


def _attr(attrs: str, key: str) -> str | None:
    i = attrs.find(key + ' "')
    if i < 0:
        return None
    j = attrs.index('"', i + len(key) + 2)
    return attrs[i + len(key) + 2 : j]


def parse_hgnc(tsv: Path) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """approved-symbol, previous-symbol and alias-symbol maps to Ensembl IDs.

    Previous/alias columns are pipe-separated multi-values. A symbol claimed by
    more than one gene is dropped from that map rather than resolved arbitrarily
    -- an ambiguous drift match is worse than no match, because it is silent.
    """
    df = pd.read_csv(tsv, sep="\t", dtype=str, low_memory=False)
    df = df[df["ensembl_gene_id"].notna()]
    approved = dict(zip(df["symbol"], df["ensembl_gene_id"]))

    def _expand(col: str) -> dict[str, str]:
        claims: dict[str, set[str]] = defaultdict(set)
        sub = df[df[col].notna()]
        for names, eid in zip(sub[col], sub["ensembl_gene_id"]):
            for nm in str(names).strip('"').split("|"):
                nm = nm.strip()
                if nm:
                    claims[nm].add(eid)
        return {k: next(iter(v)) for k, v in claims.items() if len(v) == 1}

    return approved, _expand("prev_symbol"), _expand("alias_symbol")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", default="artifacts/gene_axis.parquet")
    ap.add_argument("--gencode", default="data/annotation/gencode.v50.basic.annotation.gtf.gz")
    ap.add_argument("--hgnc", default="data/annotation/hgnc_complete_set.txt")
    ap.add_argument("--out", default="artifacts/ensembl_map.csv")
    ap.add_argument("--report", default="artifacts/ensembl_map_report.md")
    ap.add_argument("--write-axis", action="store_true", help="also fill gene_axis.parquet")
    ap.add_argument(
        "--reference-var", action="append", default=[],
        help="h5ad whose .var carries gene_id; authoritative for ambiguous symbols. Repeatable.",
    )
    a = ap.parse_args()

    symbols = axis_symbols(a.axis)
    gencode, gc_ambig, n_par_y = parse_gencode(Path(a.gencode))
    hgnc, hgnc_prev, hgnc_alias = parse_hgnc(Path(a.hgnc))
    ref = load_reference_var(a.reference_var) if a.reference_var else {}
    if ref:
        conflict = sum(1 for s in symbols if s in gencode and s in ref and ref[s] != gencode[s])
        print(f"reference .var: {len(ref):,} symbols from {len(a.reference_var)} file(s); "
              f"{conflict} conflict with an UNambiguous gencode call "
              f"({'investigate — the axis may not be this reference build' if conflict else 'none, as expected'})")

    rows, misses, resolved_ambig = [], [], []
    for s in symbols:
        if s in gencode:
            rows.append((s, gencode[s], "gencode"))
            continue
        if s in gc_ambig:
            gid, why = disambiguate(gc_ambig[s], hgnc.get(s), ref.get(s))
            rows.append((s, gid, why))
            resolved_ambig.append((s, gid, why, gc_ambig[s], hgnc.get(s)))
            continue
        for src, table in (("hgnc", hgnc), ("hgnc_prev", hgnc_prev), ("hgnc_alias", hgnc_alias)):
            if s in table:
                rows.append((s, table[s], src))
                break
        else:
            rows.append((s, None, "unmapped"))
            misses.append(s)

    m = pd.DataFrame(rows, columns=["symbol", "ensembl_id", "source"])
    counts = m["source"].value_counts().to_dict()
    n_ok = int(m["ensembl_id"].notna().sum())

    # GENCODE and HGNC disagreeing on an approved symbol is a genuine signal, not noise.
    both = [(s, gencode[s], hgnc[s]) for s in symbols if s in gencode and s in hgnc and gencode[s] != hgnc[s]]
    dup = m[m["ensembl_id"].notna()]["ensembl_id"].duplicated(keep=False)
    collisions = m[m["ensembl_id"].notna()][dup].sort_values("ensembl_id")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(a.out, index=False)

    lines = [
        "# Ensembl mapping report",
        "",
        f"axis: `{a.axis}` ({len(symbols)} symbols)",
        f"gencode: `{Path(a.gencode).name}` ({len(gencode)} unambiguous symbols, "
        f"{len(gc_ambig)} ambiguous, {n_par_y} PAR_Y skipped)",
        f"hgnc: `{Path(a.hgnc).name}` ({len(hgnc)} approved, {len(hgnc_prev)} prev, {len(hgnc_alias)} alias)",
        "",
        f"**mapped {n_ok}/{len(symbols)} ({100*n_ok/len(symbols):.2f}%)**",
        "",
        "| source | n |",
        "|---|---|",
    ]
    for k, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{k}` | {n} |")
    lines += ["", f"## GENCODE/HGNC disagreements on an approved symbol: {len(both)}", ""]
    for s, g, h in both[:50]:
        lines.append(f"- `{s}`: gencode `{g}` vs hgnc `{h}`")
    lines += ["", f"## Symbols ambiguous in the source annotation, disambiguated: {len(resolved_ambig)}", ""]
    for s, gid, why, cands, hpick in resolved_ambig:
        lines.append(f"- `{s}` -> `{gid}` ({why}); HGNC said `{hpick}`")
        for c in cands:
            lines.append(
                f"    - `{c['gene_id']}` v{c['version']} level={c['level']} "
                f"{c['gene_type']} hgnc_id={c['hgnc_id']}"
            )
    lines += ["", f"## Two symbols sharing one Ensembl ID: {len(collisions)}", ""]
    for _, r in collisions.head(50).iterrows():
        lines.append(f"- `{r.symbol}` -> `{r.ensembl_id}` (via {r.source})")
    lines += ["", f"## Unmapped: {len(misses)}", ""]
    lines += [f"- `{s}`" for s in misses]
    Path(a.report).write_text("\n".join(lines) + "\n")

    print(f"mapped {n_ok}/{len(symbols)} ({100*n_ok/len(symbols):.2f}%)")
    for k, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:22s} {n:>6}")
    print(f"  gencode/hgnc disagreements: {len(both)}")
    print(f"  ensembl IDs claimed by >1 axis symbol: {len(collisions)}")
    print(f"wrote {a.out} and {a.report}")

    if a.write_axis:
        mapping = dict(zip(m["symbol"], m["ensembl_id"]))
        build_gene_axis(_symbols_csv(symbols), a.axis, ensembl_map=mapping)
        load_gene_axis(a.axis)
        print(f"filled ensembl_id in {a.axis}")
    else:
        print("(axis NOT modified; pass --write-axis to fill it)")
    return 0


def _symbols_csv(symbols: list[str]) -> Path:
    """build_gene_axis reads symbols from a CSV; round-trip the axis order via a temp file."""
    import tempfile

    p = Path(tempfile.mkstemp(suffix=".csv")[1])
    pd.Series(symbols).to_csv(p, index=False, header=False)
    return p


if __name__ == "__main__":
    raise SystemExit(main())
