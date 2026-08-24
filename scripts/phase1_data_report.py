#!/usr/bin/env python
"""Phase 1 acceptance: profile every landed raw source and assert it is usable.

PLAN.md: "a `data_report.md` lists every (dataset, cell_line) with its stats,
and asserts each is raw counts with a non-empty NTC group. **Manually eyeball
it** -- this is where silent corruption enters. Any line lacking controls is
unusable for the harness; flag it now."

Two hard failures (exit 1), because both make a source unusable rather than
merely awkward:
  * `.X` is not raw counts -- someone normalized upstream, and every downstream
    DE number would be quietly wrong.
  * no non-targeting cells -- there is nothing to compute a delta against, so
    the line cannot enter the harness at all.

Everything else is reported for you to read, not enforced. In particular the
regime columns (median UMI, cells/pert) are *expected* to be far from the
challenge's 20,000 / 400 -- that gap is what Phase 3 exists to close, so a
source sitting at 54k UMI is a fact to carry forward, not a defect.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vccjudge.loaders import SOURCES, load_raw, profile_raw  # noqa: E402

CHALLENGE_UMI = 20_000
CHALLENGE_CELLS_PER_PERT = 400


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="data/data_report.md")
    ap.add_argument("--chunk-size", type=int, default=20_000)
    ap.add_argument("--only", default=None, help="dataset name to restrict to")
    a = ap.parse_args()

    rows, problems, missing = [], [], []
    for (dataset, cell_line), src in sorted(SOURCES.items()):
        if a.only and dataset != a.only:
            continue
        path = Path(a.root) / src.path
        if not path.exists():
            missing.append((dataset, cell_line, str(src.path)))
            print(f"[skip] {dataset}/{cell_line}: {src.path} not downloaded")
            continue
        print(f"[profile] {dataset}/{cell_line} ...", flush=True)
        adata = None
        try:
            adata = load_raw(dataset, cell_line, backed="r", root=a.root)
            st = profile_raw(adata, chunk_size=a.chunk_size)
        except OSError as e:
            # A download still in flight passes .exists() and then fails deep inside
            # h5py with "truncated file". That is not a data defect, so it must not
            # read as one -- but a genuinely corrupt file must not be waved through.
            msg = str(e)
            if "truncated" in msg.lower():
                got, want = _truncation_sizes(msg)
                missing.append((dataset, cell_line, f"{src.path} (INCOMPLETE: {got}/{want})"))
                print(f"    incomplete — {got}/{want}; still downloading? skipping")
                continue
            problems.append(f"{dataset}/{cell_line}: unreadable — {msg.splitlines()[0]}")
            print(f"    UNREADABLE — {msg.splitlines()[0]}")
            continue
        finally:
            if adata is not None and adata.isbacked:
                adata.file.close()
        st["notes"] = src.notes
        rows.append(st)
        if not st["raw_counts"]:
            problems.append(f"{dataset}/{cell_line}: .X is NOT raw counts "
                            f"(non_integer={st['non_integer']}, negative={st['negative']})")
        if not st["has_controls"]:
            problems.append(f"{dataset}/{cell_line}: NO non-targeting cells -- unusable for the harness")
        print(f"    {st['n_cells']:,} cells x {st['n_genes']:,} genes | "
              f"{st['n_perturbations']} perts | {st['n_ntc_cells']:,} NTC | "
              f"median UMI {st['median_umi']:,.0f}")

    out = Path(a.root) / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_render(rows, problems, missing))
    print(f"\nwrote {out}")

    if problems:
        print(f"\nFAIL — {len(problems)} blocking problem(s):")
        for p in problems:
            print("  -", p)
        return 1
    if not rows:
        print("\nFAIL — no sources profiled (nothing downloaded yet)")
        return 1
    print(f"\nPASS — Phase 1: {len(rows)} source(s) landed, raw counts and controls verified")
    return 0


def _truncation_sizes(msg: str) -> tuple[str, str]:
    """Pull the on-disk vs expected byte counts out of h5py's truncation error."""
    import re

    got = re.search(r"eof\s*=\s*(\d+)", msg)
    want = re.search(r"stored_eof\s*=\s*(\d+)", msg)
    fmt = lambda m: f"{int(m.group(1))/1073741824:.2f} GB" if m else "?"  # noqa: E731
    return fmt(got), fmt(want)


def _render(rows, problems, missing) -> str:
    L = ["# Phase 1 — raw data report", ""]
    if not rows:
        L += ["_No sources profiled._", ""]
    else:
        L += [
            "| dataset | line | modality | chemistry | cells | genes | perts | NTC cells |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            L.append(
                f"| {r['dataset']} | {r['cell_line']} | {r['modality']} | {r['chemistry']} "
                f"| {r['n_cells']:,} | {r['n_genes']:,} | {r['n_perturbations']} | {r['n_ntc_cells']:,} |"
            )
        L += ["", "## Measurement regime vs the challenge", "",
              f"The challenge regime is **{CHALLENGE_CELLS_PER_PERT} cells/pert at "
              f"~{CHALLENGE_UMI:,} median UMI**. Distance from it is what Phase 3 must close; "
              "DE computed before rendering is a bug (CLAUDE.md).", "",
              "| dataset | line | median UMI | x challenge | median cells/pert | x challenge | genes/cell |",
              "|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append(
                f"| {r['dataset']} | {r['cell_line']} | {r['median_umi']:,.0f} "
                f"| {r['median_umi']/CHALLENGE_UMI:.2f}x | {r['median_cells_per_pert']:,.0f} "
                f"| {r['median_cells_per_pert']/CHALLENGE_CELLS_PER_PERT:.2f}x "
                f"| {r['median_genes_per_cell']:,.0f} |"
            )
        L += ["", "## Raw-counts verification", "",
              "| dataset | line | raw counts | non-integer | negative | has controls |",
              "|---|---|---|---|---|---|"]
        for r in rows:
            L.append(
                f"| {r['dataset']} | {r['cell_line']} | {'✅' if r['raw_counts'] else '❌'} "
                f"| {r['non_integer']} | {r['negative']} | {'✅' if r['has_controls'] else '❌'} |"
            )
        L += ["", "## Notes", ""]
        for r in rows:
            if r.get("notes"):
                L.append(f"- **{r['dataset']}/{r['cell_line']}** — {r['notes']}")
    if missing:
        L += ["", "## Registered but not downloaded", ""]
        L += [f"- `{d}/{c}` — expected at `{p}`" for d, c, p in missing]
    L += ["", "## Blocking problems", ""]
    L += [f"- ❌ {p}" for p in problems] if problems else ["_None._"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
