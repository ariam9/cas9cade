#!/usr/bin/env python
"""Package a validated submission .h5ad into a .vcc without loading the matrix.

    python scripts/make_vcc.py submission.h5ad -o prediction.vcc

Why this exists: `vcc prep` needs ~61 GB for a realistic-density submission
(measured: 29.7 GB per 1e9 stored values; ours has 2.06e9). That exceeds a
22 GB laptop AND a 31.3 GB Kaggle session, so the official packager cannot run
on any machine we have.

But a .vcc turns out to be a POSIX tar containing `pred.h5ad.zst`, and prep
passes X through BYTE-IDENTICALLY -- verified on a real file: X/data,
X/indices and X/indptr all compare equal. The only transformation is cosmetic:
obs string columns become categorical, and the standard empty AnnData groups
are added. Both are metadata-only, so this repackages by streaming: obs is
rewritten in place (360k rows, trivial), then zstd and tar stream the bytes.
Peak memory is a few hundred MB regardless of matrix size.

⚠️ This SKIPS Arc's validation. Only ever run it on a file that has already
passed `python -m vccjudge.contract` with --perts and --contexts, which checks
the same rules and is cross-certified against `vcc prep --dry-run`.
"""
from __future__ import annotations
import argparse, shutil, subprocess, sys, tarfile, tempfile
from pathlib import Path

EMPTY_GROUPS = ("layers", "obsm", "obsp", "varm", "varp", "uns")


def categoricalize_obs(path: Path, cols=("target_gene", "context")) -> None:
    """Rewrite obs string columns as categorical, in place. X is never touched."""
    import h5py, numpy as np

    with h5py.File(path, "r+") as f:
        obs = f["obs"]
        for c in cols:
            if c not in obs:
                continue
            item = obs[c]
            if isinstance(item, h5py.Group):
                continue  # already categorical
            raw = item[:]
            vals = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in raw])
            cats, codes = np.unique(vals, return_inverse=True)
            del obs[c]
            g = obs.create_group(c)
            g.attrs["encoding-type"] = "categorical"
            g.attrs["encoding-version"] = "0.2.0"
            g.attrs["ordered"] = False
            g.create_dataset("categories", data=np.array(cats, dtype=object),
                             dtype=h5py.string_dtype())
            g.create_dataset("codes", data=codes.astype(np.int8 if len(cats) < 128 else np.int32))
            print(f"  obs/{c}: {len(cats)} categories")
        for name in EMPTY_GROUPS:
            if name not in f:
                g = f.create_group(name)
                g.attrs["encoding-type"] = "dict"
                g.attrs["encoding-version"] = "0.1.0"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("h5ad")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--level", type=int, default=3, help="zstd level (3 is prep-like and fast)")
    ap.add_argument("--keep-temp", action="store_true")
    a = ap.parse_args()

    src = Path(a.h5ad)
    if shutil.which("zstd") is None:
        sys.exit("FATAL: zstd not on PATH (apt-get install zstd / conda install zstd)")

    tmp = Path(tempfile.mkdtemp(prefix="vccpack-"))
    staged = tmp / "pred.h5ad"
    print(f"[1/4] copying {src} -> {staged}  ({src.stat().st_size/2**30:.1f} GB)")
    shutil.copy2(src, staged)          # streamed by the OS, not read into RAM

    print("[2/4] re-encoding obs as categorical (metadata only)")
    categoricalize_obs(staged)

    print(f"[3/4] zstd -{a.level} (streaming)")
    subprocess.run(["zstd", f"-{a.level}", "-T0", "-q", "--rm", "-f",
                    str(staged), "-o", str(tmp / "pred.h5ad.zst")], check=True)

    print(f"[4/4] tar -> {a.out}")
    with tarfile.open(a.out, "w") as t:
        t.add(tmp / "pred.h5ad.zst", arcname="pred.h5ad.zst")
    if not a.keep_temp:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {a.out}  ({Path(a.out).stat().st_size/2**20:,.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
