"""Download the raw corpus ON KAGGLE, where the network is fast.

Pulling 8.2 GB took an hour on the laptop (~2-4 MB/s) and never finished; here
it is minutes. Output goes to /kaggle/working, which the runner turns into a
Dataset so every later kernel mounts it read-only at no working-dir cost.

⚠️ /kaggle/working is capped at ~20 GB. H1 (14.4 GB) and Replogle (8.2 GB)
together are 22.6 GB, so they are fetched in SEPARATE runs -- set WHICH below.
"""
import hashlib, os, subprocess, sys, time
from pathlib import Path

WHICH = os.environ.get("WHICH", "replogle")   # "replogle" | "h1"
OUT = Path("/kaggle/working"); OUT.mkdir(exist_ok=True)

SOURCES = {
    "replogle": [
        # scPerturb mirror: identical matrix to Figshare's 61.31 GB original,
        # gzip-compressed to 8.20 GB. Only the genome-wide screen -- the two
        # "essential" screens cover 0/300 challenge perturbations.
        ("ReplogleWeissman2022_K562_gwps.h5ad",
         "https://zenodo.org/api/records/13350497/files/ReplogleWeissman2022_K562_gwps.h5ad/content",
         "md5:13db594f8f1d2ccb88fec44a13e414dc"),
    ],
    "h1": [
        ("adata_Training.h5ad",
         "https://storage.googleapis.com/arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/train/adata_Training.h5ad",
         None),
        ("gene_names_2025.csv",
         "https://storage.googleapis.com/arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/gene_names.csv",
         None),
        ("pert_counts_Training.csv",
         "https://storage.googleapis.com/arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/train/pert_counts_Training.csv",
         None),
    ],
}

def md5(p, chunk=1 << 24):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

for name, url, checksum in SOURCES[WHICH]:
    dest = OUT / name
    t0 = time.time()
    print(f"[get] {name}", flush=True)
    subprocess.run(["curl", "-fL", "--retry", "5", "--retry-delay", "5",
                    "-o", str(dest), url], check=True)
    mb = dest.stat().st_size / 2**20
    print(f"  {mb:,.0f} MB in {time.time()-t0:.0f}s ({mb/max(time.time()-t0,1):.1f} MB/s)")
    if checksum and checksum.startswith("md5:"):
        got = md5(dest)
        want = checksum[4:]
        print(f"  md5 {'OK' if got == want else 'MISMATCH got=' + got}")
        if got != want:
            sys.exit(f"checksum mismatch on {name}")

print("\nfiles in /kaggle/working:")
for p in sorted(OUT.iterdir()):
    print(f"  {p.stat().st_size/2**20:10,.0f} MB  {p.name}")
