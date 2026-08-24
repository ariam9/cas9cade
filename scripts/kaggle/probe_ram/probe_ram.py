"""How much RAM and disk does a Kaggle TPU session actually have?

`vcc prep` was OOM-killed at 32 GB on a standard Kaggle CPU session, so the
question is whether the TPU tier's host VM is larger. Historically TPU VMs
carry far more host RAM than the accelerator tier suggests. One minute to find
out, versus 30 minutes to rebuild a submission and discover it again.
"""
import os, shutil, subprocess
print("=== memory ===")
print(subprocess.run(["free","-g"], capture_output=True, text=True).stdout)
with open("/proc/meminfo") as f:
    for line in f:
        if line.startswith(("MemTotal","MemAvailable","SwapTotal")):
            kb = int(line.split()[1]); print(f"  {line.split(':')[0]:14} {kb/2**20:8.1f} GB")
print("=== cpu ===")
print("  cores:", os.cpu_count())
print("=== disk ===")
for p in ("/kaggle/working", "/kaggle/temp", "/tmp", "/"):
    if os.path.isdir(p):
        t,u,f = shutil.disk_usage(p)
        print(f"  {p:16} {t/2**30:7.1f} GB total, {f/2**30:7.1f} GB free")
