#!/usr/bin/env bash
# Build the Python environment ON THE LOGIN NODE.
#
#     bash scripts/aqua/01_setup_env_login.sh
#
# This MUST run on a login node and MUST NOT be folded into the PBS job: Aqua's
# compute nodes have no outbound internet, so a `pip install` inside a job would
# fail after the scheduler has already spent your queue time. The venv is built
# once here, lives on shared storage, and every compute job just activates it.
#
# Pin the module name before first run -- `00_probe_login.sh` reports what is
# actually available:
#     PYTHON_MODULE=python/3.12.0 bash scripts/aqua/01_setup_env_login.sh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv-aqua}"
# Pinned from 00_probe_login.sh on Aqua (2026-08-24): the bare login shell has
# only python 2.6.6 and no conda, so a module load is mandatory. anaconda3_2024.10
# is the newest available and ships python 3.12, matching the repo's pin.
PYTHON_MODULE="${PYTHON_MODULE:-anaconda3_2024.10}"

echo "project: $PROJECT_DIR"
echo "venv:    $VENV_DIR"
echo "loading module: $PYTHON_MODULE"
# shellcheck disable=SC1091
module load "$PYTHON_MODULE"

PY="${PYTHON_BIN:-python3}"
command -v "$PY" >/dev/null 2>&1 || { echo "FATAL: no $PY after 'module load $PYTHON_MODULE'"; exit 1; }
echo "python:  $($PY --version 2>&1)"

# The repo pins 3.12 for laptop/Kaggle/Colab/Aqua parity (see README). 3.11+ works;
# below that, anndata/cell-eval2 will not install.
$PY - <<'EOF'
import sys
if sys.version_info < (3, 11):
    raise SystemExit(f"FATAL: need >=3.11, got {sys.version.split()[0]}. Load a newer python module.")
if sys.version_info[:2] != (3, 12):
    print(f"NOTE: {sys.version.split()[0]} — repo targets 3.12; fine, but pins may resolve differently.")
EOF

$PY -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel

# requirements-aqua.txt, NOT requirements.lock.txt. Aqua is RHEL 7.6 (glibc
# 2.17) and every compiled pin in the laptop lock needs glibc 2.24-2.28, so the
# lock cannot install here at all. See the header of requirements-aqua.txt.
#
# --only-binary=:all: makes pip fail fast on any package with no compatible
# wheel, instead of quietly attempting a source build of numpy against RHEL 7's
# toolchain -- which burns an hour and then fails anyway.
echo "installing from requirements-aqua.txt (glibc 2.17 ceilings)"
python -m pip install --only-binary=:all: -r "$PROJECT_DIR/requirements-aqua.txt"

# Editable install so `python -m vccjudge.contract` resolves from any cwd.
python -m pip install -e "$PROJECT_DIR" --no-deps

echo
echo "=== verifying (this is the last moment internet is available) ==="
python - <<'EOF'
import importlib.metadata as m, sys
print("  python  ", sys.version.split()[0])
for p in ("anndata", "scipy", "numpy", "pandas", "pyarrow", "h5py"):
    try:
        print(f"  {p:8s}", m.version(p))
    except Exception as e:
        raise SystemExit(f"FATAL: {p} missing — {e}")
import vccjudge.contract, vccjudge.loaders  # noqa: F401
print("  vccjudge imports OK")
EOF

# Record exactly what resolved. We cannot pre-compute an Aqua lock from a laptop
# (glibc constraints are not expressible in a requirements file), so the lock is
# captured here, after the fact, and committed for reproducibility.
python -m pip freeze > "$PROJECT_DIR/requirements-aqua.lock.txt"
echo "  recorded resolved versions -> requirements-aqua.lock.txt"

cat <<EOF

=== done ===
Activate in every job script with:
    source $VENV_DIR/bin/activate
$( [ -n "$PYTHON_MODULE" ] && echo "(after: module load $PYTHON_MODULE)" )

Next: bash scripts/aqua/02_fetch_login.sh   # also login-node only
EOF
