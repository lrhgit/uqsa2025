# ============================================================
# Cell 1: Install chaospy (Colab-friendly)
# ============================================================
# --- cell: install_chaospy ---
# @title Install chaospy (Colab-friendly)

try:
    import chaospy as cp
    import numpoly
    import numpy as np
    print("chaospy er allerede installert.")
except ImportError:
    # Installer chaospy fra PyPI. Dette drar inn numpoly automatisk.
    %pip install chaospy==4.3.21 --no-cache-dir
    import chaospy as cp
    import numpoly
    import numpy as np

print("numpy  :", np.__version__)
print("numpoly:", numpoly.__version__)
print("chaospy:", cp.__version__)

# --- endcell: install_chaospy ---


# ============================================================
# Cell 2: Repo sync and environment setup
# ============================================================
# --- cell: repo_setup ---
# @title Repo sync and environment setup

import os
import sys
import subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
REMOTE = "https://github.com/lrhgit/uqsa2025.git"
REPO_PATH_COLAB = Path("/content/uqsa2025")

if IN_COLAB:
    if not REPO_PATH_COLAB.exists():
        print("Cloning repository...")
        subprocess.run(
            ["git", "clone", REMOTE, str(REPO_PATH_COLAB)],
            check=True
        )
    else:
        print("Updating existing repository...")
        subprocess.run(
            ["git", "-C", str(REPO_PATH_COLAB), "pull"],
            check=True
        )
    os.chdir(REPO_PATH_COLAB)

# --- Find repo root (works locally + in Colab) ---
cwd = Path.cwd().resolve()
repo_root = next(
    (p for p in [cwd] + list(cwd.parents) if (p / ".git").exists()),
    cwd
)

PY_SRC = repo_root / "python_source"
if PY_SRC.exists() and str(PY_SRC) not in sys.path:
    sys.path.insert(0, str(PY_SRC))

print("CWD:", Path.cwd())
print("repo_root:", repo_root)
print("python_source exists:", PY_SRC.exists())
print("python_source in sys.path:", str(PY_SRC) in sys.path)

# --- endcell: repo_setup ---


# ============================================================
# Cell 3: Layout fix, imports, NumPy–numpoly compatibility
# ============================================================
# --- cell: layout_and_numpy_patch ---
# @title Layout fix, imports, and NumPy compatibility patch

import warnings
warnings.filterwarnings("ignore")

from IPython.display import HTML

HTML("""
<style>
div.cell.code_cell, div.output {
    max-width: 100% !important;
}
</style>
""")

import numpy as np
import matplotlib.pyplot as plt
import chaospy as cp
import numpoly
import pandas as pd


# Pretty-print helpers (used across notebooks)
from pretty_printing import section_title, pretty_table, pretty_print_sobol_mc


# --- NumPy reshape compatibility patch for numpoly ---
_old_reshape = np.reshape

def _reshape_compat(a, *args, **kwargs):
    newshape = None
    if "newshape" in kwargs:
        newshape = kwargs.pop("newshape")
    if "shape" in kwargs and newshape is None:
        newshape = kwargs.pop("shape")
    if newshape is not None:
        return _old_reshape(a, newshape, *args, **kwargs)
    return _old_reshape(a, *args, **kwargs)

np.reshape = _reshape_compat
print("✓ numpy.reshape patched for numpoly compatibility")

# --- endcell: layout_and_numpy_patch ---
