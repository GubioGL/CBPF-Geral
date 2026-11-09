"""Regenerate the tables and every manuscript figure (no QPU or Aer job is run).

    python manuscript/scripts/make_all.py                     # tables and figures, from the recorded data/derived/
    python manuscript/scripts/make_all.py --refresh-derived   # recompute data/derived/ first: see the warning below

1. build_paper_data.py     raw results -> manuscript/data/*, the only data that goes to Overleaf
                           (forward problem here, inverse problem in build_inverse_data.py)
2. the figure scripts      manuscript/data/ -> manuscript/figures/*.pdf

Then build the PDF with manuscript/build.sh.

WARNING. data/derived/ is a record of the author's Windows machine, not a cache. The training is chaotic at the level of
rounding errors (a finite-difference residual in single precision), so on another platform these results change. They
are recomputed only with --refresh-derived, and replay_training.py refuses to write if it does not reproduce the run:
  replay_training.py          offline replay of the recorded hardware run and of the classical control
                              -> data/derived/learning_curves.npz                       (~20 s)
  simulate_same_budget.py     HQPINN ideal / ideal + shot noise at the hardware budget
                              -> data/derived/same_budget_sims.npz                      (~10 s)
  controls_hardware_budget.py classical LR sweep and sine middle block at the hardware budget
                              -> data/derived/controls_hardware_budget.npz              (~40 s)
They need src/quantum_hardware/ (see CLAUDE.md).
"""

# --- repository layout: src/ and run/ go on the import path, and relative paths
# --- such as "data/..." are resolved from the repository root.
import os as _os
import sys as _sys
from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "src").is_dir() and (p / "run").is_dir())
for _p in (_ROOT / "run", _ROOT / "src", _ROOT / "src" / "quantum_hardware", _HERE):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
_os.chdir(_ROOT)
# ---------------------------------------------------------------------------------

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = ["replay_training.py", "simulate_same_budget.py", "controls_hardware_budget.py"]
TABLES = ["build_paper_data.py"]
FIGURES = ["fig2_sim_width.py", "fig3_sim_points.py", "fig4_qpu.py", "fig6_inv_training.py", "figC1_inv_long.py",
           "fig7_inv_qpu.py", "figB1_chip.py"]

def environment():
    """As run/env.sh: in a pip venv on Linux, qiskit_aer and torch need their own libgomp preloaded (otherwise: cannot
    allocate memory in static TLS block). The paths have to be absolute: the scripts run from this folder."""
    env = dict(os.environ)
    site = "lib/python3.*/site-packages"
    libgomp = [p for pattern in (f"{site}/qiskit_aer.libs/libgomp*.so*", f"{site}/torch/lib/libgomp.so*")
               for p in sorted(Path(sys.prefix).glob(pattern))[:1]]
    if libgomp:
        env["LD_PRELOAD"] = ":".join(str(p) for p in libgomp)
    return env


if __name__ == "__main__":
    if "--refresh-derived" in sys.argv:
        for name in CACHE:
            print(f"--- {name} --force", flush=True)
            subprocess.run([sys.executable, str(HERE / name), "--force"], check=True, cwd=HERE, env=environment())
    for name in TABLES + FIGURES:
        print(f"--- {name}", flush=True)
        subprocess.run([sys.executable, str(HERE / name)], check=True, cwd=HERE, env=environment())
