"""Shared paths, style and data loaders for the manuscript figures.

Every figure script imports this module. Run the scripts with the conda env `qiskit`
(torch + qiskit) from any directory, e.g.

    C:/Users/Gubio/anaconda3/envs/qiskit/python.exe manuscript/scripts/make_all.py

Data layout (relative to the repository root), grouped as in the paper:

  SIMULATIONS
    data/Forward/CBPF_qubit_sweep_quantum_classical_fixed_validation_results/
        same-protocol sweep, classical MLP vs exact HQPINN, n = 2..15, 50 seeds   (Sec. IV A)
    data/Forward/CBPF_quantum_exact_results_*/          exact HQPINN vs N_col    (Sec. IV A)
    data/quantum_ibm_faker/T_0.1/                       FakeBerlin (noisy emulation)
  REAL QPU
    data/quantum_forward_hardware_real/T_0.1/           training on ibm_berlin
    data/quantum_ibm_faker/T_0.1/prediction_real_*.npz  FakeBerlin-trained, evaluated on ibm_berlin
    data/Forward/classical_vs_qpu/T_0.1/                classical control, same protocol as the QPU run
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "quantum_hardware") not in sys.path:
    sys.path.insert(0, str(ROOT / "quantum_hardware"))

DATA = ROOT / "data"
FIG_DIR = ROOT / "manuscript" / "figures"
CACHE_DIR = ROOT / "manuscript" / "cache"

SWEEP_CSV = DATA / "Forward" / "CBPF_qubit_sweep_quantum_classical_fixed_validation_results" / "qubit_sweep_runs.csv"
# exact HQPINN vs number of collocation points: 50-seed folders, same protocol as the sweep
EXACT_NCOL_DIRS = {
    3: DATA / "Forward" / "CBPF_quantum_exact_results_Brickwork3",
    5: DATA / "Forward" / "CBPF_quantum_exact_results__Brickwork5",
    7: DATA / "Forward" / "CBPF_quantum_exact_results_fixed_BrickWork7",
    10: DATA / "Forward" / "CBPF_quantum_exact_results_fixed_BrickWork10",
    15: DATA / "Forward" / "CBPF_quantum_exact_results_fixed_BrickWork15",
}
REAL_DIR = DATA / "quantum_forward_hardware_real" / "T_0.1"
REAL_TAG = "10_NQ10_0_ibm_berlin_none_HW110"
FAKE_DIR = DATA / "quantum_ibm_faker" / "T_0.1"
FAKE_TAG = "10_NQ10_0_fake_berlin_HW110"
CVQ_DIR = DATA / "Forward" / "classical_vs_qpu" / "T_0.1"
CVQ_SUMMARY = CVQ_DIR / "summary_10_NEURO110_tanh_seeds0-4_E110.json"

# ---- physics (forward problem) ---------------------------------------------------
L, NU = 1.0, 0.01
HX, HT = L / 512, 1.0e-3
TRIVIAL_MSE = {0.1: 3.93e-2, 1.0: 2.82e-1}     # u = sin(2 pi x): the "network learned nothing" level

# ---- figure geometry (APS: 3.375 in single column, 7.0 in double column) ----------
COL1, COL2 = 3.375, 7.0

# ---- colour roles: slots 1-3 of the reference categorical palette, fixed per entity --
INK = "#0b0b0b"          # text and the FDM reference
INK2 = "#52514e"         # secondary text
MUTED = "#8a8984"        # baselines, annotations
GRID = "#e6e5e1"         # hairline grid
SURFACE = "#ffffff"
C_CLASSICAL = "#2a78d6"  # classical PINN                      (slot 1, blue)
C_QPU = "#eb6834"        # HQPINN: real QPU / exact HQPINN      (slot 2, orange)
C_FAKE = "#1baf7a"       # HQPINN: FakeBerlin noisy emulation    (slot 3, aqua)
C_EXACT = C_QPU          # the noiseless HQPINN is the same model family -> same hue


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 7.5,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "lines.linewidth": 1.2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    })


def panel_label(ax, label, x=-0.16, y=1.04):
    """APS-style panel label, '(a)' in bold at the top left of the axes."""
    ax.text(x, y, f"({label})", transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="left", color=INK)


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"saved {FIG_DIR / name}.pdf/.png")


# ---- loaders -----------------------------------------------------------------------
def load_sweep():
    """Same-protocol sweep: one row per run (model_family, n_qubits, Tfin, n_collocation, seed)."""
    import pandas as pd
    return pd.read_csv(SWEEP_CSV)


def load_exact_ncol():
    """Exact HQPINN runs vs N_col for n = 3, 5, 7, 10, 15 qubits (50 seeds each)."""
    import pandas as pd
    frames = []
    for n, folder in EXACT_NCOL_DIRS.items():
        df = pd.read_csv(folder / "quantum_exact_runs.csv")
        df["n_qubits"] = n
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_npz(path):
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def load_json(path):
    return json.loads(Path(path).read_text())


def real_file(kind, ext="npz"):
    return REAL_DIR / f"{kind}_{REAL_TAG}.{ext}"


def fake_file(kind, ext="npz"):
    return FAKE_DIR / f"{kind}_{FAKE_TAG}.{ext}"


def mse(u, ref):
    return float(np.mean((np.ravel(u) - np.ravel(ref)) ** 2))


def rel_l2(u, ref):
    u, ref = np.ravel(u), np.ravel(ref)
    return float(np.linalg.norm(u - ref) / np.linalg.norm(ref))


def quantiles(values, q=(0.25, 0.5, 0.75)):
    return np.quantile(np.asarray(values, float), q)
