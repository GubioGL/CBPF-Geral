"""Shared paths, style and data loaders for the manuscript figures.

Every figure script imports this module. From any directory:

    python manuscript/scripts/make_all.py

Two layers of data:
  data/              raw results, never sent to Overleaf
  manuscript/data/   the small tables the figures are drawn from (build_paper_data.py writes them)

Raw results behind the manuscript, grouped as in the paper:

  FORWARD PROBLEM, SIMULATION (same protocol, 50 seeds)
    data/Forward/CBPF_qubit_sweep_quantum_classical_fixed_validation_results/
        classical PINN vs noiseless HQPINN, n = 2..15, N_col = 10 and 100, T = 0.1 and 1
    data/Forward/CBPF_quantum_exact_results_*/     noiseless HQPINN vs N_col, n = 3, 5, 7, 10, 15
    data/Forward/classical_same_protocol_ncol/     classical PINN vs N_col, same protocol
  FORWARD PROBLEM, REAL QPU (one case per T, listed in QPU_CASES)
    data/quantum_forward_hardware_real/T_<T>/      training on ibm_berlin
    data/Forward/classical_vs_qpu/T_<T>/           classical control, same protocol as the QPU run
    data/derived/                                  replayed learning curves, same-budget simulations
  INVERSE PROBLEM, SIMULATION (33 paired runs per model and final time, listed in INVERSE_BUDGETS)
    data/Inverso_110/                              110 epochs, classical vs HQPINN on FakeBerlin   } deleted on 09/11/2026 (git
    data/Inverso_500/                              500 epochs, classical vs HQPINN on FakeAachen   } commit 201f0b3c6); their
                                                   tables stay in manuscript/data/inverse/
    data/Inverso/                                  20 000 epochs, classical vs HQPINN with the ideal circuit (Fig. 5, App. C)
  INVERSE PROBLEM, REAL QPU (listed in INVERSE_QPU_CASES): no run yet
    data/Inverso_real/<backend>/quantico/<case>/   where Inverser_quantum_hardware_pinn_E110.py --real writes
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
for _p in (ROOT / "run", ROOT / "src", ROOT / "src" / "quantum_hardware"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DATA = ROOT / "data"
FIG_DIR = ROOT / "manuscript" / "figures"
PAPER_DATA = ROOT / "manuscript" / "data"
CACHE_DIR = ROOT / "data" / "derived"

SWEEP_CSV = DATA / "Forward" / "CBPF_qubit_sweep_quantum_classical_fixed_validation_results" / "qubit_sweep_runs.csv"
# noiseless HQPINN vs number of collocation points: same protocol as the sweep, 50 seeds
EXACT_NCOL_DIRS = {
    3: DATA / "Forward" / "CBPF_quantum_exact_results_Brickwork3",
    5: DATA / "Forward" / "CBPF_quantum_exact_results__Brickwork5",
    7: DATA / "Forward" / "CBPF_quantum_exact_results_fixed_BrickWork7",
    10: DATA / "Forward" / "CBPF_quantum_exact_results_fixed_BrickWork10",
    15: DATA / "Forward" / "CBPF_quantum_exact_results_fixed_BrickWork15",
}
# Classical PINN at the N_col values missing from the sweep, same protocol: run_qubit_sweep.py --fixed-validation
# (run/run_classical_ncol.sh); build_paper_data.py adds it to Fig. 3.
CLASSICAL_NCOL_CSV = DATA / "Forward" / "classical_same_protocol_ncol" / "qubit_sweep_runs.csv"
REAL_DIR = DATA / "quantum_forward_hardware_real" / "T_0.1"
REAL_TAG = "10_NQ10_0_ibm_berlin_none_HW110"
FAKE_DIR = DATA / "quantum_ibm_faker" / "T_0.1"
FAKE_TAG = "10_NQ10_0_fake_berlin_HW110"
CVQ_DIR = DATA / "Forward" / "classical_vs_qpu" / "T_0.1"
COLS = 10                      # ibm_berlin: qubit q sits at row q // COLS, column q % COLS

# The one real-QPU case reported per horizon. The T = 1 entries are a proposal until that run exists:
# run_quantum_ibm_faker.py currently uses n = 10, N_col = 20 at T = 1. Edit them to match the run.
QPU_CASES = [
    dict(Tfin=0.1, backend="ibm_berlin", n=10, n_collocation=10, seed=0, epochs=110, shots=1024, mitigation="none",
         raw_dir=REAL_DIR, tag=REAL_TAG, classical_dir=CVQ_DIR,
         curves=CACHE_DIR / "learning_curves.npz", sims=CACHE_DIR / "same_budget_sims.npz"),
    dict(Tfin=1.0, backend="ibm_berlin", n=10, n_collocation=20, seed=0, epochs=110, shots=1024, mitigation="none",
         raw_dir=DATA / "quantum_forward_hardware_real" / "T_1.0", tag="20_NQ10_0_ibm_berlin_none_HW110",
         classical_dir=DATA / "Forward" / "classical_vs_qpu" / "T_1.0",
         curves=CACHE_DIR / "learning_curves_T1.npz", sims=CACHE_DIR / "same_budget_sims_T1.npz"),
]

# ---- inverse problem: nu = 0.01, N = 10 points per epoch, 3/7/10 qubits or neurons x seeds 0-10 --------
INV_NU = 0.01
INV_SIZES = (3, 7, 10)
INV_N = 10
INV_CASES = {0.1: "T_0.1_nu_0.01", 1.0: "T_1.0_nu_0.01"}
# The emulator changes together with the budget: the effect of the epochs on the HQPINN cannot be separated
# from the effect of the backend. The classical model has no backend, so its 110 -> 500 change is the epochs alone.
INVERSE_BUDGETS = {110: dict(root=DATA / "Inverso_110", backend="FakeBerlin"),
                   500: dict(root=DATA / "Inverso_500", backend="FakeAachen")}
# Long budget with the noiseless layer: same training, CONFIGs and adjustments, 20 000 epochs, ideal circuit
# (run/Inverser_runs_paralelo.py, preset ideal). The HQPINN runs are in quantico_ideal/.
INVERSE_IDEAL = dict(root=DATA / "Inverso", epochs=20000, backend="noiseless")
# Real-QPU inverse runs: none exists yet, every entry is a proposal. A case fills in by itself once its raw files
# are found. Keep them out of data/Inverso_110 (see --out-dir in Inverser_quantum_hardware_pinn_E110.py).
INVERSE_REAL_ROOT = DATA / "Inverso_real" / "ibm_berlin"
INVERSE_QPU_CASES = [
    dict(Tfin=0.1, backend="ibm_berlin", n=10, n_collocation=10, seed=0, epochs=110, shots=1024, mitigation="none"),
    dict(Tfin=1.0, backend="ibm_berlin", n=10, n_collocation=10, seed=0, epochs=110, shots=1024, mitigation="none"),
]

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
C_EMU = C_FAKE           # HQPINN on a noisy emulator (FakeBerlin, FakeAachen): aqua in the inverse figures too


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


def read_table(relative_path):
    """A table of manuscript/data/, e.g. read_table("forward/simulation/width_sweep.csv")."""
    import pandas as pd
    return pd.read_csv(PAPER_DATA / relative_path)


# ---- loaders -----------------------------------------------------------------------
def load_sweep():
    """Same-protocol sweep: one row per run (model_family, n_qubits, Tfin, n_collocation, seed)."""
    import pandas as pd
    return pd.read_csv(SWEEP_CSV)


def load_exact_ncol():
    """Noiseless HQPINN runs vs N_col for n = 3, 5, 7, 10, 15 qubits (50 seeds each)."""
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
