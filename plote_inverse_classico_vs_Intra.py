"""Classical vs quantum inverse-problem PINN (nu estimation): mean
estimation error of nu vs INTRA_NEURO/N_QUBITS (architecture width), at a
fixed N_COLLOCATION. Mirrors `plote_classico_vs_Intra.py`'s classical/
quantum overlay convention, but reads checkpoints from
`data/classical_inverse/` and `data/quantum_inverse/` (written by
`Inverser_classical_pinn_paralelo.py` and `Inverser_quantum_pinn_paralelo.py`)
and plots the nu-estimation error instead of the solution MSE against the
FDM reference.

Outputs (one pair of figures per T_LABEL):
  NuError_std_vs_IntraNeuro_T_<Tfin>_NCOL<CLASSICAL_N_COLLOCATION>.png
"""
import os

import matplotlib.pyplot as plt

from main.functions import (
    compute_classical_inverse_nu_error_stats_vs_intra,
    compute_quantum_inverse_nu_error_stats_vs_intra,
    error_bar_yerr,
)

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex", "grid"])
except Exception:
    plt.style.use("seaborn-v0_8-whitegrid")

CLASSICAL_INVERSE_DIR = "data/classical_inverse"
QUANTUM_INVERSE_DIR = "data/quantum_inverse_exact"
T_LABELS = ["0.1", "1"]
CLASSICAL_N_COLLOCATION = 10  # only N_COLLOCATION with the full INTRA_NEURO sweep
QUANTUM_N_COLLOCATION = CLASSICAL_N_COLLOCATION  # also has the full N_QUBITS sweep
ERROR_BAR_TYPE = "variancia"
COLOR_MSE = "b"
QUANTUM_COLOR = "black"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})


def plot_nu_mse_vs_intra(intra_neurons, mse_mean, mse_std, mse_var, mse_min, mse_max,
                          quantum_stats, Tfin, save_path, error_type=ERROR_BAR_TYPE, color=COLOR_MSE):
    """quantum_stats: optional (n_qubits, mse_mean, mse_std, mse_var, mse_min,
    mse_max) tuple, plotted as a black curve (QUANTUM_COLOR) sharing the same
    x-axis as the classical curve, for direct comparison. Pass None to skip
    the quantum curve entirely."""
    error_labels = {"minimo_maximo": "min/max", "desvio_padrao": r"$\pm$std", "variancia": r"$\pm$var"}
    error_label = error_labels.get(error_type, error_type)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    yerr = error_bar_yerr(error_type, mse_mean, mse_std, mse_var, mse_min, mse_max)
    ax.errorbar(intra_neurons, mse_mean, yerr=yerr, fmt="o-", color=color,
                linewidth=2.0, markersize=6, capsize=4,
                label=fr"Classical ({error_label})")

    all_widths = set(intra_neurons)
    if quantum_stats is not None:
        q_neurons, q_mse_mean, q_mse_std, q_mse_var, q_mse_min, q_mse_max = quantum_stats
        q_yerr = error_bar_yerr(error_type, q_mse_mean, q_mse_std, q_mse_var, q_mse_min, q_mse_max)
        ax.errorbar(q_neurons, q_mse_mean, yerr=q_yerr, fmt="s--", color=QUANTUM_COLOR,
                    linewidth=2.0, markersize=6, capsize=4,
                    label=fr"Quantum ({error_label})")
        all_widths.update(q_neurons)

    ax.set_xlabel("N° of Neurons / Qubits", fontsize=14)
    ax.set_ylabel(r"MSE($\nu$)", fontsize=14)
    ax.set_yscale("log")
    ax.set_xticks(sorted(all_widths))
    ax.set_title(f"T = {Tfin}", fontsize=13)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for t_label in T_LABELS:
        t_dir = os.path.join(CLASSICAL_INVERSE_DIR, f"T_{t_label}")
        (intra_neurons, mse_mean, mse_std, mse_var, mse_min, mse_max, abs_mean, abs_std,
         abs_var, abs_min, abs_max, n_samples, Tfin) = compute_classical_inverse_nu_error_stats_vs_intra(
            t_dir, n_collocation=CLASSICAL_N_COLLOCATION
        )

        if not intra_neurons:
            print(f"No classical inverse models found in {t_dir} for N_COLLOCATION={CLASSICAL_N_COLLOCATION}, skipping.")
            continue

        quantum_t_dir = os.path.join(QUANTUM_INVERSE_DIR, f"T_{t_label}")
        (n_qubits, q_mse_mean, q_mse_std, q_mse_var, q_mse_min, q_mse_max, q_abs_mean, q_abs_std,
         q_abs_var, q_abs_min, q_abs_max, q_n_samples, q_Tfin) = compute_quantum_inverse_nu_error_stats_vs_intra(
            quantum_t_dir, n_collocation=QUANTUM_N_COLLOCATION
        )
        quantum_stats = None
        if n_qubits:
            quantum_stats = (n_qubits, q_mse_mean, q_mse_std, q_mse_var, q_mse_min, q_mse_max)
        else:
            print(f"No quantum inverse models found in {quantum_t_dir} for N_COLLOCATION={QUANTUM_N_COLLOCATION}, skipping quantum curve.")

        save_path = os.path.join(
            CLASSICAL_INVERSE_DIR,
            f"NuError_std_vs_IntraNeuro_T_{t_label}_NCOL{CLASSICAL_N_COLLOCATION}.png",
        )
        plot_nu_mse_vs_intra(
            intra_neurons, mse_mean, mse_std, mse_var, mse_min, mse_max, quantum_stats, Tfin, save_path,
            error_type=ERROR_BAR_TYPE,
        )
        print(f"  -> saved {save_path}  (n_samples per point: {n_samples})")
