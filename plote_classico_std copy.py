import os

import matplotlib.pyplot as plt
import numpy as np

from old.plote_classico import find_classical_configs
from src.config import NU
from src.functions import asymmetric_yerr, predict, reference_at_time, solve_burgers_fd
from src.models import ClassicalPINN2

CLASSICAL_DIR = "data/classical"
T_LABELS = ["0.1", "1"]
COLOR_MAE = "#1baf7a"


def compute_error_stats(t_dir):
    """Loads every classical model in `t_dir`, evaluates each against the FDM
    reference at t=Tfin, and returns per-INTRA_NEURO mean/std/min/max MAE
    across the run_id samples. Returns (intra_neurons, mae_mean, mae_std,
    mae_min, mae_max, n_samples, Tfin), one point per INTRA_NEURO.
    mae_min/mae_max (the extremes actually observed across run_ids) are what
    the plot uses for its error bars -- unlike mean +/- std, they can never
    go negative on the plot's log-scale y-axis."""
    entries = find_classical_configs(t_dir)
    if not entries:
        return [], [], [], [], [], [], None

    first_model = ClassicalPINN2.load(entries[0][2])
    L, Tfin = first_model.L, first_model.Tfin

    x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=NU)
    x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)
    u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)

    errors_by_neuron = {}
    for intra_neuron, _run_id, pt_path in entries:
        model = ClassicalPINN2.load(pt_path)
        u_pred = predict(model, x_plot, Tfin)
        entry = errors_by_neuron.setdefault(intra_neuron, [])
        entry.append(float(np.mean(np.abs(u_pred - u_ref))))

    intra_neurons = sorted(errors_by_neuron)
    mae_mean = [float(np.mean(errors_by_neuron[n])) for n in intra_neurons]
    mae_std = [float(np.std(errors_by_neuron[n])) for n in intra_neurons]
    mae_min = [float(np.min(errors_by_neuron[n])) for n in intra_neurons]
    mae_max = [float(np.max(errors_by_neuron[n])) for n in intra_neurons]
    n_samples = [len(errors_by_neuron[n]) for n in intra_neurons]
    return intra_neurons, mae_mean, mae_std, mae_min, mae_max, n_samples, Tfin


def plot_errors_std(intra_neurons, mae_mean, mae_min, mae_max, Tfin, save_path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    yerr = asymmetric_yerr(mae_mean, mae_min, mae_max)
    ax.errorbar(intra_neurons, mae_mean, yerr=yerr, fmt="o-", color=COLOR_MAE,
                linewidth=2.0, markersize=6, capsize=4, label="MAE (mean, [min,max])")
    ax.set_xlabel("INTRA_NEURO", fontsize=14)
    ax.set_ylabel("Error", fontsize=14)
    ax.set_yscale("log")
    ax.set_xticks(intra_neurons)
    ax.set_title(f"Classical PINN MAE vs INTRA_NEURO (T = {Tfin}, legacy data)", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for t_label in T_LABELS:
        t_dir = os.path.join(CLASSICAL_DIR, f"T_{t_label}")
        intra_neurons, mae_mean, mae_std, mae_min, mae_max, n_samples, Tfin = compute_error_stats(t_dir)
        if not intra_neurons:
            print(f"No classical models found in {t_dir}, skipping.")
            continue

        save_path = f"Error_std_vs_IntraNeuro_T_{t_label}_old.png"
        plot_errors_std(intra_neurons, mae_mean, mae_min, mae_max, Tfin, save_path)

        print(f"T = {Tfin}")
        for n, a_mean, a_std, a_min, a_max, n_s in zip(
            intra_neurons, mae_mean, mae_std, mae_min, mae_max, n_samples
        ):
            flag = "  <<< mean-std<0, plot uses [min,max] instead" if a_mean - a_std < 0 else ""
            print(f"  INTRA_NEURO={n:2d}  MAE={a_mean:.6e}+/-{a_std:.6e}  range=[{a_min:.6e}, {a_max:.6e}]  (n={n_s}){flag}")
        print(f"  -> saved {save_path}")
